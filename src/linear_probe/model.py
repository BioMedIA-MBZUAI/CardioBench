import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from open_clip import create_model_and_transforms
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from ..models import resolve_model_id


class EchoClipLinearProbe(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "hf-hub:mkaichristensen/echo-clip",
        precision: str = "bf16",
        task: str = "regression",
        num_classes: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        freeze_backbone: bool = True,
        loss: str = "huber",
        huber_delta: float = 1.0,
        log_confusion: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        resolved_id = resolve_model_id(model_id)
        model, _, _ = create_model_and_transforms(resolved_id, precision=precision, device="cpu")
        self.backbone = model.eval()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        feat_dim = getattr(self.backbone.visual, "output_dim", 512)
        self.task = task.lower()
        if self.task == "regression":
            self.head = nn.Linear(feat_dim, 1)
            self.reg_use_huber = (loss.lower() == "huber")
            self.huber_delta = float(huber_delta)
        else:
            self.head = nn.Linear(feat_dim, num_classes)
            if num_classes == 1:
                self.cls_loss = nn.BCEWithLogitsLoss()
            else:
                self.cls_loss = nn.CrossEntropyLoss()

        self._sk_buf = {
            "val": {"preds": [], "targets": []},
            "test": {"preds": [], "targets": []},
        }

    def _reset_stage_buffers(self, stage: str):
        if stage in self._sk_buf:
            self._sk_buf[stage]["preds"].clear()
            self._sk_buf[stage]["targets"].clear()

    def _accumulate_for_stage(self, preds: torch.Tensor, targets: torch.Tensor, stage: str):
        if self.task != "classification" or stage not in self._sk_buf:
            return
        self._sk_buf[stage]["preds"].append(preds.detach().view(-1).cpu().numpy())
        self._sk_buf[stage]["targets"].append(targets.detach().view(-1).cpu().numpy())

    def encode_batch(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        flat = video.reshape(B * T, C, H, W)
        prec = str(self.hparams.precision).lower()
        use_bf16 = ("bf16" in prec)
        use_fp16 = ("fp16" in prec) or ("16" == prec)
        dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        with torch.autocast(device_type=self.device.type, dtype=dtype if dtype != torch.float32 else None, enabled=dtype != torch.float32):
            feats = self.backbone.encode_image(flat)
        pooled = feats.view(B, T, -1).mean(1)
        return F.normalize(pooled, dim=-1).float()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        emb = self.encode_batch(video)
        out = self.head(emb)
        if self.task == "regression":
            return out.squeeze(1)
        return out

    @staticmethod
    def _extract_labels(meta):
        if isinstance(meta, dict) and "label" in meta:
            return meta["label"]
        if isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
            return [m.get("label") for m in meta]
        raise ValueError("Could not extract 'label' from meta.")

    def _step(self, batch, stage="train"):
        video, meta = batch
        y_list = self._extract_labels(meta)
        y = torch.as_tensor(y_list, device=self.device)
        bsize = video.shape[0]

        logits = self(video)
        if self.task == "regression":
            y = y.float()
            if getattr(self, "reg_use_huber", False):
                loss = F.huber_loss(logits, y, delta=self.huber_delta)
            else:
                loss = F.mse_loss(logits, y)
            mae = torch.mean(torch.abs(logits - y))
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bsize, sync_dist=(stage != "train"))
            self.log(f"{stage}_mae", mae, prog_bar=(stage != "train"), batch_size=bsize, sync_dist=(stage != "train"))
            return loss
        if self.hparams.num_classes == 1:
            y = y.float().view(-1, 1)
            loss = self.cls_loss(logits, y)
            preds = (logits.sigmoid() > 0.5).int().view(-1)
            targets = y.int().view(-1)
        else:
            y = y.long()
            loss = self.cls_loss(logits, y)
            preds = logits.argmax(dim=-1)
            targets = y
        acc = (preds == targets).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bsize, sync_dist=(stage != "train"))
        self.log(f"{stage}_acc", acc, prog_bar=(stage == "train"), batch_size=bsize, sync_dist=(stage != "train"))
        if stage in ("val", "test"):
            self._accumulate_for_stage(preds, targets, stage)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_validation_epoch_start(self):
        self._reset_stage_buffers("val")

    def on_test_epoch_start(self):
        self._reset_stage_buffers("test")

    def on_validation_epoch_end(self):
        if self.task != "classification":
            return
        preds_list = self._sk_buf["val"]["preds"]
        targs_list = self._sk_buf["val"]["targets"]
        if not preds_list or not targs_list:
            return
        y_pred = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(targs_list, axis=0)
        avg = "binary" if int(self.hparams.num_classes) == 1 else "macro"
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_prec", prec, prog_bar=False, sync_dist=True)
        self.log("val_rec", rec, prog_bar=False, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        if bool(getattr(self.hparams, "log_confusion", False)):
            try:
                is_zero = getattr(self.trainer, "is_global_zero", True)
            except Exception:
                is_zero = True
            if is_zero:
                cm = confusion_matrix(y_true, y_pred)
                rep = classification_report(y_true, y_pred, zero_division=0)
                print("[val] confusion_matrix:\n", cm)
                print("[val] classification_report:\n", rep)

    def on_test_epoch_end(self):
        if self.task != "classification":
            return
        preds_list = self._sk_buf["test"]["preds"]
        targs_list = self._sk_buf["test"]["targets"]
        if not preds_list or not targs_list:
            return
        y_pred = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(targs_list, axis=0)
        avg = "binary" if int(self.hparams.num_classes) == 1 else "macro"
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)
        self.log("test_prec", prec, prog_bar=False, sync_dist=True)
        self.log("test_rec", rec, prog_bar=False, sync_dist=True)
        self.log("test_f1", f1, prog_bar=True, sync_dist=True)
        if bool(getattr(self.hparams, "log_confusion", False)):
            try:
                is_zero = getattr(self.trainer, "is_global_zero", True)
            except Exception:
                is_zero = True
            if is_zero:
                cm = confusion_matrix(y_true, y_pred)
                rep = classification_report(y_true, y_pred, zero_division=0)
                print("[test] confusion_matrix:\n", cm)
                print("[test] classification_report:\n", rep)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
