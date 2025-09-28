import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
import re

import torch
import torch.nn.functional as F

from open_clip import create_model_and_transforms

from .data_module import EchoCSVDataModule
from .model import EchoClipLinearProbe
from ..models import resolve_model_id


def _default_out_csv(ckpt_path: Path) -> Path:
    ckpt_dir = ckpt_path.parent
    return ckpt_dir / "predictions_test.csv"


def _first_frame_index(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() == 0:
            return None
        return int(flat[0].item())
    if isinstance(value, (list, tuple)):
        queue = [value]
        while queue:
            item = queue.pop(0)
            if isinstance(item, (list, tuple)):
                queue = list(item) + queue
            else:
                try:
                    return int(item)
                except Exception:
                    break
        return None
    if np is not None and isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return int(flat[0]) if flat.size > 0 else None
    if isinstance(value, str):
        matches = re.findall(r"\d+", value)
        return int(matches[0]) if matches else None
    try:
        return int(value)
    except Exception:
        return None


def _rows_from_batch(
    preds: torch.Tensor,
    meta_batch: List[Dict[str, Any]],
    task: str,
    num_classes: int,
    save_all_probs: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    task = task.lower()

    if task == "regression":
        for idx, meta in enumerate(meta_batch):
            base = {
                "path": meta.get("path"),
                "video_id": meta.get("video_id"),
                "key_frame": meta.get("key_frame"),
                "view": meta.get("view"),
            }
            frame_idx = _first_frame_index(meta.get("frame_indices"))
            if frame_idx is not None:
                base["frame_index"] = frame_idx
            if meta.get("label") is not None:
                base["label"] = meta.get("label")
            base["pred"] = float(preds[idx].detach().cpu().item())
            rows.append(base)
        return rows

    if num_classes == 1:
        probs = preds.sigmoid().view(-1)
        for idx, meta in enumerate(meta_batch):
            base = {
                "path": meta.get("path"),
                "video_id": meta.get("video_id"),
                "key_frame": meta.get("key_frame"),
                "view": meta.get("view"),
            }
            frame_idx = _first_frame_index(meta.get("frame_indices"))
            if frame_idx is not None:
                base["frame_index"] = frame_idx
            if meta.get("label") is not None:
                base["label"] = meta.get("label")
            prob = float(probs[idx].detach().cpu().item())
            logit = float(preds[idx].detach().cpu().item())
            base.update(
                {
                    "logit": logit,
                    "prob": prob,
                    "prob_0": float(1.0 - prob),
                    "prob_1": prob,
                    "pred": int(prob >= 0.5),
                }
            )
            rows.append(base)
        return rows

    probs = F.softmax(preds, dim=-1)
    top1 = probs.argmax(dim=-1)
    for idx, meta in enumerate(meta_batch):
        base = {
            "path": meta.get("path"),
            "video_id": meta.get("video_id"),
            "key_frame": meta.get("key_frame"),
            "view": meta.get("view"),
        }
        frame_idx = _first_frame_index(meta.get("frame_indices"))
        if frame_idx is not None:
            base["frame_index"] = frame_idx
        if meta.get("label") is not None:
            base["label"] = meta.get("label")
        base["pred"] = int(top1[idx].detach().cpu().item())
        if save_all_probs:
            for cls, value in enumerate(probs[idx].detach().cpu().tolist()):
                base[f"prob_{cls}"] = float(value)
        rows.append(base)
    return rows


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    model_arg = args.model_id or args.model or hparams.get("model_id", "hf-hub:mkaichristensen/echo-clip")
    model_id = resolve_model_id(model_arg)
    precision = args.precision or hparams.get("precision", "bf16")

    _, _, preprocess_val = create_model_and_transforms(model_id, precision=precision, device="cpu")

    datamodule = EchoCSVDataModule(
        csv_path=args.csv,
        preprocess_val=preprocess_val,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        res=(args.res, args.res),
        max_frames=args.max_frames,
        stride=args.stride,
        default_key_frame=0,
        view=args.view,
        modality=args.modality,
        key_frame_col=args.key_frame_col,
        label_col=args.label_col,
        drop_last=False,
        random_single_frame=False,
    )
    datamodule.setup(stage="predict")
    dataloader = datamodule.predict_dataloader()
    if dataloader is None:
        raise RuntimeError("No predict dataloader available. Ensure split=='test' exists.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model: EchoClipLinearProbe = EchoClipLinearProbe.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    task = getattr(model.hparams, "task", "regression")
    num_classes = int(getattr(model.hparams, "num_classes", 1))

    all_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise RuntimeError("Expected batch=(video, meta)")
            video, meta = batch
            video = video.to(device, non_blocking=True)
            preds = model(video)

            if isinstance(meta, dict):
                bs = int(video.shape[0])
                meta_list: List[Dict[str, Any]] = []

                def pick_i(value, i):
                    if isinstance(value, torch.Tensor):
                        tensor = value[i]
                        try:
                            return tensor.item()
                        except Exception:
                            return tensor
                    if isinstance(value, list):
                        if len(value) == bs:
                            return value[i]
                        if len(value) > 0 and isinstance(value[0], (list, tuple)) and all(len(v) == bs for v in value):
                            return [v[i] for v in value]
                        return value
                    if isinstance(value, tuple):
                        if len(value) == bs:
                            return value[i]
                        return value
                    return value

                for i in range(bs):
                    meta_list.append({k: pick_i(v, i) for k, v in meta.items()})
            else:
                meta_list = list(meta)

            rows = _rows_from_batch(preds, meta_list, task=task, num_classes=num_classes, save_all_probs=args.save_all_probs)
            all_rows.extend(rows)

    out_csv = Path(args.out_csv) if args.out_csv else _default_out_csv(ckpt_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = []
    seen = set()
    for row in all_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} predictions to {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained linear probe on the test split and save CSV predictions")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--key_frame_col", default="Frame")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--modality", default=None)
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--save_all_probs", action="store_true")
    return parser


def entrypoint():
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()


__all__ = ["build_arg_parser", "entrypoint", "main"]
