import argparse
import json
from pathlib import Path
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
from torch.profiler import schedule as torch_prof_schedule, ProfilerActivity, tensorboard_trace_handler
import torch

from open_clip import create_model_and_transforms

from .data_module import EchoCSVDataModule
from .model import EchoClipLinearProbe
from ..models import resolve_model_id


def _parse_devices(value):
    value = str(value).strip()
    if value in ("auto", "", "none"):
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 0
        return max(count, 1) if count > 0 else 1
    if value.startswith("["):
        return json.loads(value)
    try:
        return int(value)
    except Exception:
        return value


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)

    model_id = resolve_model_id(args.model_id or args.model)
    _, _, preprocess_val = create_model_and_transforms(
        model_id,
        precision=args.precision,
        device="cpu",
    )

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
        drop_last=True,
        random_single_frame=args.random_single_frame,
    )

    model = EchoClipLinearProbe(
        task=args.task,
        num_classes=args.num_classes,
        model_id=args.model_id or args.model,
        precision=args.precision,
        lr=args.lr,
        weight_decay=args.wd,
        freeze_backbone=True,
        loss=args.loss,
        huber_delta=args.huber_delta,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True,
        check_finite=True,
    )
    ckpt_dir = Path(args.out_dir) / args.exp_name
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_loss:.4f}",
        save_last=True,
        auto_insert_metric_name=False,
        dirpath=str(ckpt_dir),
    )

    loggers = []
    if not args.no_csv_logger and not args.disable_logging:
        loggers.append(CSVLogger(save_dir=args.out_dir, name=args.exp_name))
    logger = False
    if not args.disable_logging:
        if len(loggers) == 1:
            logger = loggers[0]
        elif len(loggers) > 1:
            logger = loggers

    lr_monitor = None
    if logger is not False:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

    profiler = None
    if args.profiler != "none":
        if args.profiler == "simple":
            profiler = SimpleProfiler(dirpath=args.out_dir, filename=f"profiler-{args.exp_name}")
        elif args.profiler == "advanced":
            profiler = AdvancedProfiler(dirpath=args.out_dir, filename=f"profiler-{args.exp_name}")
        elif args.profiler == "pytorch":
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available() and not args.prof_cpu_only:
                activities.append(ProfilerActivity.CUDA)
            profiler = PyTorchProfiler(
                dirpath=args.out_dir,
                filename=f"pt-profiler-{args.exp_name}",
                activities=activities,
                schedule=torch_prof_schedule(
                    wait=args.prof_wait,
                    warmup=args.prof_warmup,
                    active=args.prof_active,
                    repeat=args.prof_repeat,
                ),
                on_trace_ready=tensorboard_trace_handler(args.prof_tb_dir or args.out_dir),
                record_shapes=args.prof_record_shapes,
                profile_memory=args.prof_profile_memory,
                with_stack=args.prof_with_stack,
                with_flops=True,
            )

    class BatchTimeCallback(pl.Callback):
        def __init__(self):
            self._start = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self._start = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._start is None:
                return
            duration = time.perf_counter() - self._start
            batch_size = None
            try:
                if isinstance(batch, (list, tuple)) and hasattr(batch[0], "shape"):
                    batch_size = int(batch[0].shape[0])
            except Exception:
                batch_size = None
            pl_module.log("train_batch_time", duration, prog_bar=False, on_step=True, batch_size=(batch_size or 1))

    callbacks = [early_stop, checkpoint]
    if lr_monitor is not None:
        callbacks.append(lr_monitor)
    if args.enable_batch_timing:
        callbacks.append(BatchTimeCallback())

    strategy = args.strategy
    if strategy in ("auto", "", None):
        devices = _parse_devices(args.devices)
        if (isinstance(devices, int) and devices > 1) or (isinstance(devices, (list, tuple)) and len(devices) > 1):
            strategy = "ddp_find_unused_parameters_false"
        else:
            strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=_parse_devices(args.devices),
        strategy=strategy,
        precision="bf16-mixed" if "bf16" in args.precision else ("16-mixed" if "fp16" in args.precision else 32),
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        deterministic=False,
        num_sanity_val_steps=args.num_sanity_val_steps,
        profiler=profiler,
    )

    trainer.fit(model, datamodule, ckpt_path=args.resume if args.resume else None)
    best_path = checkpoint.best_model_path if checkpoint.best_model_path else None
    trainer.test(model=model if not best_path else None, datamodule=datamodule, ckpt_path=best_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a linear probe on CLIP embeddings")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--key_frame_col", default="Frame")
    parser.add_argument("--label_col", default="CalcValue")
    parser.add_argument("--view", default=None)
    parser.add_argument("--modality", default=None)
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--random_single_frame", action="store_true")

    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--model", default="echo_clip")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--loss", choices=["huber", "mse"], default="mse")
    parser.add_argument("--huber_delta", type=float, default=1.0)

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="auto")

    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--exp_name", default="echo_probe")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="echo-clip")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--no_csv_logger", action="store_true")
    parser.add_argument("--disable_logging", action="store_true")

    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--profiler", choices=["none", "simple", "advanced", "pytorch"], default="none")
    parser.add_argument("--prof_cpu_only", action="store_true")
    parser.add_argument("--prof_wait", type=int, default=1)
    parser.add_argument("--prof_warmup", type=int, default=1)
    parser.add_argument("--prof_active", type=int, default=3)
    parser.add_argument("--prof_repeat", type=int, default=1)
    parser.add_argument("--prof_tb_dir", default=None)
    parser.add_argument("--prof_record_shapes", action="store_true")
    parser.add_argument("--prof_profile_memory", action="store_true")
    parser.add_argument("--prof_with_stack", action="store_true")
    parser.add_argument("--enable_batch_timing", action="store_true")
    return parser


def entrypoint():
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()


__all__ = ["build_arg_parser", "entrypoint", "main"]
