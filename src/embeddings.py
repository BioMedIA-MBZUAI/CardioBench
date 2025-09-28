from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv

import torch
import torch.nn.functional as F

from .datasets import DatasetLoader, DatasetItem
from .models import ModelConfig, load_model
from .video import read_video, preprocess_frames, encode_video_clip_batched, indices_after_keyframe


@dataclass
class EmbeddingConfig:
    model: str = "echo_clip"             # alias or model id
    device: str = "cuda"
    precision: str = "bf16"
    res: int = 224
    max_frames: int = 16
    stride: int = 1
    batch_size: int = 64
    use_channels_last: bool = True
    pin_memory: bool = True
    overwrite: bool = False
    key_frame: int = 0
    normalize: bool = True


def _select_video_id(dataset: str, item: DatasetItem) -> str:
    metadata = item.metadata or {}
    if "unique_id" in metadata and metadata["unique_id"] not in (None, ""):
        return str(metadata["unique_id"]).strip()
    if dataset == "tmed2_csv":
        return item.path.name
    return item.path.stem


def _save_embedding(
    out_path: Path,
    video_id: str,
    item: DatasetItem,
    per_frame: torch.Tensor,
    sel_indices: List[int],
    *,
    normalize: bool,
) -> None:
    pooled = per_frame.float().mean(dim=0, keepdim=True)
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
    payload = {
        "video_id": video_id,
        "path": str(item.path),
        "embedding_per_frame": per_frame.to(torch.float16),
        "embedding_pooled": pooled.squeeze(0).to(torch.float16),
        "frame_indices": sel_indices,
        "normalized": bool(normalize),
        "dtype": "float16",
        "metadata": item.metadata,
    }
    torch.save(payload, out_path)


def generate_embeddings(
    dataset: str,
    root: str | Path,
    out_dir: str | Path,
    *,
    config: Optional[EmbeddingConfig] = None,
    split: Optional[str] = None,
    view: Optional[str] = None,
    modality: Optional[str] = None,
    fold: Optional[int] = None,
    split_csv: Optional[str | Path] = None,
    key_frame_map: Optional[Dict[str, int]] = None,
) -> Path:
    cfg = config or EmbeddingConfig()
    loader = DatasetLoader()
    items = loader.load(
        dataset,
        root=root,
        split=split,
        view=view,
        modality=modality,
        fold=fold,
        split_csv=split_csv,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_csv = out_dir / "index.csv"

    model_cfg = ModelConfig(model=cfg.model, device=cfg.device, precision=cfg.precision)
    model, _, preprocess_val, resolved_model_id = load_model(model_cfg)

    # Persist model metadata for reproducibility
    metadata_txt = out_dir / "embedding_config.txt"
    if not metadata_txt.exists() or cfg.overwrite:
        metadata_txt.write_text(
            "\n".join([
                f"model_id={resolved_model_id}",
                f"device={cfg.device}",
                f"precision={cfg.precision}",
                f"res={cfg.res}",
                f"max_frames={cfg.max_frames}",
                f"stride={cfg.stride}",
                f"batch_size={cfg.batch_size}",
                f"normalize={cfg.normalize}",
            ])
        )

    key_map = {str(k): int(v) for k, v in (key_frame_map or {}).items()}

    new_index = not index_csv.exists() or cfg.overwrite
    mode = "w" if cfg.overwrite else "a"
    with open(index_csv, mode, newline="") as handle:
        writer = csv.writer(handle)
        if new_index:
            writer.writerow(["video_id", "n_frames_raw", "n_frames_used", "key_frame", "embedding_file"])
        total = len(items)
        for idx, item in enumerate(items, 1):
            video_id = _select_video_id(dataset, item)
            out_path = out_dir / f"{video_id}.pt"
            if out_path.exists() and not cfg.overwrite:
                try:
                    payload = torch.load(out_path, map_location="cpu")
                    used = int(payload["embedding_per_frame"].shape[0]) if "embedding_per_frame" in payload else -1
                except Exception:
                    used = -1
                writer.writerow([video_id, -1, used, "SKIP", str(out_path)])
                print(f"[{idx}/{total}] skip existing: {out_path.name}")
                continue

            kf_default = item.key_frame if item.key_frame is not None else cfg.key_frame
            kf = key_map.get(video_id, kf_default)

            try:
                frames = read_video(item.path, res=(cfg.res, cfg.res))
                n_raw = int(frames.shape[0])
                sel_indices = indices_after_keyframe(n_raw, int(kf), cfg.max_frames, cfg.stride)
                if not sel_indices:
                    raise RuntimeError(f"No frames selected (n_raw={n_raw}, key_frame={kf}).")
                frames_sel = frames[sel_indices]
                frames_tensor = preprocess_frames(frames_sel, preprocess_val)
                per_frame = encode_video_clip_batched(
                    model=model,
                    frames_tensor=frames_tensor,
                    device=cfg.device,
                    precision=cfg.precision,
                    batch_size=cfg.batch_size,
                    use_channels_last=cfg.use_channels_last,
                    pin_memory=cfg.pin_memory,
                    normalize=cfg.normalize,
                )
                _save_embedding(out_path, video_id, item, per_frame, sel_indices, normalize=cfg.normalize)
                writer.writerow([video_id, n_raw, per_frame.shape[0], kf, str(out_path)])
                print(f"[{idx}/{total}] saved {out_path.name} (raw={n_raw}, used={per_frame.shape[0]}, kf={kf})")
            except Exception as exc:
                writer.writerow([video_id, -1, -1, kf, "ERROR"])
                print(f"[{idx}/{total}] ERROR {video_id}: {exc}")

    print(f"Done. Index: {index_csv}")
    return index_csv


def generate_embeddings_for_splits(
    dataset: str,
    root: str | Path,
    out_dir: str | Path,
    splits: Iterable[str],
    *,
    config: Optional[EmbeddingConfig] = None,
    view: Optional[str] = None,
    modality: Optional[str] = None,
    fold: Optional[int] = None,
    split_csv: Optional[str | Path] = None,
) -> None:
    for split in splits:
        subdir = Path(out_dir) / split
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"=== Embedding split '{split}' into {subdir} ===")
        generate_embeddings(
            dataset,
            root=root,
            out_dir=subdir,
            config=config,
            split=split,
            view=view,
            modality=modality,
            fold=fold,
            split_csv=split_csv,
        )


def main():
    import argparse

    loader = DatasetLoader()
    parser = argparse.ArgumentParser(description="Generate embeddings using the modular pipeline")
    parser.add_argument("--dataset", required=True, choices=loader.available())
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--modality", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--model", default="echo_clip", help="Model alias or id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_channels_last", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--key_frame", type=int, default=0)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    args = parser.parse_args()

    cfg = EmbeddingConfig(
        model=args.model,
        device=args.device,
        precision=args.precision,
        res=args.res,
        max_frames=args.max_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        use_channels_last=not args.no_channels_last,
        pin_memory=not args.no_pin_memory,
        overwrite=args.overwrite,
        key_frame=args.key_frame,
        normalize=bool(getattr(args, "normalize", True)),
    )

    if args.splits:
        generate_embeddings_for_splits(
            args.dataset,
            root=args.root,
            out_dir=args.out_dir,
            splits=args.splits,
            config=cfg,
            view=args.view,
            modality=args.modality,
            fold=args.fold,
            split_csv=args.split_csv,
        )
    else:
        generate_embeddings(
            args.dataset,
            root=args.root,
            out_dir=args.out_dir,
            config=cfg,
            split=args.split,
            view=args.view,
            modality=args.modality,
            fold=args.fold,
            split_csv=args.split_csv,
        )


if __name__ == "__main__":
    main()


__all__ = ["EmbeddingConfig", "generate_embeddings", "generate_embeddings_for_splits"]
