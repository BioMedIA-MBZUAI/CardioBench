from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..text import TextModelConfig, calibrate_threshold_from_dir, classify_binary_from_dir


def resolve_prompts(
    key: Optional[str],
    overrides: List[str],
    *,
    sources: List[Dict[str, List[str]]],
) -> List[str]:
    if overrides:
        return [p.strip() for p in overrides if p and p.strip()]
    if key is None:
        raise ValueError("Provide --prompts_key or at least one --prompt override")
    for src in sources:
        if key in src:
            return [p.strip() for p in src[key] if p and p.strip()]
    raise KeyError(f"Unknown prompts key: {key}")


@dataclass
class BinaryClassificationConfig(TextModelConfig):
    csv_name: str = "binary_predictions.csv"
    pos_name: str = "positive"
    neg_name: str = "negative"
    mode: str = "threshold"  # threshold | argmax
    threshold: Optional[float] = None


def run_binary_classification(
    emb_dir: str,
    *,
    positive_prompts: List[str],
    negative_prompts: Optional[List[str]] = None,
    config: BinaryClassificationConfig | None = None,
    id_map: Optional[Dict[str, str]] = None,
    output_id_name: str = "video_id",
) -> Dict[str, str]:
    cfg = config or BinaryClassificationConfig()
    preds = classify_binary_from_dir(
        emb_dir,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        mode=cfg.mode,
        threshold=cfg.threshold,
        text_cfg=cfg,
        save_csv=True,
        csv_name=cfg.csv_name,
        class_names=(cfg.pos_name, cfg.neg_name),
        id_map=id_map,
        output_id_name=output_id_name,
    )
    return preds


def main():
    import argparse
    import csv
    from pathlib import Path
    from ..prompts import zero_shot_prompts

    parser = argparse.ArgumentParser(description="Binary zero-shot classification over embeddings")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--csv_name", default="binary_predictions.csv")
    parser.add_argument("--mode", choices=["argmax", "threshold"], default="argmax")
    parser.add_argument("--pos_key", default=None)
    parser.add_argument("--neg_key", default=None)
    parser.add_argument("--pos_prompt", action="append", default=[])
    parser.add_argument("--neg_prompt", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calib_dir", default=None)
    parser.add_argument("--calib_csv", default=None)
    parser.add_argument("--labels_id_col", default="video_id")
    parser.add_argument("--labels_label_col", default=None)
    parser.add_argument("--labels_split_col", default="split")
    parser.add_argument("--labels_split", default="train")
    parser.add_argument("--map_scores_from_col", default=None)
    parser.add_argument("--map_scores_to_col", default=None)
    parser.add_argument("--map_output_from_col", default=None)
    parser.add_argument("--map_output_to_col", default=None)
    parser.add_argument("--output_id_name", default=None)
    parser.add_argument("--model_id", default="hf-hub:mkaichristensen/echo-clip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--pos_name", default="positive")
    parser.add_argument("--neg_name", default="negative")
    args = parser.parse_args()

    cfg = BinaryClassificationConfig(
        model_id=args.model_id,
        device=args.device,
        precision=args.precision,
        csv_name=args.csv_name,
        pos_name=args.pos_name,
        neg_name=args.neg_name,
        mode=args.mode,
        threshold=args.threshold,
    )

    pos_prompts = resolve_prompts(args.pos_key, args.pos_prompt, sources=[zero_shot_prompts])
    neg_prompts: Optional[List[str]] = None
    if args.mode == "argmax":
        neg_prompts = resolve_prompts(args.neg_key, args.neg_prompt, sources=[zero_shot_prompts])

    threshold = args.threshold
    if args.mode == "threshold" and threshold is None:
        if args.calib_dir and args.calib_csv:
            thr, _ = calibrate_threshold_from_dir(
                args.calib_dir,
                pos_prompts,
                labels_csv=args.calib_csv,
                id_col=args.labels_id_col,
                label_col=args.labels_label_col,
                split_col=(args.labels_split_col or None),
                split_value=(args.labels_split or None),
                text_cfg=cfg,
                map_scores_from_col=(args.map_scores_from_col or None),
                map_scores_to_col=(args.map_scores_to_col or None),
            )
            cfg.threshold = thr
        else:
            raise ValueError("threshold mode requires --threshold or calibration data")

    id_map = None
    output_id_name = args.output_id_name or "video_id"
    if args.map_output_from_col and args.map_output_to_col and args.calib_csv:
        id_map = {}
        with open(args.calib_csv, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                src = (row.get(args.map_output_from_col) or "").strip()
                dst = (row.get(args.map_output_to_col) or "").strip()
                if not src or not dst:
                    continue
                key = src
                if args.map_output_from_col.lower() in {"path", "file_path", "file", "filename", "file_name", "query_key"}:
                    name = Path(src).name
                    key = name[:-7] if name.endswith(".nii.gz") else Path(src).stem
                id_map[key] = dst
        if args.output_id_name:
            output_id_name = args.output_id_name

    run_binary_classification(
        args.emb_dir,
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        config=cfg,
        id_map=id_map,
        output_id_name=output_id_name,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BinaryClassificationConfig",
    "run_binary_classification",
    "resolve_prompts",
]
