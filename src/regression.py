from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import argparse
import csv
import torch

from .text import TextModelConfig, encode_text_prompts


def _load_per_frame_embedding(pt_path: Path) -> Tuple[str, torch.Tensor]:
    payload = torch.load(pt_path, map_location="cpu")
    vid = payload.get("video_id", pt_path.stem)
    per_frame = payload.get("embedding_per_frame")
    if per_frame is None:
        pooled = payload.get("embedding_pooled")
        if pooled is None:
            raise ValueError(f"No embeddings found in {pt_path}")
        per_frame = pooled.unsqueeze(0)
    if per_frame.ndim != 2:
        raise ValueError(f"Unexpected embedding shape in {pt_path}: {tuple(per_frame.shape)}")
    # Clone to drop the inference-mode flag carried by persisted tensors.
    return vid, per_frame.float().clone()


def _list_embedding_files(out_dir: Path) -> List[Path]:
    return sorted(p for p in out_dir.glob("*.pt") if p.is_file())


def _prepare_prompt_embeddings(prompt_embeddings: torch.Tensor, *, device: str) -> torch.Tensor:
    """Clone prompt embeddings and move to the target device/dtype."""
    return (
        prompt_embeddings.detach()
        .clone()
        .to(device=device, dtype=torch.float32)
    )


def _compute_regression_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: Sequence[float],
) -> torch.Tensor:
    per_frame_similarities = video_embeddings @ prompt_embeddings.T
    ranked_indices = torch.argsort(per_frame_similarities, dim=-1, descending=True)
    values_tensor = torch.tensor(
        prompt_values,
        device=video_embeddings.device,
        dtype=torch.float32,
    )
    ranked_values = values_tensor[ranked_indices]
    top_count = max(1, int(ranked_values.shape[-1] * 0.2))
    top_values = ranked_values[..., :top_count]
    median_top = top_values.median(dim=-1).values
    return median_top.mean(dim=-1)


def _predict_metric(
    per_frame: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: Sequence[float],
    *,
    device: str,
) -> float:
    with torch.no_grad():
        video_tensor = per_frame.to(
            device=device,
            dtype=torch.float16 if prompt_embeddings.dtype == torch.float16 else torch.float32,
        ).unsqueeze(0).clone()
        pred = _compute_regression_metric(video_tensor, prompt_embeddings, prompt_values).item()
    return float(pred)


def _expand_integer_prompts(base_prompts: Iterable[str], start: int, stop: int) -> Tuple[List[str], List[int]]:
    prompts: List[str] = []
    values: List[int] = []
    for value in range(start, stop + 1):
        replacement = str(value)
        for template in base_prompts:
            prompts.append(template.replace("<#>", replacement))
            values.append(value)
    return prompts, values


def _expand_float_prompts(
    base_prompts: Iterable[str],
    *,
    start: float,
    stop: float,
    step: float,
    fmt: str = "{:.1f}",
) -> Tuple[List[str], List[float]]:
    prompts: List[str] = []
    values: List[float] = []
    steps = int(round((stop - start) / step)) + 1
    for idx in range(steps):
        value = start + idx * step
        formatted = fmt.format(value)
        for template in base_prompts:
            prompts.append(template.replace("<#>", formatted))
            values.append(value)
    return prompts, values


@dataclass
class EjectionFractionConfig(TextModelConfig):
    csv_name: str = "ef_predictions.csv"
    min_value: int = 0
    max_value: int = 100


def estimate_ejection_fraction(
    emb_dir: str | Path,
    base_prompts: Iterable[str],
    *,
    config: Optional[EjectionFractionConfig] = None,
) -> Dict[str, float]:
    cfg = config or EjectionFractionConfig()
    emb_dir_path = Path(emb_dir)
    prompts, values = _expand_integer_prompts(base_prompts, cfg.min_value, cfg.max_value)
    prompt_embeddings = encode_text_prompts(
        prompts,
        model_id=cfg.model_id,
        device=cfg.device,
        precision=cfg.precision,
    )
    prompt_embeddings = _prepare_prompt_embeddings(prompt_embeddings, device=cfg.device)

    predictions: Dict[str, float] = {}
    files = _list_embedding_files(emb_dir_path)
    if not files:
        raise FileNotFoundError(f"No embeddings found in {emb_dir_path}")

    for idx, pt_file in enumerate(files, 1):
        try:
            vid, per_frame = _load_per_frame_embedding(pt_file)
            value = _predict_metric(per_frame, prompt_embeddings, values, device=cfg.device)
            predictions[vid] = value
            print(f"[{idx}/{len(files)}] {vid}: EF={value:.1f}%")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] ERROR {pt_file.name}: {exc}")

    csv_path = emb_dir_path / cfg.csv_name
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "ef_pred"])
        for vid, value in predictions.items():
            writer.writerow([vid, f"{value:.1f}"])
    print(f"Wrote {csv_path}")

    return predictions


@dataclass
class LvhRegressionConfig(TextModelConfig):
    csv_name: str = "lvh_predictions.csv"
    lvidd_prompts: Tuple[str, ...] = (
        "LEFT VENTRICULAR INTERNAL DIAMETER IN DIASTOLE (LVIDD) IS <#> CM. ",
        "LVIDD IS <#> CM. ",
    )
    ivsd_prompts: Tuple[str, ...] = (
        "INTERVENTRICULAR SEPTAL THICKNESS IN DIASTOLE (IVSD) IS <#> CM. ",
        "IVS THICKNESS (IVSD) IS <#> CM. ",
    )
    lvpwd_prompts: Tuple[str, ...] = (
        "LEFT VENTRICULAR POSTERIOR WALL THICKNESS IN DIASTOLE (LVPWD) IS <#> CM. ",
        "POSTERIOR WALL THICKNESS (LVPWD) IS <#> CM. ",
    )
    lvidd_range: Tuple[float, float, float] = (2.0, 8.0, 0.1)
    ivsd_range: Tuple[float, float, float] = (0.5, 2.0, 0.1)
    lvpwd_range: Tuple[float, float, float] = (0.5, 2.0, 0.1)


def estimate_lvh_metrics(
    emb_dir: str | Path,
    *,
    config: Optional[LvhRegressionConfig] = None,
) -> Dict[str, Dict[str, float]]:
    cfg = config or LvhRegressionConfig()
    emb_dir_path = Path(emb_dir)

    lvidd_prompts, lvidd_values = _expand_float_prompts(
        cfg.lvidd_prompts,
        start=cfg.lvidd_range[0],
        stop=cfg.lvidd_range[1],
        step=cfg.lvidd_range[2],
    )
    ivsd_prompts, ivsd_values = _expand_float_prompts(
        cfg.ivsd_prompts,
        start=cfg.ivsd_range[0],
        stop=cfg.ivsd_range[1],
        step=cfg.ivsd_range[2],
    )
    lvpwd_prompts, lvpwd_values = _expand_float_prompts(
        cfg.lvpwd_prompts,
        start=cfg.lvpwd_range[0],
        stop=cfg.lvpwd_range[1],
        step=cfg.lvpwd_range[2],
    )

    lvidd_pe = _prepare_prompt_embeddings(
        encode_text_prompts(lvidd_prompts, model_id=cfg.model_id, device=cfg.device, precision=cfg.precision),
        device=cfg.device,
    )
    ivsd_pe = _prepare_prompt_embeddings(
        encode_text_prompts(ivsd_prompts, model_id=cfg.model_id, device=cfg.device, precision=cfg.precision),
        device=cfg.device,
    )
    lvpwd_pe = _prepare_prompt_embeddings(
        encode_text_prompts(lvpwd_prompts, model_id=cfg.model_id, device=cfg.device, precision=cfg.precision),
        device=cfg.device,
    )

    predictions: Dict[str, Dict[str, float]] = {}
    files = _list_embedding_files(emb_dir_path)
    if not files:
        raise FileNotFoundError(f"No embeddings found in {emb_dir_path}")

    for idx, pt_file in enumerate(files, 1):
        try:
            vid, per_frame = _load_per_frame_embedding(pt_file)
            lvidd = _predict_metric(per_frame, lvidd_pe, lvidd_values, device=cfg.device)
            ivsd = _predict_metric(per_frame, ivsd_pe, ivsd_values, device=cfg.device)
            lvpwd = _predict_metric(per_frame, lvpwd_pe, lvpwd_values, device=cfg.device)
            predictions[vid] = {"LVIDd": lvidd, "IVSd": ivsd, "LVPWd": lvpwd}
            print(
                f"[{idx}/{len(files)}] {vid}: "
                f"LVIDd={lvidd:.1f} cm, IVSd={ivsd:.1f} cm, LVPWd={lvpwd:.1f} cm"
            )
        except Exception as exc:
            print(f"[{idx}/{len(files)}] ERROR {pt_file.name}: {exc}")

    csv_path = emb_dir_path / cfg.csv_name
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "LVIDd_cm", "IVSd_cm", "LVPWd_cm"])
        for vid, metrics in predictions.items():
            writer.writerow([
                vid,
                f"{metrics['LVIDd']:.1f}",
                f"{metrics['IVSd']:.1f}",
                f"{metrics['LVPWd']:.1f}",
            ])
    print(f"Wrote {csv_path}")

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot regression utilities")
    subparsers = parser.add_subparsers(dest="task", required=True)

    ef_parser = subparsers.add_parser("ef", help="Ejection fraction prediction")
    ef_parser.add_argument("--emb_dir", required=True)
    ef_parser.add_argument("--device", default="cuda")
    ef_parser.add_argument("--precision", default="bf16")
    ef_parser.add_argument("--model", default="echo_clip")
    ef_parser.add_argument("--csv_name", default="ef_predictions.csv")

    lvh_parser = subparsers.add_parser("lvh", help="Left ventricular hypertrophy metrics")
    lvh_parser.add_argument("--emb_dir", required=True)
    lvh_parser.add_argument("--device", default="cuda")
    lvh_parser.add_argument("--precision", default="bf16")
    lvh_parser.add_argument("--model", default="echo_clip")
    lvh_parser.add_argument("--csv_name", default="lvh_predictions.csv")

    args = parser.parse_args()
    if args.task == "ef":
        cfg = EjectionFractionConfig(
            model_id=args.model,
            device=args.device,
            precision=args.precision,
            csv_name=args.csv_name,
        )
        from .prompts import zero_shot_prompts

        base_prompts = zero_shot_prompts["ejection_fraction"]
        estimate_ejection_fraction(args.emb_dir, base_prompts, config=cfg)
    else:
        cfg = LvhRegressionConfig(
            model_id=args.model,
            device=args.device,
            precision=args.precision,
            csv_name=args.csv_name,
        )
        estimate_lvh_metrics(args.emb_dir, config=cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "EjectionFractionConfig",
    "LvhRegressionConfig",
    "estimate_ejection_fraction",
    "estimate_lvh_metrics",
]
