from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import csv

import torch

from ..text import (
    TextModelConfig,
    encode_text_prompts,
    list_embedding_files,
    load_video_embedding,
)


@dataclass
class ViewClassificationConfig(TextModelConfig):
    csv_name: str = "view_predictions.csv"
    save_csv: bool = True


@torch.inference_mode()
def predict_view(
    pooled_embedding: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_classes: List[str],
    *,
    device: str,
) -> Dict[str, float]:
    """
    Compute per-class similarity scores for a single video embedding.
    """
    vec = pooled_embedding.to(device=device, dtype=torch.float32)
    sims = vec @ prompt_embeddings.T
    classes = sorted(set(prompt_classes))
    class_to_indices = {cls: [] for cls in classes}
    for idx, cls in enumerate(prompt_classes):
        class_to_indices[cls].append(idx)
    return {cls: sims[class_to_indices[cls]].mean().item() for cls in classes}


def classify_views_dir(
    emb_dir: str | Path,
    prompts_per_class: Dict[str, List[str]],
    *,
    config: ViewClassificationConfig | None = None,
) -> Dict[str, str]:
    """
    Run zero-shot view classification over all embeddings in a directory.
    """
    cfg = config or ViewClassificationConfig()
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(prompts_per_class.keys())
    all_prompts: List[str] = []
    prompt_classes: List[str] = []
    for cls, prompts in prompts_per_class.items():
        for prompt in prompts:
            if prompt and prompt.strip():
                all_prompts.append(prompt.strip())
                prompt_classes.append(cls)
    if not all_prompts:
        raise ValueError("No prompts provided for view classification.")

    prompt_embeddings = encode_text_prompts(
        all_prompts,
        model_id=cfg.model_id,
        device=cfg.device,
        precision=cfg.precision,
    )

    files = list_embedding_files(emb_dir)
    if not files:
        raise FileNotFoundError(f"No embeddings found in {emb_dir}")

    predictions: Dict[str, str] = {}
    scores_per_id: Dict[str, Dict[str, float]] = {}
    for idx, pt_file in enumerate(files, 1):
        try:
            vid, pooled = load_video_embedding(pt_file)
            scores = predict_view(pooled, prompt_embeddings, prompt_classes, device=cfg.device)
            top_label = max(scores.items(), key=lambda kv: kv[1])[0]
            predictions[vid] = top_label
            scores_per_id[vid] = scores
            print(f"[{idx}/{len(files)}] {vid}: {top_label} | {scores}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] ERROR {pt_file.name}: {exc}")

    if cfg.save_csv:
        csv_path = emb_dir / cfg.csv_name
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_id", "pred_view"] + [f"prob_{c}" for c in class_names])
            for vid in sorted(predictions.keys()):
                scores = scores_per_id.get(vid, {})
                raw = [scores.get(c, float("nan")) for c in class_names]
                # softmax for probabilities
                torch_scores = torch.tensor(raw, dtype=torch.float32)
                probs = torch.softmax(torch_scores, dim=0)
                writer.writerow([vid, predictions[vid]] + [f"{float(p):.6f}" for p in probs])
        print(f"Wrote {csv_path}")

    return predictions


def estimate_views(
    emb_dir: str | Path,
    prompts_per_class: Dict[str, List[str]],
    *,
    config: ViewClassificationConfig | None = None,
) -> Dict[str, str]:
    """
    Backward-compat wrapper for ``classify_views_dir``.
    """
    return classify_views_dir(emb_dir, prompts_per_class, config=config)


def main():
    import argparse
    from ..prompts import view_prompt

    parser = argparse.ArgumentParser(description="Zero-shot view classification over embeddings")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--csv_name", default="view_predictions.csv")
    parser.add_argument("--model_id", default="hf-hub:mkaichristensen/echo-clip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()

    cfg = ViewClassificationConfig(model_id=args.model_id, device=args.device, precision=args.precision, csv_name=args.csv_name)
    classify_views_dir(args.emb_dir, prompts_per_class=view_prompt, config=cfg)


if __name__ == "__main__":
    main()


__all__ = ["ViewClassificationConfig", "estimate_views", "predict_view"]
