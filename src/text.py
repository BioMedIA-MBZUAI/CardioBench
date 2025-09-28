from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv

import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms

from .models import resolve_model_id

try:
    from open_clip import get_tokenizer as _oc_get_tokenizer
except Exception:
    _oc_get_tokenizer = None


@dataclass
class TextModelConfig:
    model_id: str = "echo_clip"
    device: str = "cuda"
    precision: str = "bf16"


@torch.inference_mode()
def encode_text_prompts(prompts: List[str], *, model_id: str, device: str, precision: str) -> torch.Tensor:
    resolved_id = resolve_model_id(model_id)
    model, _, _ = create_model_and_transforms(resolved_id, precision=precision, device=device)
    name = resolved_id.split(":", 1)[1] if resolved_id.startswith("hf-hub:") else resolved_id

    toks = None
    if _oc_get_tokenizer is not None:
        try:
            tokenizer = _oc_get_tokenizer(name)
            toks = tokenizer(prompts)
        except Exception:
            toks = None
    if toks is None and any(s in name.lower() for s in ["biomedclip", "pubmedbert", "bert"]):
        try:
            from transformers import AutoTokenizer  # type: ignore
            hf_tok = AutoTokenizer.from_pretrained(name)
            ml = getattr(hf_tok, "model_max_length", None)
            max_len = int(ml) if isinstance(ml, int) and 0 < ml < 100_000 else 256
            toks = hf_tok(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
        except Exception:
            toks = None
    if toks is None:
        from open_clip import tokenize as _clip_tokenize
        toks = _clip_tokenize(prompts)

    if isinstance(toks, dict) or hasattr(toks, "keys"):
        try:
            input_ids = toks["input_ids"]  # type: ignore[index]
        except Exception:
            input_ids = None
        payload = toks if input_ids is None else torch.as_tensor(input_ids, device=device, dtype=torch.long)
    else:
        payload = toks
    if isinstance(payload, torch.Tensor):
        payload = payload.to(device=device, dtype=torch.long)

    emb = model.encode_text(payload)
    return F.normalize(emb, dim=-1).to(torch.float32)


def list_embedding_files(out_dir: str | Path) -> List[Path]:
    return sorted(p.resolve() for p in Path(out_dir).glob("*.pt") if p.is_file())


def load_video_embedding(pt_path: str | Path) -> Tuple[str, torch.Tensor]:
    payload = torch.load(pt_path, map_location="cpu")
    vid = payload.get("video_id", Path(pt_path).stem)
    if "embedding_pooled" in payload:
        vec = payload["embedding_pooled"].float()
        vec = F.normalize(vec, dim=-1)
    else:
        per_frame = payload["embedding_per_frame"].float()
        vec = F.normalize(per_frame.mean(dim=0), dim=-1)
    return vid, vec


@torch.inference_mode()
def score_against_positive_prompts(
    pooled_embedding: torch.Tensor,
    pos_prompt_emb: torch.Tensor,
    *,
    device: str = "cuda",
) -> float:
    vec = pooled_embedding.to(device=device, dtype=torch.float32)
    sims = vec @ pos_prompt_emb.T
    return sims.mean().item()


@torch.inference_mode()
def mean_scores_for_dir(
    emb_dir: str | Path,
    pos_prompt_emb: torch.Tensor,
    *,
    device: str = "cuda",
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for pt_file in list_embedding_files(emb_dir):
        try:
            vid, pooled = load_video_embedding(pt_file)
            scores[vid] = score_against_positive_prompts(pooled, pos_prompt_emb, device=device)
        except Exception as exc:
            print(f"ERROR scoring {pt_file.name}: {exc}")
    return scores


def _f1_for_threshold(scores: torch.Tensor, labels: torch.Tensor, thr: float) -> float:
    pred = (scores > thr).to(labels.dtype)
    tp = (pred * labels).sum().item()
    fp = (pred * (1 - labels)).sum().item()
    fn = ((1 - pred) * labels).sum().item()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def calibrate_f1_threshold(scores: Iterable[float], labels: Iterable[int]) -> Tuple[float, float]:
    s = torch.tensor(list(scores), dtype=torch.float32)
    y = torch.tensor(list(labels), dtype=torch.float32)
    if s.numel() == 0:
        raise ValueError("No scores provided for calibration.")
    if y.numel() != s.numel():
        raise ValueError("scores and labels length mismatch.")

    uniq = torch.unique(s).tolist()
    if len(uniq) == 1:
        uniq = [uniq[0] - 1e-6, uniq[0] + 1e-6]
    else:
        uniq = sorted(uniq)
        uniq = [uniq[0] - 1e-6] + uniq + [uniq[-1] + 1e-6]

    best_f1, best_thr = -1.0, float("nan")
    for thr in uniq:
        f1 = _f1_for_threshold(s, y, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def read_labels_csv(
    labels_csv: str | Path,
    *,
    id_col: str = "video_id",
    label_col: Optional[str] = None,
    label_cols: Tuple[str, ...] = ("label", "target", "y"),
    split_col: Optional[str] = None,
    split_value: Optional[str] = None,
) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(labels_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        target_col: Optional[str] = label_col
        if target_col is None:
            for col in reader.fieldnames or []:
                if col in label_cols:
                    target_col = col
                    break
        if target_col is None:
            raise ValueError(f"No label column found; expected one of {label_cols} or set label_col.")
        for row in reader:
            if split_col and split_value is not None:
                val = (row.get(split_col) or "").strip()
                if val.lower() != str(split_value).lower():
                    continue
            raw_id = (row.get(id_col) or "").strip()
            if not raw_id:
                continue
            if id_col.lower() in {"path", "file_path", "file", "filename", "file_name", "query_key"}:
                try:
                    name = Path(raw_id).name
                    raw_id = name[:-7] if name.endswith(".nii.gz") else Path(raw_id).stem
                except Exception:
                    pass
            try:
                labels[raw_id] = int(float(row.get(target_col, 0)))
            except Exception:
                continue
    return labels


@torch.inference_mode()
def calibrate_threshold_from_dir(
    calib_emb_dir: str | Path,
    positive_prompts: List[str],
    *,
    labels_csv: str | Path,
    id_col: str = "video_id",
    label_col: Optional[str] = None,
    split_col: Optional[str] = None,
    split_value: Optional[str] = None,
    text_cfg: TextModelConfig | None = None,
    map_scores_from_col: Optional[str] = None,
    map_scores_to_col: Optional[str] = None,
) -> Tuple[float, float]:
    cfg = text_cfg or TextModelConfig()
    pos_emb = encode_text_prompts(positive_prompts, model_id=cfg.model_id, device=cfg.device, precision=cfg.precision)
    scores = mean_scores_for_dir(calib_emb_dir, pos_emb, device=cfg.device)

    if map_scores_from_col and map_scores_to_col:
        mapping: Dict[str, str] = {}
        with open(labels_csv, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                src = (row.get(map_scores_from_col) or "").strip()
                dst = (row.get(map_scores_to_col) or "").strip()
                if not src or not dst:
                    continue
                key = src
                if map_scores_from_col.lower() in {"path", "file_path", "file", "filename", "file_name", "query_key"}:
                    try:
                        name = Path(src).name
                        key = name[:-7] if name.endswith(".nii.gz") else Path(src).stem
                    except Exception:
                        key = src
                mapping[key] = dst
        if mapping:
            scores = {mapping.get(k, k): v for k, v in scores.items()}

    labels = read_labels_csv(
        labels_csv,
        id_col=id_col,
        label_col=label_col,
        split_col=split_col,
        split_value=split_value,
    )

    xs: List[float] = []
    ys: List[int] = []
    for vid, score in scores.items():
        if vid in labels:
            xs.append(score)
            ys.append(int(labels[vid]))
    if not xs:
        raise ValueError("No overlapping IDs between embeddings and labels CSV for calibration.")

    thr, f1 = calibrate_f1_threshold(xs, ys)
    print(f"Calibrated F1 threshold: {thr:.6f} (F1={f1:.4f}) using N={len(xs)}")
    return thr, f1


@torch.inference_mode()
def classify_binary_from_dir(
    emb_dir: str | Path,
    *,
    positive_prompts: List[str],
    negative_prompts: Optional[List[str]] = None,
    mode: str = "argmax",
    threshold: Optional[float] = None,
    text_cfg: TextModelConfig | None = None,
    save_csv: bool = True,
    csv_name: str = "binary_predictions.csv",
    class_names: Tuple[str, str] = ("positive", "negative"),
    id_map: Optional[Dict[str, str]] = None,
    output_id_name: str = "video_id",
) -> Dict[str, str]:
    cfg = text_cfg or TextModelConfig()
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    pos_emb = encode_text_prompts(positive_prompts, model_id=cfg.model_id, device=cfg.device, precision=cfg.precision)
    if mode == "argmax":
        if not negative_prompts:
            raise ValueError("argmax mode requires negative_prompts")
        neg_emb = encode_text_prompts(negative_prompts, model_id=cfg.model_id, device=cfg.device, precision=cfg.precision)
    elif mode == "threshold":
        if threshold is None:
            raise ValueError("threshold mode requires a threshold")
        neg_emb = None
    else:
        raise ValueError("mode must be 'argmax' or 'threshold'")

    preds: Dict[str, str] = {}
    rows: List[List[object]] = []
    pos_name, neg_name = class_names

    files = list_embedding_files(emb_dir)
    if not files:
        raise FileNotFoundError(f"No embeddings (.pt) found in: {emb_dir.resolve()}")

    for idx, fpath in enumerate(files, 1):
        try:
            vid, pooled = load_video_embedding(fpath)
            out_id = id_map.get(vid, vid) if id_map else vid
            if mode == "argmax":
                vec = pooled.to(device=cfg.device, dtype=torch.float32)
                pos_score = (vec @ pos_emb.T).mean().item()
                neg_score = (vec @ neg_emb.T).mean().item()  # type: ignore[arg-type]
                mval = max(pos_score, neg_score)
                e_pos = torch.exp(torch.tensor(pos_score - mval))
                e_neg = torch.exp(torch.tensor(neg_score - mval))
                denom = (e_pos + e_neg).item()
                prob_pos = float(e_pos.item() / denom)
                prob_neg = float(e_neg.item() / denom)
                label = pos_name if prob_pos >= prob_neg else neg_name
                rows.append([out_id, label, prob_pos, prob_neg])
            else:
                score = score_against_positive_prompts(pooled, pos_emb, device=cfg.device)
                label = pos_name if score > float(threshold) else neg_name
                rows.append([out_id, label, float(score)])
            preds[out_id] = label
            if idx % 50 == 0 or idx == len(files):
                print(f"Scored {idx}/{len(files)}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] ERROR {fpath.name}: {exc}")

    if save_csv:
        csv_path = emb_dir / csv_name
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            if mode == "argmax":
                writer.writerow([output_id_name, "pred_class", f"prob_{pos_name}", f"prob_{neg_name}"])
            else:
                writer.writerow([output_id_name, "pred_class", "score_pos_mean"])
            writer.writerows(rows)
        print(f"Wrote {csv_path}")

    return preds


__all__ = [
    "TextModelConfig",
    "encode_text_prompts",
    "list_embedding_files",
    "load_video_embedding",
    "score_against_positive_prompts",
    "mean_scores_for_dir",
    "calibrate_f1_threshold",
    "read_labels_csv",
    "calibrate_threshold_from_dir",
    "classify_binary_from_dir",
]
