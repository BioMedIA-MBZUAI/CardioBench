import os, math, glob, re, numpy as np, pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy.stats import pearsonr, spearmanr
from utils import (
    per_view_accuracy,
    safe_join,
    bootstrap_classification,
    bootstrap_view,
    summarize_pred_distribution,
)
from config import (
    TMED2_SPLIT_PER_IMAGE,
    TMED2_SPLIT_PER_STUDY,
    VIEW_PRED_DIR,
    AS_PRED_DIR,
    TMED2_OUT_DIR,
    VIEW_CLASS_NAMES,
    B,
    SEED,
    SPLIT,
)


os.makedirs(TMED2_OUT_DIR, exist_ok=True)

per_image_gt = pd.read_csv(TMED2_SPLIT_PER_IMAGE)
per_image_gt = per_image_gt[per_image_gt["split"] == SPLIT]
per_study_gt = pd.read_csv(TMED2_SPLIT_PER_STUDY)
per_study_gt = per_study_gt[per_study_gt["split"] == SPLIT]

AS_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(AS_PRED_DIR, "*.csv")))
}
as_rows = []
for model_name, pred_path in sorted(AS_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[AS][WARN] missing AS preds for {model_name} at {pred_path}")
        continue
    pred = pd.read_csv(pred_path)
    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    if "_" in model_name:
        if "query_key" not in pred.columns or "as_pred" not in pred.columns:
            raise ValueError(f"{pred_path} must have columns: query_key, as_pred")
        merged = safe_join(
            per_image_gt[["query_key", "diagnosis_label"]],
            pred[["query_key", "as_pred"]],
            key="query_key",
        )
        if merged.empty:
            per_image_gt.head(1)
            pred.head(1)
            print(f"[WARN] No overlap for {model_name} on TEST split.")
            continue
        print(
            f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(per_image_gt)}"
        )
    else:
        if "study_id" not in pred.columns or "as_pred" not in pred.columns:
            raise ValueError(f"{pred_path} must have columns: study_id, as_pred")
        merged = safe_join(
            per_study_gt[["study_id", "diagnosis_label"]],
            pred[["study_id", "as_pred"]],
            key="study_id",
        )
        if merged.empty:
            print(f"[WARN] No overlap for {model_name} on TEST split.")
            continue
        print(
            f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(per_study_gt)}"
        )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["diagnosis_label"].astype(float).to_numpy()
    yhat = (
        pd.to_numeric(merged["as_pred"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    keep = ~np.isnan(yhat)
    if (~keep).any():
        print(f"[WARN] {model_name}: dropped {(~keep).sum()} rows with NaN/inf AS_pred")
    y, yhat = y[keep], yhat[keep]
    y_true_set = set(np.unique(y))
    y_pred_set = set(np.unique(yhat))

    extra_preds = y_pred_set - y_true_set
    missing_in_preds = y_true_set - y_pred_set
    if extra_preds:
        print(
            f"[WARN] {model_name}: predictions contain classes not in ground truth: {extra_preds}"
        )
        print("Classes in y_true:", y_true_set)
        print("Classes in y_pred:", y_pred_set)
        print("Classes predicted but not in ground truth:", extra_preds)
        print("Classes in ground truth but never predicted:", missing_in_preds)
    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_classification(y, yhat, B=B, seed=SEED)
    as_rows.append({"model": model_name, "n": len(y), **stats_boot})

as_metrics = pd.DataFrame(as_rows).sort_values("model")
# ---------------------------- VIEW EVALUATION ----------------------------
view_rows = []
confusion_rows = []
VIEW_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(VIEW_PRED_DIR, "*.csv")))
}
perview_rows = []
for model_name, pred_path in sorted(VIEW_MODELS.items()):
    print(f"[INFO] Evaluating VIEW model: {model_name}")
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[AS][WARN] missing VIEW preds for {model_name} at {pred_path}")
        continue
    pred = pd.read_csv(pred_path)
    if "query_key" not in pred.columns or "view_pred" not in pred.columns:
        raise ValueError(f"{pred_path} must have columns: query_key, view_pred")
    pred_cols = [
        c for c in [f"prob_{c}" for c in VIEW_CLASS_NAMES[:-1]] if c in pred.columns
    ]
    if len(pred_cols) < len(VIEW_CLASS_NAMES) - 1:
        missing_prob_cols = sorted(
            set(f"prob_{c}" for c in VIEW_CLASS_NAMES[:-1]) - set(pred_cols)
        )
        print(f"[WARN] Missing prob columns in {model_name}: {missing_prob_cols}")

    pred["prob_Other"] = 1.0 - pred[pred_cols].sum(axis=1)
    pred["prob_Other"] = pred["prob_Other"].clip(lower=0.0, upper=1.0)
    needed = ["query_key", "view_pred"] + [f"prob_{c}" for c in VIEW_CLASS_NAMES]

    merged = safe_join(
        per_image_gt[["query_key", "view"]], pred[needed], key="query_key"
    )
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(per_image_gt)}"
    )
    merged["view"] = merged["view"].astype(str).str.strip().str.upper()
    merged.loc[merged["view"].str.contains("OR", na=False), "view"] = "Other"

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["view"].astype(str).str.strip().str.upper()
    view_to_idx = {c.upper(): i for i, c in enumerate(VIEW_CLASS_NAMES)}
    y_true_idx = y.map(view_to_idx).fillna(-1).astype(int).to_numpy()
    y_pred_idx = merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]].values.argmax(axis=1)
    prob_mat = (
        merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    keep = (y_true_idx >= 0) & (~np.isnan(prob_mat).any(axis=1))
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with invalid GT or NaN/inf probs"
        )

    y_true_idx = y_true_idx[keep]
    y_pred_idx = y_pred_idx[keep]
    prob_mat = prob_mat[keep]
    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_view(y_true_idx, prob_mat, y_pred_idx, B=B, seed=SEED)
    view_rows.append({"model": model_name, "n": len(y_true_idx), **stats_boot})
    perview_stats = per_view_accuracy(
        y_true_idx, y_pred_idx, VIEW_CLASS_NAMES, restrict_to_gt=True
    )
    perview_stats["model"] = model_name
    perview_rows.append(perview_stats)

    # ------------------- PREDICTION DISTRIBUTION -------------------
    merged.loc[~merged["view_pred"].isin(VIEW_CLASS_NAMES), "view_pred"] = "Other"

    cnt = Counter(merged["view_pred"])
    dist = {c: int(cnt.get(c, 0)) for c in VIEW_CLASS_NAMES}
    dist["model"] = model_name
    confusion_rows.append(dist)
    print(f"[INFO] Prediction distribution for {model_name}: {dist}")
os.makedirs(TMED2_OUT_DIR, exist_ok=True)

view_metrics = (
    pd.DataFrame(view_rows).sort_values("model") if view_rows else pd.DataFrame()
)
pred_distribution = (
    pd.DataFrame(confusion_rows)[["model"] + VIEW_CLASS_NAMES]
    if confusion_rows
    else pd.DataFrame()
)
perview_metrics = (
    pd.DataFrame(perview_rows).sort_values("model") if perview_rows else pd.DataFrame()
)

os.makedirs(TMED2_OUT_DIR, exist_ok=True)
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(TMED2_OUT_DIR, "view_per_class_accuracy.csv"), index=False
    )
view_metrics.to_csv(os.path.join(TMED2_OUT_DIR, "as_view_metrics.csv"), index=False)
pred_distribution.to_csv(
    os.path.join(TMED2_OUT_DIR, "as_view_pred_distribution.csv"), index=False
)
as_metrics.to_csv(os.path.join(TMED2_OUT_DIR, "as_metrics.csv"), index=False)
