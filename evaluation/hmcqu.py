import os, glob, re, math, numpy as np, pandas as pd
from collections import Counter
from utils import per_view_accuracy, safe_join, bootstrap_classification, bootstrap_view
from config import (
    B,
    SEED,
    SPLIT,
    VIEW_CLASS_NAMES,
    HMCQU_CSV,
    VIEW_PRED_DIR,
    STEMI_PRED_DIR,
    HMCQU_OUT_DIR,
)


gt = pd.read_csv(HMCQU_CSV)
gt = gt[gt["split"] == SPLIT]
stemi_rows = []

STEMI_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(STEMI_PRED_DIR, "*.csv")))
}

for model_name, pred_path in sorted(STEMI_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[EF][WARN] missing STEMI preds for {model_name} at {pred_path}")
        continue
    pred = pd.read_csv(pred_path)
    if "unique_id" not in pred.columns:
        raise ValueError(f"{pred_path} must have column: unique_id")
    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        gt[["unique_id", "STEMI"]], pred[["unique_id", "stemi_pred"]], key="unique_id"
    )
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(gt)}"
    )
    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["STEMI"].astype(float).to_numpy()
    yhat = (
        pd.to_numeric(merged["stemi_pred"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    keep = ~np.isnan(yhat)
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with NaN/inf STEMI_pred"
        )
    y, yhat = y[keep], yhat[keep]
    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_classification(y, yhat, B=B, seed=SEED)
    stemi_rows.append({"model": model_name, "n": len(y), **stats_boot})

stemi_metrics = pd.DataFrame(stemi_rows).sort_values("model")
# ---------------------------- VIEW EVALUATION ----------------------------
view_rows = []
confusion_rows = []
VIEW_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(VIEW_PRED_DIR, "*.csv")))
}
perview_rows = []

for model_name, pred_path in sorted(VIEW_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[EF][WARN] missing VIEW preds for {model_name} at {pred_path}")
        continue
    pred = pd.read_csv(pred_path)
    if "unique_id" not in pred.columns or "view_pred" not in pred.columns:
        raise ValueError(f"{pred_path} must have columns: unique_id, view_pred")
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
    needed = ["unique_id", "view_pred"] + [f"prob_{c}" for c in VIEW_CLASS_NAMES]
    merged = safe_join(gt[["unique_id", "view"]], pred[needed], key="unique_id")
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(gt)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    gt_view = merged["view"].astype(str).str.strip().str.upper()
    view_to_idx = {c.upper(): i for i, c in enumerate(VIEW_CLASS_NAMES)}

    y_true_idx = gt_view.map(view_to_idx).fillna(-1).astype(int).to_numpy()
    valid = y_true_idx >= 0
    merged = merged.loc[valid].reset_index(drop=True)
    y_true_idx = y_true_idx[valid]

    prob_mat = (
        merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )

    y_pred_idx = prob_mat.argmax(axis=1)

    keep = (y_true_idx >= 0) & (~np.isnan(prob_mat).any(axis=1))
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with invalid GT or NaN/inf probs"
        )

    y_true_idx = y_true_idx[keep]
    y_pred_idx = y_pred_idx[keep]
    prob_mat = prob_mat[keep]
    perview_stats = per_view_accuracy(
        y_true_idx, y_pred_idx, VIEW_CLASS_NAMES, restrict_to_gt=True
    )
    perview_stats["model"] = model_name
    perview_rows.append(perview_stats)
    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_view(y_true_idx, prob_mat, y_pred_idx, B=B, seed=SEED)
    view_rows.append({"model": model_name, "n": len(y_true_idx), **stats_boot})

    # ------------------- PREDICTION DISTRIBUTION -------------------
    merged.loc[~merged["view_pred"].isin(VIEW_CLASS_NAMES), "view_pred"] = "Other"

    cnt = Counter(merged["view_pred"])
    dist = {c: int(cnt.get(c, 0)) for c in VIEW_CLASS_NAMES}
    dist["model"] = model_name
    confusion_rows.append(dist)

perview_metrics = (
    pd.DataFrame(perview_rows).sort_values("model") if perview_rows else pd.DataFrame()
)
view_metrics = (
    pd.DataFrame(view_rows).sort_values("model") if view_rows else pd.DataFrame()
)
pred_distribution = (
    pd.DataFrame(confusion_rows)[["model"] + VIEW_CLASS_NAMES]
    if confusion_rows
    else pd.DataFrame()
)
os.makedirs(HMCQU_OUT_DIR, exist_ok=True)
view_metrics.to_csv(os.path.join(HMCQU_OUT_DIR, "hmcqu_view_metrics.csv"), index=False)
pred_distribution.to_csv(
    os.path.join(HMCQU_OUT_DIR, "hmcqu_view_pred_distribution.csv"), index=False
)
stemi_metrics.to_csv(
    os.path.join(HMCQU_OUT_DIR, "hmcqu_stemi_metrics.csv"), index=False
)
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(HMCQU_OUT_DIR, "view_per_class_accuracy.csv"), index=False
    )
