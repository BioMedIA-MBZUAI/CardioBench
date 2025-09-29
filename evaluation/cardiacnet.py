import os, glob, numpy as np, pandas as pd
from typing import List, Dict, Tuple, Optional
from utils import safe_join, bootstrap_classification, bootstrap_view, per_view_accuracy
from collections import Counter
from config import (
    ASD_GT,
    PAH_GT,
    PRED_ROOT,
    CARDIACNET_OUT_DIR,
    B,
    SEED,
    SPLIT,
    VIEW_CLASS_NAMES,
)

ASD_PRED_DIR = os.path.join(PRED_ROOT, "ASD")
PAH_PRED_DIR = os.path.join(PRED_ROOT, "PAH")
VIEW_PRED_DIR = os.path.join(PRED_ROOT, "view")

asd_gt = pd.read_csv(ASD_GT)
asd_gt["split"] = asd_gt["split"].astype(str).str.strip()
test_df = asd_gt[asd_gt["split"] == SPLIT].copy()

asd_rows = []
ASD_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(ASD_PRED_DIR, "*.csv")))
}

for model_name, pred_path in sorted(ASD_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[EF][WARN] missing ASD preds for {model_name} at {pred_path}")
        continue
    pred = pd.read_csv(pred_path)
    if "unique_id" not in pred.columns or "asd_pred" not in pred.columns:
        raise ValueError(f"{pred_path} must have columns: unique_id, asd_pred")

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        test_df[["unique_id", "ASD"]], pred[["unique_id", "asd_pred"]], key="unique_id"
    )
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(test_df)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["ASD"].astype(float).to_numpy()
    yhat = (
        pd.to_numeric(merged["asd_pred"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    keep = ~np.isnan(yhat)
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with NaN/inf ASD_pred"
        )
    y, yhat = y[keep], yhat[keep]

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_classification(y, yhat, B=B, seed=SEED)
    asd_rows.append({"model": model_name, "n": len(y), **stats_boot})

asd_metrics = pd.DataFrame(asd_rows).sort_values("model")

# ---------------------------- PAH EVALUATION ----------------------------
pah_gt = pd.read_csv(PAH_GT)
pah_gt["split"] = pah_gt["split"].astype(str).str.strip()
test_df = pah_gt[pah_gt["split"] == SPLIT].copy()

pah_rows = []
PAH_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(PAH_PRED_DIR, "*.csv")))
}

for model_name, pred_path in sorted(PAH_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[EF][WARN] missing PAH preds for {model_name} at {pred_path}")
        continue
    pred = pd.read_csv(pred_path)
    if "unique_id" not in pred.columns or "pah_pred" not in pred.columns:
        raise ValueError(f"{pred_path} must have columns: unique_id, pah_pred")

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        test_df[["unique_id", "PAH"]], pred[["unique_id", "pah_pred"]], key="unique_id"
    )
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(test_df)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["PAH"].astype(float).to_numpy()
    yhat = (
        pd.to_numeric(merged["pah_pred"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    keep = ~np.isnan(yhat)
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with NaN/inf PAH_pred"
        )
    y, yhat = y[keep], yhat[keep]

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_classification(y, yhat, B=B, seed=SEED)
    pah_rows.append({"model": model_name, "n": len(y), **stats_boot})

pah_metrics = pd.DataFrame(pah_rows).sort_values("model")

# ---------------------------- VIEW EVALUATION ----------------------------
view_rows = []
perview_rows = []
confusion_rows = []
VIEW_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(VIEW_PRED_DIR, "*.csv")))
}
test_df = pd.concat(
    [asd_gt[["unique_id", "split"]], pah_gt[["unique_id", "split"]]]
).drop_duplicates(subset=["unique_id"])
test_df = test_df[test_df["split"] == SPLIT]

for model_name, pred_path in sorted(VIEW_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[WARN] View file missing for {model_name}: {pred_path}")
        continue

    pred = pd.read_csv(pred_path)
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
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise ValueError(f"{pred_path} missing columns: {missing}")
    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    # print(pred[needed].head(1))
    # print(test_df[["unique_id"]].head(1))
    merged = safe_join(test_df[["unique_id"]], pred[needed], key="unique_id")
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(test_df)}"
    )
    y_true_idx = np.array([VIEW_CLASS_NAMES.index("A4C")] * len(merged))
    y_pred_labels = merged["view_pred"].astype(str).str.strip().to_numpy()
    prob_mat = (
        merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )

    keep = ~np.isnan(prob_mat).any(axis=1)
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with NaN/inf in probs"
        )
    y_true_idx = y_true_idx[keep]
    y_pred_labels = y_pred_labels[keep]
    y_pred_idx = prob_mat[keep].argmax(axis=1)

    perview_stats = per_view_accuracy(
        y_true_idx, y_pred_idx, VIEW_CLASS_NAMES, restrict_to_gt=True
    )
    perview_stats["model"] = model_name
    perview_rows.append(perview_stats)

    prob_mat = prob_mat[keep]

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_view(y_true_idx, prob_mat, y_pred_labels, B=B, seed=SEED)
    view_rows.append({"model": model_name, "n": len(y_true_idx), **stats_boot})

    # ------------------- PREDICTION DISTRIBUTION -------------------
    merged.loc[~merged["view_pred"].isin(VIEW_CLASS_NAMES), "view_pred"] = "Other"

    cnt = Counter(merged["view_pred"])
    dist = {c: int(cnt.get(c, 0)) for c in VIEW_CLASS_NAMES}
    dist["model"] = model_name
    confusion_rows.append(dist)

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

os.makedirs(CARDIACNET_OUT_DIR, exist_ok=True)
asd_metrics.to_csv(os.path.join(CARDIACNET_OUT_DIR, "ASD.csv"), index=False)
pah_metrics.to_csv(os.path.join(CARDIACNET_OUT_DIR, "PAH.csv"), index=False)
if not view_metrics.empty:
    view_metrics.to_csv(
        os.path.join(CARDIACNET_OUT_DIR, "view_metrics.csv"), index=False
    )
if not pred_distribution.empty:
    pred_distribution.to_csv(
        os.path.join(CARDIACNET_OUT_DIR, "view_pred.csv"), index=False
    )
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(CARDIACNET_OUT_DIR, "view_per_class_accuracy.csv"), index=False
    )
