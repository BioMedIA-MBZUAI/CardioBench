import os, math, numpy as np, pandas as pd
from utils import (
    per_view_accuracy,
    safe_join,
    bootstrap_regression,
    bootstrap_view,
)
from collections import Counter
from config import (
    B,
    SEED,
    SPLIT,
    VIEW_CLASS_NAMES,
    GT_ECHONET_DYNAMIC,
    EF_MODELS,
    VIEW_MODELS,
    ECHONETDYNAMIC_OUT_DIR,
)

filelist = pd.read_csv(GT_ECHONET_DYNAMIC)
filelist["Split"] = filelist["Split"].astype(str).str.strip()
test_df = filelist[filelist["Split"] == SPLIT].copy().reset_index(drop=True)

ef_rows = []

# ---------------------------- EF EVALUATION ----------------------------
for name, path in EF_MODELS.items():
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(path):
        print(f"[WARN] EF file missing for {name}: {path}")
        continue
    pred = pd.read_csv(path)

    if "FileName" not in pred.columns or "EF_pred" not in pred.columns:
        raise ValueError(f"{path} must have columns: FileName, EF_pred")

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        test_df[["FileName", "EF"]], pred[["FileName", "EF_pred"]], key="FileName"
    )
    if merged.empty:
        print(f"[WARN] No overlap for {name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {name} on TEST split. Total TEST size={len(test_df)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["EF"].astype(float).to_numpy()
    yhat = (
        pd.to_numeric(merged["EF_pred"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    keep = ~np.isnan(yhat)
    if (~keep).any():
        print(f"[WARN] {name}: dropped {(~keep).sum()} rows with NaN/inf EF_pred")
    y, yhat = y[keep], yhat[keep]

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_regression(y, yhat, B=B, seed=SEED)
    ef_rows.append({"model": name, "n": len(y), **stats_boot})

ef_metrics = pd.DataFrame(ef_rows).sort_values("model")

# ---------------------------- VIEW EVALUATION ----------------------------
view_rows = []
confusion_rows = []
perview_rows = []

for name, path in VIEW_MODELS.items():
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(path):
        print(f"[WARN] View file missing for {name}: {path}")
        continue
    pred = pd.read_csv(path)
    pred_cols = [
        c for c in [f"prob_{c}" for c in VIEW_CLASS_NAMES[:-1]] if c in pred.columns
    ]
    if len(pred_cols) < len(VIEW_CLASS_NAMES) - 1:
        missing_prob_cols = sorted(
            set(f"prob_{c}" for c in VIEW_CLASS_NAMES[:-1]) - set(pred_cols)
        )
        print(f"[WARN] Missing prob columns in {name}: {missing_prob_cols}")

    pred["prob_Other"] = 1.0 - pred[pred_cols].sum(axis=1)
    pred["prob_Other"] = pred["prob_Other"].clip(lower=0.0, upper=1.0)
    needed = ["FileName", "view_pred"] + [f"prob_{c}" for c in VIEW_CLASS_NAMES]
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(test_df[["FileName"]], pred[needed], key="FileName")
    if merged.empty:
        print(f"[WARN] No overlap for {name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {name} on TEST split. Total TEST size={len(test_df)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
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
        print(f"[WARN] {name}: dropped {(~keep).sum()} rows with NaN/inf in probs")
    y_true_idx = y_true_idx[keep]
    y_pred_labels = y_pred_labels[keep]
    prob_mat = prob_mat[keep]
    y_pred_idx = prob_mat[keep].argmax(axis=1)

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_view(y_true_idx, prob_mat, y_pred_labels, B=B, seed=SEED)
    view_rows.append({"model": name, "n": len(y_true_idx), **stats_boot})

    perview_stats = per_view_accuracy(
        y_true_idx, y_pred_idx, VIEW_CLASS_NAMES, restrict_to_gt=True
    )
    perview_stats["model"] = name
    perview_rows.append(perview_stats)

    # ------------------- PREDICTION DISTRIBUTION -------------------
    merged.loc[~merged["view_pred"].isin(VIEW_CLASS_NAMES), "view_pred"] = "Other"

    cnt = Counter(merged["view_pred"])
    dist = {c: int(cnt.get(c, 0)) for c in VIEW_CLASS_NAMES}
    dist["model"] = name
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

# ---------------------------- Save CSVs ----------------------------
os.makedirs(ECHONETDYNAMIC_OUT_DIR, exist_ok=True)

ef_metrics.to_csv(
    os.path.join(ECHONETDYNAMIC_OUT_DIR, "EF_metrics_bootstrap.csv"), index=False
)
if not view_metrics.empty:
    view_metrics.to_csv(
        os.path.join(ECHONETDYNAMIC_OUT_DIR, "view_metrics_bootstrap.csv"), index=False
    )
if not pred_distribution.empty:
    pred_distribution.to_csv(
        os.path.join(ECHONETDYNAMIC_OUT_DIR, "view_pred_distribution.csv"), index=False
    )
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(ECHONETDYNAMIC_OUT_DIR, "view_per_class_accuracy.csv"), index=False
    )
