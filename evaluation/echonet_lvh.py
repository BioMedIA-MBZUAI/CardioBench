import os, math, numpy as np, pandas as pd
from utils import safe_join, bootstrap_regression, bootstrap_view, per_view_accuracy
from collections import Counter
from config import (
    B,
    SEED,
    SPLIT,
    VIEW_CLASS_NAMES,
    LVH_GT_FILES,
    IVSd_ROOT,
    LVIDd_ROOT,
    LVPWd_ROOT,
    LVH_VIEW_ROOT,
    LVH_OUT_DIR,
)


all_reg_rows = []

REG_MODELS = {
    "IVSd": {
        os.path.splitext(f)[0]: os.path.join(IVSd_ROOT, f)
        for f in os.listdir(IVSd_ROOT)
        if f.endswith(".csv")
    },
    "LVIDd": {
        os.path.splitext(f)[0]: os.path.join(LVIDd_ROOT, f)
        for f in os.listdir(LVIDd_ROOT)
        if f.endswith(".csv")
    },
    "LVPWd": {
        os.path.splitext(f)[0]: os.path.join(LVPWd_ROOT, f)
        for f in os.listdir(LVPWd_ROOT)
        if f.endswith(".csv")
    },
}
for target, gt_path in LVH_GT_FILES.items():
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(gt_path):
        print(f"[WARN] missing GT for {target}: {gt_path}")
        continue

    gt = pd.read_csv(gt_path)
    gt = gt[gt["split"] == SPLIT]
    if "HashedFileName" not in gt.columns:
        raise ValueError(f"{gt_path} must have column: HashedFileName")

    gt["CalcValue"] = pd.to_numeric(gt["CalcValue"], errors="coerce")

    for model_name, pred_path in REG_MODELS[target].items():
        if not os.path.exists(pred_path):
            print(f"[WARN] missing preds for {target}/{model_name}")
            continue

        pred = pd.read_csv(pred_path)
        pred_col = f"{target}_pred"
        if "HashedFileName" not in pred.columns or pred_col not in pred.columns:
            raise ValueError(
                f"{pred_path} must have columns: HashedFileName,{pred_col}"
            )

        pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

        # ------------------- JOIN PREDICTIONS WITH GT -------------------
        merged = safe_join(
            gt[["HashedFileName", "CalcValue"]],
            pred[["HashedFileName", pred_col]],
            "HashedFileName",
        ).dropna()
        if merged.empty:
            print(f"[WARN] no overlap {merged['CalcValue'].mean()}/{model_name}")
            continue
        print(
            f"[INFO] Found {len(merged)} overlapping rows for {target}/{model_name} on TEST split. Total TEST size={len(gt)}"
        )
        # ------------------- JOIN PREDICTIONS WITH GT -------------------
        y = merged["CalcValue"].to_numpy(float)
        yhat = merged[pred_col].to_numpy(float)

        keep = ~np.isnan(yhat)
        if (~keep).any():
            print(
                f"[WARN] {model_name}: dropped {(~keep).sum()} rows with NaN/inf {pred_col}"
            )

        # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
        stats_boot = bootstrap_regression(y, yhat, B=B, seed=SEED)
        all_reg_rows.append(
            {"target": target, "model": model_name, "n": int(y.size), **stats_boot}
        )

reg_df = pd.DataFrame(all_reg_rows).sort_values(["target", "model"])

# ---------------------------- VIEW EVALUATION ----------------------------

view_rows, confusion_rows, perview_rows = [], [], []
VIEW_MODELS = {
    os.path.splitext(f)[0]: os.path.join(LVH_VIEW_ROOT, f)
    for f in os.listdir(LVH_VIEW_ROOT)
    if f.endswith(".csv")
}
gt = pd.read_csv(gt_path)
gt = gt[gt["split"] == SPLIT]
for model_name, pred_path in VIEW_MODELS.items():
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[WARN] missing view preds for {model_name}")
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
    needed = ["HashedFileName", "view_pred"] + [f"prob_{c}" for c in VIEW_CLASS_NAMES]
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise ValueError(f"{pred_path} missing columns: {missing}")

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(gt[["HashedFileName"]], pred[needed], key="HashedFileName")

    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(gt)}"
    )

    y_true_idx = np.array([VIEW_CLASS_NAMES.index("PLAX")] * len(merged))
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
    prob_mat = prob_mat[keep]
    y_pred_idx = prob_mat[keep].argmax(axis=1)

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats_boot = bootstrap_view(y_true_idx, prob_mat, y_pred_labels, B=B, seed=SEED)
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

os.makedirs(LVH_OUT_DIR, exist_ok=True)
reg_df.to_csv(os.path.join(LVH_OUT_DIR, "LVH_regression_bootstrap.csv"), index=False)
view_metrics.to_csv(os.path.join(LVH_OUT_DIR, "LVH_view_metrics.csv"), index=False)
pred_distribution.to_csv(
    os.path.join(LVH_OUT_DIR, "LVH_pred_distribution.csv"), index=False
)
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(LVH_OUT_DIR, "view_per_class_accuracy.csv"), index=False
    )
