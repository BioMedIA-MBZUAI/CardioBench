import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
from utils import (
    per_view_accuracy,
    view_from_filename,
    bootstrap_regression,
    safe_join,
    regression_metrics,
    bootstrap_view,
    boxplot_abs_err,
)
from config import (
    B,
    SPLIT,
    SEED,
    VIEW_MAP,
    CAMUS_SPLIT_CSV,
    CAMUS_EF_PRED_DIR,
    CAMUS_VIEW_PRED_DIR,
    CAMUS_OUTPUT_DIR,
    VIEW_CLASS_NAMES,
)

EF_MODELS = sorted(glob.glob(os.path.join(CAMUS_EF_PRED_DIR, "*.csv")))
EF_MODELS = {os.path.splitext(os.path.basename(p))[0]: p for p in EF_MODELS}


GROUP_COLUMNS = ["patient_id", "Sex", "Age", "ImageQuality"]

GROUP_SPECS = [
    ("Sex", ["F", "M"]),
    ("ImageQuality", ["Poor", "Medium", "Good"]),
    ("AgeBin", ["≤45", "46–65", "66–80", "80"]),
]

# ----------------- READ GT -----------------
gt = pd.read_csv(CAMUS_SPLIT_CSV)
test = gt[gt["split"] == SPLIT].copy()

gt_patient = (
    gt.sort_values(["patient_id"])
    .drop_duplicates(subset=["patient_id"], keep="first")[["patient_id", "EF", "split"]]
    .reset_index(drop=True)
)

demo_cols = ["patient_id", "Sex", "Age", "ImageQuality"]
gt_demo = (
    gt[demo_cols].drop_duplicates("patient_id", keep="first").reset_index(drop=True)
)

gt_demo["Sex"] = gt_demo["Sex"].astype(str).str.upper().map({"F": "F", "M": "M"})
gt_demo["ImageQuality"] = gt_demo["ImageQuality"].astype(str).str.title()
gt_demo["Age"] = pd.to_numeric(gt_demo["Age"], errors="coerce")
age_bins = [-np.inf, 45, 65, 80, np.inf]
age_labels = ["≤45", "46–65", "66–80", "≥80"]
gt_demo["AgeBin"] = pd.cut(gt_demo["Age"], bins=age_bins, labels=age_labels, right=True)

gt_patient = (
    test.sort_values("patient_id")
    .drop_duplicates("patient_id", keep="first")[["patient_id", "EF"]]
    .reset_index(drop=True)
)
test_gt = pd.merge(
    gt_patient,
    gt_demo[["patient_id", "Sex", "ImageQuality", "AgeBin"]],
    on="patient_id",
    how="left",
)

gt_views = gt[["unique_id", "patient_id", "view", "split"]].reset_index(drop=True)

# ----------------- EF MODEL LIST -----------------
EF_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(CAMUS_EF_PRED_DIR, "*.csv")))
}

# ----------------- EVALUATION -----------------
ef_overall_rows, ef_group_rows, abs_err_long_for_plots = [], [], []

for model_name, pred_path in sorted(EF_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"missing EF preds for {model_name} at {pred_path}")
        continue

    pred = pd.read_csv(pred_path)
    if "patient_id" not in pred.columns:
        if "unique_id" in pred.columns:
            pred = pred.copy()
            pred["patient_id"] = pred["unique_id"].astype(str).str.split("_").str[0]
        else:
            print(f"Skip (no patient_id/unique_id): {pred_path}")
            continue

    if "EF_pred" not in pred.columns:
        print(f"Skip (no EF_pred column): {pred_path}")
        continue

    pred["EF_pred"] = pd.to_numeric(pred["EF_pred"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    # ------------------- SELECT GT SCOPE -------------------
    pred_view = view_from_filename(pred_path)
    if pred_view is None:  # the file doesn't have a _<VIEW>> tag, so use all patients
        g = test_gt.copy()
        scope = "per-patient"
    else:  # take only that view from patients
        has_view = gt_views.loc[gt_views["view"] == pred_view, "patient_id"].unique()
        g = test_gt[test_gt["patient_id"].isin(has_view)].copy()
        scope = f"per-patient (only pred on {pred_view})"
    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        g[["patient_id", "EF", "Sex", "ImageQuality", "AgeBin"]], pred, "patient_id"
    )
    merged["EF"] = pd.to_numeric(merged["EF"], errors="coerce")
    merged["EF_pred"] = pd.to_numeric(merged["EF_pred"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    merged = merged.dropna(subset=["EF", "EF_pred"]).reset_index(drop=True)

    if merged.empty:
        print(f"[WARN] no overlap for {model_name}")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(test)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    y = merged["EF"].to_numpy(float)
    yhat = merged["EF_pred"].to_numpy(float)
    stats_boot = bootstrap_regression(y, yhat, B=B, seed=SEED)
    ef_overall_rows.append({"model": model_name, "n": len(y), **stats_boot})

    # ------------------- ABS ERR LONG FOR PLOTS -------------------
    merged["abs_err"] = np.abs(merged["EF_pred"] - merged["EF"])
    for gcol, _ in GROUP_SPECS:
        df_g = merged[[gcol, "abs_err"]].copy()
        df_g["model"] = model_name
        df_g["grouping"] = gcol
        abs_err_long_for_plots.append(df_g)

    # ------------------- SUBGROUP BREAKDOWN -------------------
    for gcol, order in GROUP_SPECS:
        sub_tbl = []
        for gval in order:
            sub = merged[merged[gcol].astype(str) == str(gval)]
            if sub.empty:
                continue
            yy, yyhat = sub["EF"].to_numpy(float), sub["EF_pred"].to_numpy(float)
            m = regression_metrics(yy, yyhat)
            m.update(
                {
                    "model": model_name,
                    "grouping": gcol,
                    "group": str(gval),
                    "n": len(sub),
                }
            )
            sub_tbl.append(m)
        if not sub_tbl:
            continue

        sub_df = pd.DataFrame(sub_tbl)
        deltas = {
            "delta MAE": sub_df["MAE"].max() - sub_df["MAE"].min(),
            "delta NMAE(%)": sub_df["NMAE(%)"].max() - sub_df["NMAE(%)"].min(),
            "delta MSE": sub_df["MSE"].max() - sub_df["MSE"].min(),
            "delta RMSE": sub_df["RMSE"].max() - sub_df["RMSE"].min(),
            "delta R2": (
                (sub_df["R2(%)"].max() - sub_df["R2(%)"].min())
                if sub_df["R2(%)"].notna().any()
                else np.nan
            ),
            "delta Pearson_r(%)": (
                (sub_df["Pearson_r(%)"].max() - sub_df["Pearson_r(%)"].min())
                if sub_df["Pearson_r(%)"].notna().any()
                else np.nan
            ),
            "delta Spearman_rho(%)": (
                (sub_df["Spearman_rho(%)"].max() - sub_df["Spearman_rho(%)"].min())
                if sub_df["Spearman_rho(%)"].notna().any()
                else np.nan
            ),
        }
        sub_df["delta_row"] = False
        ef_group_rows.append(sub_df)

        ef_group_rows.append(
            pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "grouping": gcol,
                        "group": "delta(max−min)",
                        "n": int(sub_df["n"].sum()),
                        **{k: v for k, v in deltas.items()},
                        "MAE": np.nan,
                        "MSE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "Pearson_r": np.nan,
                        "Spearman_rho": np.nan,
                        "delta_row": True,
                    }
                ]
            )
        )

ef_overall = pd.DataFrame(ef_overall_rows).sort_values("model")
ef_groups = (
    pd.concat(ef_group_rows, ignore_index=True) if ef_group_rows else pd.DataFrame()
)

# ---------------------------- VIEW EVALUATION ----------------------------
view_rows, view_group_rows, confusion_rows = [], [], []
view_to_idx = {c: i for i, c in enumerate(VIEW_CLASS_NAMES)}
VIEW_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(CAMUS_VIEW_PRED_DIR, "*.csv")))
}
perview_rows = []

for model_name, pred_path in sorted(VIEW_MODELS.items()):
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[EF][WARN] missing EF preds for {model_name} at {pred_path}")
        continue

    pred = pd.read_csv(pred_path)
    base_prob_cols = [f"prob_{c}" for c in VIEW_CLASS_NAMES if c != "Other"]
    existing_base = [c for c in base_prob_cols if c in pred.columns]
    pred["prob_Other"] = 1.0 - pred[existing_base].sum(axis=1)
    pred["prob_Other"] = pred["prob_Other"].clip(lower=0.0, upper=1.0)

    need = ["unique_id", "view_pred"] + [f"prob_{c}" for c in VIEW_CLASS_NAMES]
    if any(c not in pred.columns for c in need):
        print(
            f"columns missing in {model_name} view file: {set(need)-set(pred.columns)}"
        )
        continue
    g = gt_views.copy()

    # print(g.head())
    # print(pred[need].head())
    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        g[["unique_id", "view"]], pred[need], "unique_id", validate_method="many_to_one"
    )
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(test)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    gt_view = merged["view"].astype(str).str.strip().str.upper()
    gt_view = gt_view.replace(VIEW_MAP)
    y_true_idx = gt_view.map(lambda s: view_to_idx.get(s, -1)).astype(int).to_numpy()

    y_pred_idx = merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]].values.argmax(axis=1)
    y_pred_labels = np.array(VIEW_CLASS_NAMES, dtype=object)[y_pred_idx]

    prob_mat = (
        merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    gt_view.map(view_to_idx).fillna(-1).astype(int)

    valid_gt = y_true_idx >= 0
    keep = valid_gt & (~np.isnan(prob_mat).any(axis=1))
    if (~keep).any():
        print(f"{model_name}: dropped {(~keep).sum()} rows with invalid GT/probs")

    y_true_idx = y_true_idx[keep]
    y_pred_labels = y_pred_labels[keep]
    prob_mat = prob_mat[keep]
    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats = bootstrap_view(y_true_idx, prob_mat, y_pred_labels, B=B, seed=SEED)
    view_rows.append({"model": model_name, "n": int(len(y_true_idx)), **stats})
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

os.makedirs(CAMUS_OUTPUT_DIR, exist_ok=True)
ef_overall.to_csv(os.path.join(CAMUS_OUTPUT_DIR, "EF_overall.csv"), index=False)
ef_groups.to_csv(os.path.join(CAMUS_OUTPUT_DIR, "EF_subgroups.csv"), index=False)

if not view_metrics.empty:
    view_metrics.to_csv(os.path.join(CAMUS_OUTPUT_DIR, "view_metrics.csv"), index=False)
if not pred_distribution.empty:
    pred_distribution.to_csv(
        os.path.join(CAMUS_OUTPUT_DIR, "view_pred.csv"), index=False
    )
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(CAMUS_OUTPUT_DIR, "view_per_class_accuracy.csv"), index=False
    )
if abs_err_long_for_plots:
    abs_long = pd.concat(abs_err_long_for_plots, ignore_index=True)
    boxplot_abs_err(
        abs_long,
        "Sex",
        os.path.join(CAMUS_OUTPUT_DIR, "ef_abs_error_by_Sex.png"),
        "EF |error| by Sex",
    )
    boxplot_abs_err(
        abs_long,
        "AgeBin",
        os.path.join(CAMUS_OUTPUT_DIR, "ef_abs_error_by_Age.png"),
        "EF |error| by Age",
        group_order=["≤45", "46–65", "66–80", "≥80"],
    )
    boxplot_abs_err(
        abs_long,
        "ImageQuality",
        os.path.join(CAMUS_OUTPUT_DIR, "ef_abs_error_by_ImageQuality.png"),
        "EF |error| by Image Quality",
    )
