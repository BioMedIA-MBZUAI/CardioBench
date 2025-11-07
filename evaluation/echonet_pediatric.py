import os, math, numpy as np, pandas as pd
from collections import Counter
from utils import (
    per_view_accuracy,
    safe_join,
    bootstrap_regression,
    regression_metrics,
    bootstrap_view,
    boxplot_abs_err,
)
import glob
from config import (
    B,
    SEED,
    SPLIT,
    VIEW_CLASS_NAMES,
    PEDIATRIC_FILELIST_CSV,
    PEDIATRIC_EF_PRED_DIR,
    PEDIATRIC_OUT_DIR,
    PEDIATRIC_VIEW_MODELS,
)

EF_MODELS = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in sorted(glob.glob(os.path.join(PEDIATRIC_EF_PRED_DIR, "*.csv")))
}
EF_MODELS = {k: v for k, v in EF_MODELS.items() if "_" not in k}
VIEW_CLASS_NAMES = ["A2C", "A3C", "A4C", "PSAX", "PLAX", "Other"]
SPLIT = 0  # 0=TEST, 1=VAL, else is TRAIN
B = 1000
SEED = 42

gt = pd.read_csv(PEDIATRIC_FILELIST_CSV)
test = gt[gt["Split"].astype(int) == 0].copy()

# -------------- FOR BIAS ANALYSIS SPLIT --------------
test["Height_m"] = pd.to_numeric(test["Height"], errors="coerce") / 100.0
test["BMI"] = pd.to_numeric(test["Weight"], errors="coerce") / (test["Height_m"] ** 2)
test["Sex"] = (
    test["Sex"].astype(str).str.upper().map({"F": "F", "M": "M"}).fillna("UNK")
)
test["Age"] = pd.to_numeric(test["Age"], errors="coerce")

age_bins = [-np.inf, 1, 5, 12, 18]
age_labels = ["0–1", "1–5", "6–12", "13–18"]
test["AgeBin"] = pd.cut(test["Age"], bins=age_bins, labels=age_labels, right=True)

bmi_bins = [-np.inf, 18.5, 25, np.inf]
bmi_labels = ["Low", "Healthy", "High"]

test["BMI_bin"] = pd.cut(test["BMI"], bins=bmi_bins, labels=bmi_labels, right=False)

ef_overall_rows, ef_group_rows, abs_err_long_for_plots = [], [], []

group_specs = [
    ("Sex", ["F", "M"]),
    ("AgeBin", age_labels),
    ("BMI_bin", ["Low", "Healthy", "High"]),
]

# ---------------------------- EF EVALUATION ----------------------------
for model_name, pred_path in EF_MODELS.items():
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[WARN] missing EF preds for {model_name}")
        continue

    preds = pd.read_csv(pred_path)[["FileName", "EF_pred"]]

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        test[["FileName", "EF", "Sex", "Age", "AgeBin", "BMI", "BMI_bin"]],
        preds,
        "FileName",
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
    for gcol, _ in group_specs:
        df_g = merged[[gcol, "abs_err"]].copy()
        df_g["model"] = model_name
        df_g["grouping"] = gcol
        abs_err_long_for_plots.append(df_g)

    # ------------------- SUBGROUP BREAKDOWN -------------------
    for gcol, order in group_specs:
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
perview_rows = []

for model_name, pred_path in PEDIATRIC_VIEW_MODELS.items():
    # ------------------- SANITY CHECKS -------------------
    if not os.path.exists(pred_path):
        print(f"[WARN] missing view preds for {model_name}")
        continue

    pred = pd.read_csv(pred_path)
    base_prob_cols = [f"prob_{c}" for c in VIEW_CLASS_NAMES if c != "Other"]
    existing_base = [c for c in base_prob_cols if c in pred.columns]
    pred["prob_Other"] = 1.0 - pred[existing_base].sum(axis=1)
    pred["prob_Other"] = pred["prob_Other"].clip(lower=0.0, upper=1.0)

    need = ["FileName", "view_pred"] + [f"prob_{c}" for c in VIEW_CLASS_NAMES]
    if any(c not in pred.columns for c in need):
        print(
            f"[WARN] columns missing in {model_name} view file: {set(need)-set(pred.columns)}"
        )
        continue

    # ------------------- JOIN PREDICTIONS WITH GT -------------------
    merged = safe_join(
        test[["FileName", "view", "Sex", "Age", "AgeBin", "BMI", "BMI_bin"]],
        pred[need],
        "FileName",
    )
    if merged.empty:
        print(f"[WARN] No overlap for {model_name} on TEST split.")
        continue
    print(
        f"[INFO] Found {len(merged)} overlapping rows for {model_name} on TEST split. Total TEST size={len(test)}"
    )

    # ------------------- EXTRACT GT and PRED VALUES -------------------
    view_to_idx = {c.upper(): i for i, c in enumerate(VIEW_CLASS_NAMES)}

    gt_view = merged["view"].astype(str).str.strip().str.upper()
    y_true_idx = gt_view.map(view_to_idx).fillna(-1).astype(int).to_numpy()

    y_pred_idx = merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]].values.argmax(axis=1)
    y_pred_labels = np.array(VIEW_CLASS_NAMES, dtype=object)[y_pred_idx]

    prob_mat = (
        merged[[f"prob_{c}" for c in VIEW_CLASS_NAMES]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    valid_gt = y_true_idx >= 0
    keep = valid_gt & (~np.isnan(prob_mat).any(axis=1))
    if (~keep).any():
        print(
            f"[WARN] {model_name}: dropped {(~keep).sum()} rows with invalid GT/probs"
        )

    y_true_idx = y_true_idx[keep]
    y_pred_labels = y_pred_labels[keep]
    prob_mat = prob_mat[keep]
    perview_stats = per_view_accuracy(
        y_true_idx, y_pred_idx, VIEW_CLASS_NAMES, restrict_to_gt=True
    )
    perview_stats["model"] = model_name
    perview_rows.append(perview_stats)

    # ------------------- COMPUTE METRICS with BOOTSTRAP -------------------
    stats = bootstrap_view(y_true_idx, prob_mat, y_pred_labels, B=B, seed=SEED)
    view_rows.append({"model": model_name, "n": int(len(y_true_idx)), **stats})

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

# ---------------------------- Save CSVs ----------------------------
os.makedirs(PEDIATRIC_OUT_DIR, exist_ok=True)

ef_overall.to_csv(
    os.path.join(PEDIATRIC_OUT_DIR, "EF_overall_bootstrap.csv"), index=False
)
ef_groups.to_csv(os.path.join(PEDIATRIC_OUT_DIR, "EF_subgroups.csv"), index=False)

if not view_metrics.empty:
    view_metrics.to_csv(
        os.path.join(PEDIATRIC_OUT_DIR, "view_metrics_bootstrap.csv"), index=False
    )
if not pred_distribution.empty:
    pred_distribution.to_csv(
        os.path.join(PEDIATRIC_OUT_DIR, "view_pred_distribution.csv"), index=False
    )
if not perview_metrics.empty:
    perview_metrics.to_csv(
        os.path.join(PEDIATRIC_OUT_DIR, "view_per_class_accuracy.csv"), index=False
    )

if abs_err_long_for_plots:
    abs_long = pd.concat(abs_err_long_for_plots, ignore_index=True)
    boxplot_abs_err(
        abs_long,
        "Sex",
        os.path.join(PEDIATRIC_OUT_DIR, "ef_abs_error_by_Sex.png"),
        "EF |error| by Sex",
        exclude_groups=["UNK"],
    )
    boxplot_abs_err(
        abs_long,
        "AgeBin",
        os.path.join(PEDIATRIC_OUT_DIR, "ef_abs_error_by_Age.png"),
        "EF |error| by Age",
    )
    boxplot_abs_err(
        abs_long,
        "BMI_bin",
        os.path.join(PEDIATRIC_OUT_DIR, "ef_abs_error_by_BMI.png"),
        "EF |error| by BMI",
    )
