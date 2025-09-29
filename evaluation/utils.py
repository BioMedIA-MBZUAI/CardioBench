import os, math, numpy as np, pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

VIEW_CLASS_NAMES = ["A2C", "A3C", "A4C", "PSAX", "PLAX"]


def view_from_filename(fname: str) -> Optional[str]:
    base = os.path.basename(fname).lower()
    if "_2ch" in base:
        return "2CH"
    if "_4ch" in base:
        return "4CH"
    return None


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def safe_join(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    key="FileName",
    validate_method="one_to_one",
) -> pd.DataFrame:
    pred_df = pred_df.copy()
    if pred_df.duplicated(subset=[key]).any():
        print(f"Duplicates found in predictions key={key}")
        pred_df = pred_df.drop_duplicates(subset=[key], keep="first")
    merged = gt_df.merge(pred_df, on=key, how="inner", validate=validate_method)
    return merged


def multiclass_brier_score(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    n, k = probs.shape
    oh = np.zeros((n, k), dtype=float)
    oh[np.arange(n), y_true_idx] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def multiclass_ece(
    probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 15
) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else ((conf > lo) & (conf <= hi))
        if not np.any(mask):
            continue
        ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)


def summarize_pred_distribution(y_pred_labels: List[str]) -> Dict[str, int]:
    cnt = Counter(y_pred_labels)
    return {c: cnt.get(c, 0) for c in VIEW_CLASS_NAMES}


def _ci(x: np.ndarray, alpha=0.05) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return (
        round(float(np.mean(x)), 2),
        round(float(np.quantile(x, alpha / 2)), 2),
        round(float(np.quantile(x, 1 - alpha / 2)), 2),
    )


def _pearson_safe(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return pearsonr(a, b)[0]


def _spearman_safe(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return spearmanr(a, b)[0]


def bootstrap_indices(n: int, B: int, rng: np.random.Generator):
    for _ in range(B):
        yield rng.integers(0, n, size=n)


def binary_ece(prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob <= hi) if i == 0 else ((prob > lo) & (prob <= hi))
        if not np.any(mask):
            continue
        e += abs(1.0 - prob[mask].mean()) * mask.mean()
    return float(e)


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    out = {}
    denom = np.max(y) - np.min(y)

    mae = mean_absolute_error(y, yhat)
    nmae = mae / denom if denom != 0 else np.nan
    mse = mean_squared_error(y, yhat)
    rmse_val = rmse(y, yhat)
    r2 = r2_score(y, yhat) if len(np.unique(y)) > 1 else np.nan
    pearson = _pearson_safe(y, yhat)
    spearman = _spearman_safe(y, yhat)

    out = {
        "MAE": round(mae, 2),
        "NMAE(%)": round(nmae * 100, 2) if pd.notnull(nmae) else np.nan,
        "MSE": round(mse, 2),
        "RMSE": round(rmse_val, 2),
        "R2(%)": round(r2 * 100, 2) if pd.notnull(r2) else np.nan,
        "Pearson_r(%)": round(pearson * 100, 2) if pd.notnull(pearson) else np.nan,
        "Spearman_rho(%)": round(spearman * 100, 2) if pd.notnull(spearman) else np.nan,
    }
    return out


def classification_metrics(y_true, y_pred_labels, average="binary") -> Dict[str, float]:
    out = {}
    labels = np.unique(y_true)

    out["Accuracy(%)"] = 100 * accuracy_score(y_true, y_pred_labels)
    out["Balanced_Accuracy(%)"] = 100 * balanced_accuracy_score(y_true, y_pred_labels)
    out["F1(%)"] = 100 * f1_score(y_true, y_pred_labels, labels=labels, average=average)
    return {k: round(v, 4) for k, v in out.items()}


def view_metrics(
    y_true_idx: np.ndarray, probs: np.ndarray, y_pred_labels: np.ndarray
) -> Dict[str, float]:
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred_idx = probs.argmax(axis=1)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    ba = balanced_accuracy_score(y_true_idx, y_pred_idx)

    unique_classes = np.unique(y_true_idx)
    f1_macro_gt = f1_score(
        y_true_idx, y_pred_idx, average="macro", labels=unique_classes
    )

    brier = multiclass_brier_score(y_true_idx, probs)
    nll = log_loss(y_true_idx, probs, labels=list(range(probs.shape[1])))
    ece_top1 = multiclass_ece(probs, y_true_idx, n_bins=15)

    out = {
        "Accuracy(%)": round(acc * 100, 2),
        "Balanced_Accuracy(%)": round(ba * 100, 2),
        "F1_macro(%)": round(f1_macro_gt * 100, 2),
        "Brier_multiclass(%)": round(brier * 100, 2),
        "NLL_multiclass(%)": round(nll * 100, 2),
        "ECE_top1(%)": round(ece_top1 * 100, 2),
    }
    return out


def per_view_accuracy(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: List[str],
    restrict_to_gt: bool = True,
) -> Dict[str, float]:
    if restrict_to_gt:
        cls_idxs = sorted(np.unique(y_true_idx))
    else:
        cls_idxs = list(range(len(class_names)))

    mask = np.isin(y_true_idx, cls_idxs)
    yt = y_true_idx[mask]
    yp = y_pred_idx[mask]

    out = {}
    if len(yt) == 0:
        return out

    overall_acc = (yt == yp).mean()
    out["Acc_overall_GTonly(%)"] = round(overall_acc * 100, 2)
    out["n_overall_GTonly"] = int(len(yt))

    for ci in cls_idxs:
        cname = class_names[ci]
        m = yt == ci
        n_ci = int(m.sum())
        if n_ci == 0:
            continue
        acc_ci = (yp[m] == ci).mean()
        out[f"Acc_{cname}(%)"] = round(acc_ci * 100, 2)
        out[f"n_{cname}"] = n_ci

    return out


def bootstrap_regression(y, yhat, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    base = regression_metrics(y, yhat)

    bags = {k: [] for k in base.keys()}
    n = len(y)
    for idx in bootstrap_indices(n, B, rng):
        m = regression_metrics(y[idx], yhat[idx])
        for k, v in m.items():
            bags[k].append(v)
    out = {}
    for k in base.keys():
        mean, lo, hi = _ci(np.array(bags[k]))
        out[k] = base[k]
        out[k + "_mean"] = mean
        out[k + "_ci_lo"] = lo
        out[k + "_ci_hi"] = hi
    return out


def bootstrap_classification(y_true, y_pred_labels=None, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    base = classification_metrics(y_true, y_pred_labels, average="macro")
    bags = {k: [] for k in base.keys()}
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m = classification_metrics(
            y_true[idx],
            None if y_pred_labels is None else y_pred_labels[idx],
            average="macro",
        )
        for k, v in m.items():
            bags[k].append(v)

    def _ci(x, alpha=0.05):
        return (
            round(float(np.mean(x)), 4),
            round(float(np.quantile(x, alpha / 2)), 4),
            round(float(np.quantile(x, 1 - alpha / 2)), 4),
        )

    out = {}
    for k in base.keys():
        mean, lo, hi = _ci(np.array(bags[k]))
        out[k] = base[k]
        out[f"{k}_mean"] = mean
        out[f"{k}_ci_lo"] = lo
        out[f"{k}_ci_hi"] = hi
    return out


def bootstrap_view(y_true_idx, probs, y_pred_labels, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    base = view_metrics(y_true_idx, probs, y_pred_labels)
    bags = {k: [] for k in base.keys()}
    n = len(y_true_idx)
    for idx in bootstrap_indices(n, B, rng):
        m = view_metrics(y_true_idx[idx], probs[idx], y_pred_labels[idx])
        for k, v in m.items():
            bags[k].append(v)
    out = {}
    for k in base.keys():
        mean, lo, hi = _ci(np.array(bags[k]))
        out[k] = base[k]
        out[k + "_mean"] = mean
        out[k + "_ci_lo"] = lo
        out[k + "_ci_hi"] = hi
    return out


def boxplot_abs_err(
    abs_err_long: pd.DataFrame,
    grouping: str,
    save_path: str,
    title: str,
    group_order: Optional[List[str]] = None,
    model_order: Optional[List[str]] = None,
    exclude_groups: Optional[List[str]] = None,
    legend_map: Optional[Dict[str, str]] = None,
    flier_size: float = 3.0,
):
    df = abs_err_long[abs_err_long["grouping"] == grouping].dropna(
        subset=[grouping, "abs_err"]
    )
    df = df[~df["model"].astype(str).str.contains("_")]

    if exclude_groups:
        df = df[~df[grouping].astype(str).isin(exclude_groups)]

    models = model_order if model_order is not None else sorted(df["model"].unique())

    if group_order is not None:
        groups = [g for g in group_order if g in df[grouping].astype(str).unique()]
        df[grouping] = pd.Categorical(
            df[grouping].astype(str), categories=groups, ordered=True
        )
    else:
        if pd.api.types.is_categorical_dtype(df[grouping]) and getattr(
            df[grouping].dtype, "ordered", False
        ):
            groups = list(df[grouping].cat.categories)
        else:
            groups = sorted(df[grouping].astype(str).unique())
            df[grouping] = pd.Categorical(
                df[grouping].astype(str), categories=groups, ordered=True
            )

    n_models, n_groups = len(models), len(groups)

    group_width = 0.7
    box_w = group_width / max(n_groups, 1)
    centers = np.arange(n_models)

    colors = ["#33BBEE", "#EE7733", "#009988", "#EE3377", "#0077BB"]

    plt.figure(figsize=(0.7 * n_models + 2, 4))

    for gi, g in enumerate(groups):
        data = [
            df.loc[
                (df["model"] == m) & (df[grouping].astype(str) == str(g)), "abs_err"
            ].values
            for m in models
        ]
        pos = centers + (gi - (n_groups - 1) / 2) * box_w

        col = colors[gi % len(colors)]

        bp = plt.boxplot(
            data,
            positions=pos,
            widths=box_w * 0.9,
            patch_artist=True,
            showfliers=True,
            vert=True,
            flierprops=dict(
                marker="o",
                markersize=flier_size,
                markerfacecolor=col,
                markeredgecolor="black",
                linewidth=0.4,
                alpha=0.6,
            ),
        )

        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.45, edgecolor=col)
        for whisker in bp["whiskers"]:
            whisker.set(color=col, linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color=col, linewidth=1.0)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.2)

    rename_models = {
        "biomedclip": "BioMedCLIP",
        "dinov3": "DINOv3",
        "echoclip": "EchoCLIP",
        "echoprime": "EchoPrime",
        "panecho": "PanEcho",
        "sigclip2": "SigLIP2",
        "sigclip": "SigLIP2",
    }
    plt.xticks(centers, models, rotation=0, ha="center", fontsize=10)
    plt.yticks(fontsize=16)
    plt.ylabel("|EF error|", fontsize=16, weight="bold")
    plt.title(title, fontsize=16, weight="bold")

    labels = [legend_map.get(str(g), str(g)) if legend_map else str(g) for g in groups]
    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markersize=10,
            markerfacecolor=colors[i % len(colors)],
            markeredgecolor="none",
            alpha=0.7,
        )
        for i, _ in enumerate(groups)
    ]
    legend_title_map = {
        "Sex": "Sex",
        "ImageQuality": "Image Quality",
        "AgeBin": "Age Group",
        "BMI_bin": "BMI Group",
    }
    legend_title = legend_title_map.get(grouping, grouping)
    plt.legend(
        handles,
        labels,
        title=legend_title,
        ncol=len(labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        title_fontproperties={"weight": "bold"},
    )

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {save_path}")
