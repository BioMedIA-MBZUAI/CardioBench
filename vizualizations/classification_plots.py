import os, glob, re
import numpy as np
import pandas as pd
import torch
import umap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


BIN_CMAP = mcolors.ListedColormap(["#009988", "#CC3311"])
GT_CSV = ""  # asd split csv, e.g. ./CardiacNet/asd_split.csv
SPLIT = "test"
USE_UMAP = True
TEXT_VISION_OUT_PNG = "./output/asd_text_vision.png"

VISUALS_OUT_PNG = "./output/visuals.png"


# VISION-TEXT: Define models, text_dir must have positive/*.pt and negative/*.pt
MODELS = [
    {
        "name": "DINOv3",
        "text_dir": "/embeddings/prompt_embeddings/asd/dinov3",
        "vis_dir": "/embeddings/CardiacNet/ASD/DinoV3",
    },
    {
        "name": "SigLIP2",
        "text_dir": "/embeddings/prompt_embeddings/asd/siglip",
        "vis_dir": "/embeddings/CardiacNet/ASD/SigLip2",
    },
    {
        "name": "BioMedCLIP",
        "text_dir": "/embeddings/prompt_embeddings/asd/biomedclip",
        "vis_dir": "/embeddings/CardiacNet/ASD/BioMedClip",
    },
    {
        "name": "EchoCLIP",
        "text_dir": "/embeddings/prompt_embeddings/asd/echo_clip",
        "vis_dir": "/embeddings/CardiacNet/ASD/EchoClip",
    },
]

# VISION
DATASETS = [
    {
        "name": "CardiacNet ASD",
        "gt_csv": "",
        "split": "test",
        "id_col": "unique_id",
        "label_col": "ASD",
        "visual_sets": [
            {
                "name": "DINOv3",
                "vis_dir": "/embeddings/CardiacNet/ASD/DinoV3",
            },
            {
                "name": "SigLIP2",
                "vis_dir": "/embeddings/CardiacNet/ASD/SigLip2",
            },
            {
                "name": "BioMedCLIP",
                "vis_dir": "/embeddings/CardiacNet/ASD/BioMedClip",
            },
            {
                "name": "EchoCLIP",
                "vis_dir": "/embeddings/CardiacNet/ASD/EchoClip",
            },
            {
                "name": "PanEcho",
                "vis_dir": "/embeddings/CardiacNet/ASD/PanEcho",
            },
            {
                "name": "EchoPrime",
                "vis_dir": "/embeddings/CardiacNet/ASD/EchoPrime",
            },
        ],
    },
    {
        "name": "CardiacNet PAH",
        "gt_csv": "",
        "split": "test",
        "id_col": "unique_id",
        "label_col": "PAH",
        "visual_sets": [
            {
                "name": "DINOv3",
                "vis_dir": "/embeddings/CardiacNet/PAH/DinoV3",
            },
            {
                "name": "SigLIP2",
                "vis_dir": "/embeddings/CardiacNet/PAH/SigLip2",
            },
            {
                "name": "BioMedCLIP",
                "vis_dir": "/embeddings/CardiacNet/PAH/BioMedClip",
            },
            {
                "name": "EchoCLIP",
                "vis_dir": "/embeddings/CardiacNet/PAH/EchoClip",
            },
            {
                "name": "PanEcho",
                "vis_dir": "/embeddings/CardiacNet/PAH/PanEcho",
            },
            {
                "name": "EchoPrime",
                "vis_dir": "/embeddings/CardiacNet/PAH/EchoPrime",
            },
        ],
    },
    {
        "name": "HMC-QU",
        "gt_csv": "",
        "split": "test",
        "id_col": "unique_id",
        "label_col": "STEMI",
        "visual_sets": [
            {
                "name": "DINOv3",
                "vis_dir": "/embeddings/HMC_QU/DinoV3",
            },
            {
                "name": "SigLIP2",
                "vis_dir": "/embeddings/HMC_QU//SigLip2",
            },
            {
                "name": "BioMedCLIP",
                "vis_dir": "/embeddings/HMC_QU/BioMedClip",
            },
            {
                "name": "EchoCLIP",
                "vis_dir": "/embeddings/HMC_QU/EchoClip",
            },
            {
                "name": "PanEcho",
                "vis_dir": "/embeddings/HMC_QU/PanEcho",
            },
            {
                "name": "EchoPrime",
                "vis_dir": "/embeddings/HMC_QU/EchoPrime",
            },
        ],
    },
    {
        "name": "segRWMA",
        "gt_csv": "",
        "split": "test",
        "id_col": "patient_view_id",
        "label_col": "abnormal",
        "visual_sets": [
            {
                "name": "DINOv3",
                "vis_dir": "/embeddings/RWMA/DinoV3",
            },
            {
                "name": "SigLIP2",
                "vis_dir": "/embeddings/RWMA//SigLip2",
            },
            {
                "name": "BioMedCLIP",
                "vis_dir": "/embeddings/RWMA/BioMedClip",
            },
            {
                "name": "EchoCLIP",
                "vis_dir": "/embeddings/RWMA/EchoClip",
            },
            {
                "name": "PanEcho",
                "vis_dir": "/embeddings/RWMA/PanEcho",
            },
            {
                "name": "EchoPrime",
                "vis_dir": "/embeddings/RWMA/EchoPrime",
            },
        ],
    },
]


def l2norm_np(X):
    X = np.asarray(X, np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def to_2d_umap(X, seed=42, n_neighbors=30, min_dist=0.1, metric="cosine"):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def to_2d_tsne(X, seed=42, perp=30, n_iter=1000):
    n = X.shape[0]
    if n < 2:
        return np.zeros((n, 2))
    safe_perp = max(2, min(perp, max(2, n // 3)))
    init = (
        PCA(n_components=2, random_state=seed).fit_transform(X) if n >= 2 else "random"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perp,
        random_state=seed,
        n_iter=n_iter,
        init=init,
        learning_rate="auto",
        metric="euclidean",
        verbose=0,
    )
    return tsne.fit_transform(X)


def embed_2d(X, use_umap=True):
    return to_2d_umap(X) if use_umap else to_2d_tsne(X)


def add_binary_legend(ax, labels=("absent", "present")):
    handles = [
        mpatches.Patch(color=BIN_CMAP(0), label=labels[0]),
        mpatches.Patch(color=BIN_CMAP(1), label=labels[1]),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)


def load_pt_embedding(path):
    obj = torch.load(path, map_location="cpu")

    def _as_vec(t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if t.ndim == 1:
            return t
        while t.ndim > 1:
            t = t.mean(dim=0)
        return t

    if isinstance(obj, dict):
        for k in ["embedding_pooled", "embedding", "feat", "features", "emb", "z"]:
            if k in obj and torch.is_tensor(obj[k]):
                return _as_vec(obj[k])
        for _, v in obj.items():
            if torch.is_tensor(v):
                return _as_vec(v)
    if torch.is_tensor(obj):
        return _as_vec(obj)
    raise ValueError(f"Unsupported .pt format: {path}")


def load_asd_text_prompts(root_dir):
    pos_dir = os.path.join(root_dir, "positive")
    neg_dir = os.path.join(root_dir, "negative")
    pos_files = sorted(glob.glob(os.path.join(pos_dir, "*.pt")))
    neg_files = sorted(glob.glob(os.path.join(neg_dir, "*.pt")))
    X_pos = [load_pt_embedding(p) for p in pos_files]
    X_neg = [load_pt_embedding(p) for p in neg_files]
    if len(X_pos) == 0 or len(X_neg) == 0:
        raise SystemExit(f"No ASD text prompts found in {root_dir}")
    X_pos = torch.stack(X_pos, 0).numpy()
    X_neg = torch.stack(X_neg, 0).numpy()
    return l2norm_np(X_pos), l2norm_np(X_neg), pos_files, neg_files


def load_cardiacnet_asd_visuals(vis_dir, gt_csv, split="test"):
    df = pd.read_csv(gt_csv)
    df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    uid2y = dict(zip(df["unique_id"].astype(str), df["ASD"].astype(int)))
    pt_files = sorted(glob.glob(os.path.join(vis_dir, "*.pt")))

    X, y, ids = [], [], []
    for p in pt_files:
        uid = os.path.splitext(os.path.basename(p))[0]
        if uid not in uid2y:
            continue
        try:
            v = load_pt_embedding(p).numpy()
            X.append(v)
            y.append(uid2y[uid])
            ids.append(uid)
        except Exception:
            pass
    if len(X) == 0:
        raise SystemExit(f"No visuals matched GT split in {vis_dir}")
    X = l2norm_np(np.stack(X, 0))
    y = np.array(y, int)
    return X, y, ids


def plot_model_triplet(
    ax_vis, ax_text, ax_proj, Xv, yv, Xt_pos, Xt_neg, model_name, use_umap=True
):
    Xv = l2norm_np(Xv)
    Xt_pos = l2norm_np(Xt_pos)
    Xt_neg = l2norm_np(Xt_neg)

    Zv = embed_2d(Xv, use_umap=use_umap)
    ax = ax_vis
    sc = ax.scatter(
        Zv[:, 0],
        Zv[:, 1],
        c=yv,
        s=12,
        cmap=BIN_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.95,
        linewidths=0,
    )
    ax.set_title(f"{model_name} | Vision", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)

    Xt = np.vstack([Xt_pos, Xt_neg])
    y_text = np.array([1] * len(Xt_pos) + [0] * len(Xt_neg), int)
    Zt = embed_2d(Xt, use_umap=use_umap)

    ax = ax_text
    ax.scatter(
        Zt[y_text == 0, 0],
        Zt[y_text == 0, 1],
        s=45,
        c="#009988",
        edgecolor="k",
        lw=0.4,
        label="Absent",
    )
    ax.scatter(
        Zt[y_text == 1, 0],
        Zt[y_text == 1, 1],
        s=45,
        c="#CC3311",
        edgecolor="k",
        lw=0.4,
        label="Present",
    )

    ax.set_title(f"{model_name} | Text", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.legend(loc="best", frameon=False, fontsize=9)

    u = Xt_pos.mean(0) - Xt_neg.mean(0)
    u = u / (np.linalg.norm(u) + 1e-12)
    margin = Xv @ u
    ax = ax_proj
    ax.scatter(np.arange(len(margin)), margin, s=9, c=yv, cmap=BIN_CMAP, vmin=0, vmax=1)
    ax.axhline(0.0, ls="--", lw=1, color="#888")
    ax.set_title(f"Projection", fontsize=11, fontweight="bold")
    ax.set_xlabel("sample index")
    ax.set_ylabel("margin")
    ax.set_box_aspect(1)


def text_vision_plot():
    os.makedirs(os.path.dirname(TEXT_VISION_OUT_PNG) or ".", exist_ok=True)
    df = pd.read_csv(GT_CSV)
    _ = df[df["split"].astype(str).str.lower() == SPLIT.lower()].copy()

    n = len(MODELS)
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 11.5), constrained_layout=True)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for c, M in enumerate(MODELS):
        name, text_dir, vis_dir = M["name"], M["text_dir"], M["vis_dir"]

        Xt_pos, Xt_neg, _, _ = load_asd_text_prompts(text_dir)
        Xv, yv, ids = load_cardiacnet_asd_visuals(vis_dir, GT_CSV, SPLIT)

        plot_model_triplet(
            ax_vis=axes[0, c],
            ax_text=axes[1, c],
            ax_proj=axes[2, c],
            Xv=Xv,
            yv=yv,
            Xt_pos=Xt_pos,
            Xt_neg=Xt_neg,
            model_name=name,
            use_umap=USE_UMAP,
        )

        if c == 0:
            add_binary_legend(axes[0, c], ("Absent", "Present"))

    fig.suptitle(f"CardiacNet-ASD", fontsize=14, fontweight="bold")
    plt.savefig(TEXT_VISION_OUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved {TEXT_VISION_OUT_PNG}")


# ================ VISUALS ==========


def extract_key_from_file(path, regex=None):
    stem = os.path.splitext(os.path.basename(path))[0]
    if regex:
        m = re.match(regex, stem)
        if not m:
            return None
        if m.lastindex is None or m.lastindex < 1:
            return None
        return m.group(1)
    return stem


def load_visuals(
    vis_dir,
    gt_csv,
    split="test",
    label_col="ASD",
    id_col="unique_id",
    label_map=None,
    file_key_regex=None,
):
    df = pd.read_csv(gt_csv)
    df = df[df["split"].astype(str).str.lower() == split.lower()].copy()

    if label_map is not None:
        df[label_col] = df[label_col].map(label_map)
    df[label_col] = df[label_col].astype(int)

    uid2y = dict(zip(df[id_col].astype(str), df[label_col].astype(int)))

    pt_files = sorted(glob.glob(os.path.join(vis_dir, "*.pt")))
    X, y, ids = [], [], []
    for p in pt_files:
        key = extract_key_from_file(p, file_key_regex)
        if not key or key not in uid2y:
            continue
        try:
            v = load_pt_embedding(p).numpy()
            X.append(v)
            y.append(uid2y[key])
            ids.append(key)
        except Exception:
            pass
    if len(X) == 0:
        raise SystemExit(f"No visuals matched GT split/key in {vis_dir}")
    X = l2norm_np(np.stack(X, 0))
    y = np.array(y, int)
    return X, y, ids


def separation_metrics(Z, y):
    sil = np.nan
    try:
        if len(np.unique(y)) > 1 and len(y) >= 3:
            sil = float(silhouette_score(Z, y))
    except Exception:
        pass
    z0, z1 = Z[y == 0], Z[y == 1]
    if len(z0) and len(z1):
        mu0, mu1 = z0.mean(0), z1.mean(0)
        num = np.linalg.norm(mu1 - mu0)
        pooled = np.sqrt(0.5 * (z0.var(0).mean() + z1.var(0).mean()) + 1e-12)
        smd = float(num / pooled)
    else:
        smd = np.nan
    return sil, smd


def plot_visual_panel(ax, Xv, yv, title_prefix, use_umap=True):
    Z = embed_2d(Xv, use_umap=use_umap)
    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=yv,
        s=30,
        cmap=BIN_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.95,
        linewidths=0,
    )

    sil, smd = separation_metrics(Z, yv)
    parts = []
    if not np.isnan(sil):
        parts.append(f"sil={sil:.2f}")
    if not np.isnan(smd):
        parts.append(f"Δμ/σ={smd:.2f}")
    metric_str = " , ".join(parts) if parts else "no metric"

    ax.set_title(f"{title_prefix} | {metric_str}", fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)

    handles = [
        mpatches.Patch(color=BIN_CMAP(0), label="Absent"),
        mpatches.Patch(color=BIN_CMAP(1), label="Present"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        fontsize=10,
        ncol=2,
    )


def viasuals_plot():
    os.makedirs(os.path.dirname(VISUALS_OUT_PNG) or ".", exist_ok=True)

    n_rows = len(DATASETS)
    n_cols = max(len(d["visual_sets"]) for d in DATASETS)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 4.8 * n_rows), constrained_layout=False
    )
    plt.subplots_adjust(hspace=0.6)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r, dset in enumerate(DATASETS):
        ds_name = dset["name"]
        gt_csv = dset["gt_csv"]
        split = dset.get("split", "test")
        id_col = dset.get("id_col", "unique_id")
        label_col = dset.get("label_col", "ASD")
        label_map = dset.get("label_map")
        file_key_regex = dset.get("file_key_regex")
        models = dset["visual_sets"]

        axes[r, 0].set_ylabel(f"{ds_name}", rotation=90, fontsize=14, fontweight="bold")

        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(models):
                ax.axis("off")
                continue

            name = models[c]["name"]
            vdir = models[c]["vis_dir"]

            Xv, yv, _ = load_visuals(
                vdir,
                gt_csv,
                split=split,
                label_col=label_col,
                id_col=id_col,
                label_map=label_map,
                file_key_regex=file_key_regex,
            )
            plot_visual_panel(ax, Xv, yv, name, use_umap=USE_UMAP)

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.96])
    plt.savefig(VISUALS_OUT_PNG, dpi=220, bbox_inches="tight")
    plt.savefig(VISUALS_OUT_PNG.replace(".png", ".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    text_vision_plot()
