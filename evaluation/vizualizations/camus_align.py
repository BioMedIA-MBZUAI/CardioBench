import os
import glob
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from evaluation.config import CAMUS_SPLIT_CSV, SEED

csv_path = CAMUS_SPLIT_CSV

model_visual_dirs = {
    "BioMedClip": "evaluation/embeddings/CAMUS/vision_embeddings/BioMedClip",
    "EchoClip": "evaluation/embeddings/CAMUS/vision_embeddings/EchoClip",
    "SigLip2": "evaluation/embeddings/CAMUS/vision_embeddings/SigLIP2",
    "DinoV3": "evaluation/embeddings/CAMUS/vision_embeddings/DinoV3",
}

model_text_dirs = {
    "BioMedClip": "evaluation/embeddings/CAMUS/prompt_embeddings/ejection_fraction/BioMedClip",
    "EchoClip": "evaluation/embeddings/CAMUS/prompt_embeddings/ejection_fraction/EchoClip",
    "SigLip2": "evaluation/embeddings/CAMUS/prompt_embeddings/ejection_fraction/SigLip2",
    "DinoV3": "evaluation/embeddings/CAMUS/prompt_embeddings/ejection_fraction/DinoV3",
}

OUTPUT_DIR = "evaluation/vizualizations/output"
PNG_OUT = os.path.join(OUTPUT_DIR, "visual_text_alignment_grid_camus.png")
PDF_OUT = os.path.join(OUTPUT_DIR, "visual_text_alignment_grid_camus.pdf")

np.random.seed(SEED)
torch.manual_seed(SEED)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def add_inset_cbar(fig, ax, mappable, label, height="60%", width="3%", offset=1.04):
    cax = inset_axes(
        ax,
        width=width,
        height=height,
        loc="upper left",
        bbox_to_anchor=(offset, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=8, length=2)
    return cb


def to_2d_umap(X, seed=42, n_neighbors=30, min_dist=0.1, metric="cosine"):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def l2norm_np(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def ef_text_axis(Xt, yt):
    Xt_n = l2norm_np(Xt)
    pc1 = PCA(n_components=1, random_state=0).fit(Xt_n).components_[0]  # [Dt]
    scores = Xt_n @ pc1
    if np.corrcoef(scores, yt)[0, 1] < 0:
        pc1 = -pc1
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)
    return pc1


def align_visuals_to_axis(Xv, yv, u):
    Xv_n = l2norm_np(Xv)
    proj = Xv_n @ u  # [N]
    r = float(np.corrcoef(proj, yv)[0, 1]) if len(yv) > 1 else np.nan
    rho = float(spearmanr(proj, yv).correlation) if len(yv) > 1 else np.nan
    reg = LinearRegression().fit(yv.reshape(-1, 1), proj) if len(yv) > 1 else None
    r2 = float(reg.score(yv.reshape(-1, 1), proj)) if reg is not None else np.nan
    return {"pearson_r": r, "spearman_rho": rho, "r2": r2, "proj": proj}


def load_pt_embedding(path):
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"torch.load failed for {path}: {e}")
        raise

    def _as_vector(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            out = t.float()
        elif t.ndim == 2:
            out = t.mean(dim=0).float()
        elif t.ndim >= 3:
            x = t
            while x.ndim > 1:
                x = x.mean(dim=0)
            out = x.float()
        else:
            out = t.reshape(-1).float()
        return out

    if isinstance(obj, dict):
        for k in ["embedding_pooled", "embedding", "feat", "features", "emb", "z"]:
            v = obj.get(k, None)
            if torch.is_tensor(v):
                return _as_vector(v)
        for _, v in obj.items():
            if torch.is_tensor(v):
                return _as_vector(v)

    if torch.is_tensor(obj):
        return _as_vector(obj)

    raise ValueError(f"Unsupported .pt format: {path}")


def load_visual_for_model(visual_dir, df_test):
    if not os.path.isdir(visual_dir):
        print(f"Visual dir missing: {visual_dir}")
        return None, None, None

    pt_files = glob.glob(os.path.join(visual_dir, "*.pt"))
    name2path = {os.path.splitext(os.path.basename(p))[0]: p for p in pt_files}

    keep_rows, paths = [], []
    for _, row in df_test.iterrows():
        fname = str(row["unique_id"])
        if fname in name2path:
            keep_rows.append(row)
            paths.append(name2path[fname])

    if not keep_rows:
        print(f"No matching visuals for test IDs in {visual_dir}")
        return None, None, None

    sub = pd.DataFrame(keep_rows).reset_index(drop=True)
    embs = []
    kept_idx = []
    for i, p in enumerate(paths):
        try:
            embs.append(load_pt_embedding(p))
            kept_idx.append(i)
        except Exception:
            print(f"Skipping bad embedding: {p}")

    if not embs:
        return None, None, None

    X = torch.stack(embs, 0).numpy()
    sub = sub.iloc[kept_idx].reset_index(drop=True)
    y_ef = sub["EF"].to_numpy()
    ids = sub["unique_id"].astype(str).tolist()
    return X, y_ef, ids


def load_text_for_model(text_dir):
    print(f"Loading text embeddings from: {text_dir}")
    if not os.path.isdir(text_dir):
        print(f"Text dir missing: {text_dir}")
        return None, None

    files = sorted(glob.glob(os.path.join(text_dir, "*.pt")))
    if not files:
        print(f"No .pt files in text dir: {text_dir}")
        return None, None

    X, y = [], []
    for p in files:
        try:
            emb = load_pt_embedding(p)
        except Exception:
            print(f"Skipping bad text embedding: {p}")
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"ef_(\d+)", base)
        try:
            ef = int(m.group(1)) if m else int(re.findall(r"\d+", base)[0])
        except Exception:
            print(f"Could not parse EF from filename: {base}")
            continue
        X.append(emb)
        y.append(ef)

    if not X:
        return None, None

    return torch.stack(X, 0).numpy(), np.array(y, dtype=float)


df = pd.read_csv(csv_path)
for col in ["split", "view", "unique_id"]:
    if col not in df.columns:
        raise SystemExit(f"Required column '{col}' missing in CSV")

df["split"] = df["split"].astype(str).str.lower()
df["view"] = df["view"].astype(str)
df["unique_id"] = df["unique_id"].astype(str)
df_test = df[df["split"].eq("test")].copy()

payloads, models = {}, []
for model in ["BioMedClip", "EchoClip", "SigLip2", "DinoV3"]:
    vdir = model_visual_dirs.get(model, "")
    tdir = model_text_dirs.get(model, "")

    Xv, yv, ids = load_visual_for_model(vdir, df_test) if vdir else (None, None, None)
    Xt, yt = load_text_for_model(tdir) if tdir else (None, None)

    print(
        f"Model {model}: Visuals {None if Xv is None else Xv.shape}, "
        f"Text {None if Xt is None else Xt.shape}"
    )

    if (Xv is not None and len(Xv) > 0) or (Xt is not None and len(Xt) > 0):
        payloads[model] = {"visual": (Xv, yv, ids), "text": (Xt, yt)}
        models.append(model)

    if Xt is not None and len(Xt) == 101:
        pc1 = PCA(n_components=1, random_state=0).fit_transform(Xt)[:, 0]
        idx = np.arange(101)
        if np.corrcoef(pc1, idx)[0, 1] < 0:
            pc1 = -pc1
        r = np.corrcoef(pc1, idx)[0, 1]
        rho = spearmanr(pc1, idx).correlation
        reg = LinearRegression().fit(idx.reshape(-1, 1), pc1)
        r2 = reg.score(idx.reshape(-1, 1), pc1)

        def cos(a, b):
            return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

        adj_cos = [cos(Xt[i], Xt[i + 1]) for i in range(100)]

if not models:
    raise SystemExit("No embeddings found. Check the paths above.")

base = cm.get_cmap("winter", 256)
colors = base(np.linspace(0, 1, 256))
factor = 0.2
colors = colors * (1 - factor) + np.ones_like(colors) * factor
soft_winter = mcolors.ListedColormap(colors, name="soft_winter")

ensure_dir(OUTPUT_DIR)
n_cols = len(models)
fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14), constrained_layout=True)
fig.suptitle("CAMUS Test Embedding Spaces (EF)", fontsize=16, fontweight="bold", y=1.02)

if n_cols == 1:
    axes = np.array(axes).reshape(3, 1)

for c, model in enumerate(models):
    Xv, yv, ids = payloads[model]["visual"]
    Xt, yt = payloads[model]["text"]
    title_model = "DINOv3" if model == "DinoV3" else model

    ax = axes[0, c]
    if Xv is not None and len(Xv) > 0:
        Z = to_2d_umap(Xv, seed=SEED)
        sc = ax.scatter(
            Z[:, 0], Z[:, 1], s=14, c=yv, vmin=0, vmax=100, cmap=soft_winter
        )
        add_inset_cbar(fig, ax, sc, "EF")
        ax.set_title(f"VISUAL: {title_model}", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No visual", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)

    ax = axes[1, c]
    if Xt is not None and len(Xt) > 0:
        Zt = to_2d_umap(Xt, seed=SEED)
        sc2 = ax.scatter(
            Zt[:, 0], Zt[:, 1], s=14, c=yt, vmin=0, vmax=100, cmap=soft_winter
        )
        add_inset_cbar(fig, ax, sc2, "EF Prompt")
        ax.set_title(f"TEXT: {title_model}", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No text", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)

    ax = axes[2, c]
    if (
        (Xv is not None)
        and (Xt is not None)
        and (len(yt) == 101)
        and (Xv.shape[1] == Xt.shape[1])
    ):
        u = ef_text_axis(Xt, yt)
        stats = align_visuals_to_axis(Xv, yv, u)
        sc3 = ax.scatter(
            yv, stats["proj"], s=6, alpha=0.6, c=yv, vmin=0, vmax=100, cmap=soft_winter
        )
        add_inset_cbar(fig, ax, sc3, "EF")
        ax.set_title(
            f"ALIGN: {title_model} | r={stats['pearson_r']:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("True EF")
        ax.set_ylabel("Projection on EF text axis")
        ax.grid(alpha=0.15, linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "Alignment skipped", ha="center", va="center")
    ax.set_box_aspect(1)

plt.savefig(PNG_OUT, dpi=200, bbox_inches="tight")
plt.savefig(PDF_OUT, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to:\n - {PNG_OUT}\n - {PDF_OUT}")
