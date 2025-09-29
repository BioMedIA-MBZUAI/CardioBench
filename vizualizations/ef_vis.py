import os, re, glob
import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from scipy.stats import spearmanr


base = cm.get_cmap("winter", 256)
colors = base(np.linspace(0, 1, 256))
colors = colors * 0.8 + np.ones_like(colors) * 0.2
soft_winter = mcolors.ListedColormap(colors, name="soft_winter")
EF_NORM = mcolors.Normalize(vmin=0, vmax=100)
SM_SHARED = mpl.cm.ScalarMappable(norm=EF_NORM, cmap=soft_winter)

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["BioMedCLIP", "EchoCLIP", "SigLIP2", "DINOv3"]

DATASETS = {
    "EchoNet-Dynamic": {
        "csv": "./EchoNet-Dynamic/FileList.csv",
        "split_fn": lambda df: df[
            df["Split"].astype(str).str.upper().eq("TEST")
        ].copy(),
        "id_getter": lambda row: str(row["FileName"]),
        "visual": {
            "BioMedCLIP": "/embeddings/EchoNetDynamic/BioMedClip",
            "EchoCLIP": "/embeddings/EchoNetDynamic/EchoClip",
            "SigLIP2": "/embeddings/EchoNetDynamic/SigLip2",
            "DINOv3": "/embeddings/EchoNetDynamic/DinoV3",
        },
        "text": {
            "BioMedCLIP": "/embeddings/prompt_embeddings/ejection_fraction/biomedclip",
            "EchoCLIP": "/embeddings/prompt_embeddings/ejection_fraction/echo_clip",
            "SigLIP2": "/embeddings/prompt_embeddings/ejection_fraction/sigclip",
            "DINOv3": "/embeddings/prompt_embeddings/ejection_fraction/dinov3",
        },
    },
    "CAMUS": {
        "csv": "./CAMUS_public/camus_split.csv",
        "split_fn": lambda df: (
            df.assign(split=df["split"].astype(str).str.lower())[
                lambda x: x["split"].eq("test")
            ].copy()
        ),
        "id_getter": lambda row: str(row["unique_id"]),
        "visual": {
            "BioMedCLIP": "/embeddings/CAMUS/BioMedClip",
            "EchoCLIP": "/embeddings/CAMUS/EchoClip",
            "SigLIP2": "/embeddings/CAMUS/SigLip2",
            "DINOv3": "/embeddings/CAMUS/DinoV3",
        },
        "text": {
            "BioMedCLIP": "/embeddings/prompt_embeddings/ejection_fraction/biomedclip",
            "EchoCLIP": "/embeddings/prompt_embeddings/ejection_fraction/echo_clip",
            "SigLIP2": "/embeddings/prompt_embeddings/ejection_fraction/sigclip",
            "DINOv3": "/embeddings/prompt_embeddings/ejection_fraction/dinov3",
        },
    },
    "EchoNet-Pediatric": {
        "csv": "./EchoNet_Pediatric/EchoNet_Pediatric/FileList.csv",
        "split_fn": lambda df: (
            df.assign(Split=df["Split"].astype(int))[lambda x: x["Split"].eq(0)].copy()
        ),
        "id_getter": lambda row: str(row["FileName"]).replace(".avi", ""),
        "visual": {
            "BioMedCLIP": "/embeddings/EchoNetPediatric/BioMedClip",
            "EchoCLIP": "/embeddings/EchoNetPediatric/EchoClip",
            "SigLIP2": "/embeddings/EchoNetPediatric/SigLip2",
            "DINOv3": "/embeddings/EchoNetPediatric/DinoV3",
        },
        "text": {
            "BioMedCLIP": "/embeddings/prompt_embeddings/ejection_fraction/biomedclip",
            "EchoCLIP": "/embeddings/prompt_embeddings/ejection_fraction/echo_clip",
            "SigLIP2": "/embeddings/prompt_embeddings/ejection_fraction/sigclip",
            "DINOv3": "/embeddings/prompt_embeddings/ejection_fraction/dinov3",
        },
    },
}


def to_2d_umap(X, seed=42, n_neighbors=30, min_dist=0.12, metric="cosine"):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    ).fit_transform(X)


embed2d = to_2d_umap


def add_inset_cbar_shared(fig, ax, label, ticks=(0, 20, 40, 60, 80, 100)):
    cax = inset_axes(
        ax,
        width="2%",
        height="40%",
        loc="upper left",
        bbox_to_anchor=(1.02, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(SM_SHARED, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=7, length=2)
    return cb


def l2norm_np(X):
    X = np.asarray(X, np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def ef_text_axis(Xt, yt):
    Xt_n = l2norm_np(Xt)
    pc1 = PCA(n_components=1, random_state=0).fit(Xt_n).components_[0]
    scores = Xt_n @ pc1
    if np.corrcoef(scores, yt)[0, 1] < 0:
        pc1 = -pc1
    return pc1 / (np.linalg.norm(pc1) + 1e-12)


def align_visuals_to_axis(Xv, yv, u):
    Xv_n = l2norm_np(Xv)
    proj = Xv_n @ u
    r = float(np.corrcoef(proj, yv)[0, 1])
    rho = float(spearmanr(proj, yv).correlation)
    reg = LinearRegression().fit(yv.reshape(-1, 1), proj)
    r2 = float(reg.score(yv.reshape(-1, 1), proj))
    return {"pearson_r": r, "spearman_rho": rho, "r2": r2, "proj": proj}


def load_pt_embedding(path):
    obj = torch.load(path, map_location="cpu")

    def _as_vec(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t.float()
        while t.ndim > 1:
            t = t.mean(dim=0)
        return t.float()

    if isinstance(obj, dict):
        for k in ["embedding_pooled", "embedding", "feat", "features", "emb", "z"]:
            if k in obj and torch.is_tensor(obj[k]):
                return _as_vec(obj[k])
        for v in obj.values():
            if torch.is_tensor(v):
                return _as_vec(v)
    if torch.is_tensor(obj):
        return _as_vec(obj)
    raise ValueError(f"Unsupported .pt format: {path}")


def load_text_dir(text_dir):
    if not os.path.isdir(text_dir):
        print(f"[warn] text dir not found: {text_dir}")
        return None, None
    files = sorted(glob.glob(os.path.join(text_dir, "*.pt")))
    if not files:
        print(f"[warn] no text .pt files in: {text_dir}")
        return None, None
    X, y = [], []
    for p in files:
        try:
            emb = load_pt_embedding(p)
        except Exception as e:
            print(f"[warn] failed to load text emb: {p} ({e})")
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"ef_(\d+)", base)
        ef = int(m.group(1)) if m else int(re.findall(r"\d+", base)[0])
        X.append(emb)
        y.append(ef)
    if not X:
        return None, None
    return torch.stack(X, 0).numpy(), np.asarray(y, float)


def load_visual_dir(visual_dir, df_test, id_getter):
    if not os.path.isdir(visual_dir):
        print(f"[warn] visual dir not found: {visual_dir}")
        return None, None, None
    pt_files = glob.glob(os.path.join(visual_dir, "*.pt"))
    name2path = {os.path.splitext(os.path.basename(p))[0]: p for p in pt_files}
    keep_rows, paths = [], []
    for _, row in df_test.iterrows():
        key = str(id_getter(row))
        if key in name2path:
            keep_rows.append(row)
            paths.append(name2path[key])
    if not keep_rows:
        print(f"[warn] no matching .pt for df_test ids in: {visual_dir}")
        return None, None, None
    sub = pd.DataFrame(keep_rows).reset_index(drop=True)
    embs = []
    for p in paths:
        try:
            embs.append(load_pt_embedding(p))
        except Exception as e:
            print(f"[warn] failed to load visual emb: {p} ({e})")
    if not embs:
        return None, None, None
    X = torch.stack(embs, 0).numpy()
    y = sub.loc[: len(X) - 1, "EF"].to_numpy()
    ids = [str(id_getter(r)) for _, r in sub.loc[: len(X) - 1, :].iterrows()]
    return X, y, ids


def load_dataset_payloads():
    payloads = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n=== Loading dataset: {ds_name} ===")
        df = pd.read_csv(cfg["csv"])
        if "EF" not in df.columns:
            raise ValueError(f"EF column not found in {cfg['csv']}")
        df["EF"] = pd.to_numeric(df["EF"], errors="coerce")
        df = df.dropna(subset=["EF"]).reset_index(drop=True)

        df_test = cfg["split_fn"](df)
        print(f"{ds_name}: test size = {len(df_test)}")

        payloads[ds_name] = {}
        for model in MODELS:
            vdir = cfg["visual"][model]
            tdir = cfg["text"][model]
            Xv, yv, ids = load_visual_dir(vdir, df_test, cfg["id_getter"])
            Xt, yt = load_text_dir(tdir)
            payloads[ds_name][model] = {"visual": (Xv, yv, ids), "text": (Xt, yt)}
            vshape = None if Xv is None else Xv.shape
            tshape = None if Xt is None else Xt.shape
            print(f"  {model:10s} | visual {vshape} | text {tshape}")
    return payloads


payloads = load_dataset_payloads()

n_rows, n_cols = len(DATASETS), len(MODELS)
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True
)
if n_rows == 1:
    axes = axes[np.newaxis, :]
if n_cols == 1:
    axes = axes[:, np.newaxis]

for r, (ds_name, _) in enumerate(DATASETS.items()):
    for c, model in enumerate(MODELS):
        ax = axes[r, c]
        Xv, yv, _ = payloads[ds_name][model]["visual"]
        if Xv is not None and len(Xv) > 1:
            Z = embed2d(Xv)
            ax.scatter(
                Z[:, 0],
                Z[:, 1],
                s=12,
                c=np.clip(yv, 0, 100),
                cmap=soft_winter,
                norm=EF_NORM,
            )
            add_inset_cbar_shared(fig, ax, "EF")
        else:
            ax.text(0.5, 0.5, "No visual", ha="center", va="center")
        if c == 0:
            ax.set_ylabel(ds_name, fontsize=14, fontweight="bold")
        ax.set_title(model, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

fig.suptitle(
    "Visual Embedding Spaces Across EF Datasets", fontsize=16, fontweight="bold", y=1.02
)
plt.savefig(
    os.path.join(OUTPUT_DIR, "visual_embeddings.png"), dpi=220, bbox_inches="tight"
)
plt.savefig(os.path.join(OUTPUT_DIR, "visual_embeddings.pdf"), bbox_inches="tight")
print(f"{OUTPUT_DIR}/visual_embeddings.(png|pdf)")

n_rows = 1 + len(DATASETS)
n_cols = len(MODELS)

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(4.8 * n_cols, 4.6 * n_rows), constrained_layout=False
)
plt.subplots_adjust(hspace=0.6, wspace=0.3)

if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = np.array([axes])
elif n_cols == 1:
    axes = axes.reshape(n_rows, 1)
for c, model in enumerate(MODELS):
    ax_text = axes[0, c]
    first_ds = list(DATASETS.keys())[0]
    Xt, yt = payloads[first_ds][model]["text"]
    if Xt is not None and len(Xt) > 1:
        Zt = embed2d(Xt)
        sc = ax_text.scatter(
            Zt[:, 0],
            Zt[:, 1],
            s=12,
            c=np.clip(yt, 0, 100),
            cmap=soft_winter,
            norm=EF_NORM,
        )
        add_inset_cbar_shared(fig, ax_text, "EF Prompt")
    else:
        ax_text.text(0.5, 0.5, "No text", ha="center", va="center")

    ax_text.set_title(model, fontsize=16, fontweight="bold")
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_text.set_box_aspect(1)

axes[0, 0].set_ylabel("Text", fontsize=14, fontweight="bold", rotation=90)

for r, (ds_name, _) in enumerate(DATASETS.items(), start=1):
    axes[r, 0].set_ylabel(ds_name, fontsize=14, fontweight="bold", rotation=90)
    for c, model in enumerate(MODELS):
        ax = axes[r, c]
        Xv, yv, _ = payloads[ds_name][model]["visual"]
        Xt, yt = payloads[ds_name][model]["text"]
        ok = (
            Xt is not None
            and Xv is not None
            and len(yt) == 101
            and Xv.shape[1] == Xt.shape[1]
        )
        if ok:
            u = ef_text_axis(Xt, yt)
            stats = align_visuals_to_axis(Xv, yv, u)
            ax.scatter(
                yv,
                stats["proj"],
                s=8,
                c=np.clip(yv, 0, 100),
                cmap=soft_winter,
                norm=EF_NORM,
            )
            ax.set_title(f"r={stats['pearson_r']:.2f}", fontsize=14, weight="bold")
        else:
            ax.text(0.5, 0.5, "Skipped", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)


fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.96])

plt.savefig(
    os.path.join(OUTPUT_DIR, "text_align_grid.png"), dpi=220, bbox_inches="tight"
)
plt.savefig(os.path.join(OUTPUT_DIR, "text_align_grid.pdf"), bbox_inches="tight")
