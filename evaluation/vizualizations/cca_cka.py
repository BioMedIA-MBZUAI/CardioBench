import numpy as np
import pandas as pd
import os, glob, sys
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.preprocessing import normalize as l2norm
from sklearn.cross_decomposition import CCA

SEED = 42
CSV_PATH = ""  # path to split, e.g. ./CAMUS_public/camus_split.csv
EMB_ROOT = (
    ""  # path to embedding dir, e.g. ./embeddings/CAMUS with /{model_name} inside
)
OUT_DIR = ""  # cca_cka
os.makedirs(OUT_DIR, exist_ok=True)


def list_model_files(model_name, default_root=EMB_ROOT):
    model_dir = os.path.join(default_root, model_name)
    if not os.path.isdir(model_dir):
        print(f"Directory not found for {model_name}: {model_dir}")
        return pd.DataFrame(columns=["unique_id", "embedding_path"])

    files = glob.glob(os.path.join(model_dir, "*.pt"))
    return pd.DataFrame(
        {
            "unique_id": [os.path.splitext(os.path.basename(f))[0] for f in files],
            "embedding_path": files,
        }
    )


def extract_1d_vector(data):
    t = None
    if isinstance(data, dict):
        for k in ["embedding", "embedding_pooled"]:
            if k in data and torch.is_tensor(data[k]):
                t = data[k]
                break
        if t is None:
            for v in data.values():
                if torch.is_tensor(v):
                    t = v
                    break
    elif torch.is_tensor(data):
        t = data
    if t is None:
        raise ValueError("No tensor embedding found in .pt")

    t = t.detach().cpu().float()
    # (1, D) -> (D,)
    if t.ndim > 1 and t.shape[0] == 1:
        t = t.squeeze(0)
    # (T, D) -> average over T
    if t.ndim == 2 and t.shape[0] > 1:
        t = t.mean(0)
    # anything else >1D -> flatten
    if t.ndim > 1:
        t = t.reshape(-1)
    return t.numpy()


def load_vectors_for(df_rows):
    vecs, ids = [], []
    for _, row in df_rows.iterrows():
        try:
            data = torch.load(row["embedding_path"], map_location="cpu")
            v = extract_1d_vector(data)
            vecs.append(v)
            ids.append(row["unique_id"])
        except Exception:
            pass
    if not vecs:
        return None, None
    return np.vstack(vecs).astype(np.float32), np.array(ids)


def _prep_for_cka(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    X = l2norm(X, norm="l2", axis=1)
    return X


def linear_cka(X, Y):
    X, Y = _prep_for_cka(X), _prep_for_cka(Y)
    K, L = X @ X.T, Y @ Y.T
    hsic = np.sum(K * L)
    return float(hsic / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-12))


def paired_embeddings(df_test, model_a, model_b):
    A = list_model_files(model_a)
    B = list_model_files(model_b)
    if A.empty or B.empty:
        return None, None, None, None

    dfa = df_test.merge(A, on="unique_id", how="inner")[
        ["unique_id", "embedding_path", "EF_bin"]
    ]
    dfb = df_test.merge(B, on="unique_id", how="inner")[["unique_id", "embedding_path"]]

    Xa_raw, ids_a = load_vectors_for(
        dfa.rename(columns={"embedding_path": "embedding_path"})
    )
    Xb_raw, ids_b = load_vectors_for(
        dfb.rename(columns={"embedding_path": "embedding_path"})
    )
    if Xa_raw is None or Xb_raw is None:
        return None, None, None, None

    # align by unique_id
    da = pd.DataFrame({"unique_id": ids_a})
    db = pd.DataFrame({"unique_id": ids_b})
    common = da.merge(db, on="unique_id")
    if len(common) < 5:
        return None, None, None, None

    idx_a = {u: i for i, u in enumerate(ids_a)}
    idx_b = {u: i for i, u in enumerate(ids_b)}
    ia = [idx_a[u] for u in common["unique_id"]]
    ib = [idx_b[u] for u in common["unique_id"]]

    Xa = Xa_raw[ia, :]
    Xb = Xb_raw[ib, :]

    meta = dfa.set_index("unique_id").loc[common["unique_id"], ["EF_bin"]].reset_index()
    return Xa, Xb, meta, common


def cca_pair_2d(df_test, model_a, model_b, seed=SEED):
    Xa, Xb, meta, pair = paired_embeddings(df_test, model_a, model_b)
    if Xa is None:
        return None

    Xa_std = StandardScaler(with_mean=True, with_std=True).fit_transform(Xa)
    Xb_std = StandardScaler(with_mean=True, with_std=True).fit_transform(Xb)

    # CCA to 2D
    nc = min(2, Xa_std.shape[1], Xb_std.shape[1], len(meta) - 1)
    if nc < 2:
        nc = 2
    cca = CCA(n_components=nc, max_iter=500)
    A_c, B_c = cca.fit_transform(Xa_std, Xb_std)  # [N,2], [N,2]

    cka = linear_cka(Xa, Xb)

    dfa = meta.copy()
    dfa["x"], dfa["y"] = A_c[:, 0], A_c[:, 1]
    dfa["model"] = model_a
    dfb = meta.copy()
    dfb["x"], dfb["y"] = B_c[:, 0], B_c[:, 1]
    dfb["model"] = model_b
    df_ab = pd.concat([dfa, dfb], ignore_index=True)
    return {"df": df_ab, "cka": cka}


def run_cca(
    df_test,
    models_others,
    out_path=os.path.join(OUT_DIR, "cca_dinov3.png"),
    ref="DinoV3",
):
    results = {}
    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    for other in models_others:
        res = cca_pair_2d(df_test, ref, other)
        results[other] = res
        if res is None:
            continue
        df_ab = res["df"]
        xmins.append(df_ab["x"].min())
        xmaxs.append(df_ab["x"].max())
        ymins.append(df_ab["y"].min())
        ymaxs.append(df_ab["y"].max())

    if not xmins:
        print("No pairs available.")
        return None

    x_min, x_max = min(xmins), max(xmaxs)
    y_min, y_max = min(ymins), max(ymaxs)
    pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    X_LIM = (x_min - pad_x, x_max + pad_x)
    Y_LIM = (y_min - pad_y, y_max + pad_y)

    all_bins = []
    for res in results.values():
        if res is not None:
            all_bins.extend(res["df"]["EF_bin"].dropna().unique().tolist())
    all_bins = sorted(set(all_bins))

    def ef_mid(bin_str):
        a, b = bin_str.split("-")
        return (int(a) + int(b)) / 2.0

    cmap = mpl.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    bin_to_color = {b: cmap(norm(ef_mid(b))) for b in all_bins}

    n = len(models_others)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), sharex=True, sharey=True)

    if n == 1:
        axes = axes.reshape(2, 1)

    for col, other in enumerate(models_others):
        res = results.get(other, None)
        if res is None:
            for r in [0, 1]:
                axes[r, col].axis("off")
                axes[r, col].text(
                    0.5,
                    0.5,
                    "no overlap",
                    ha="center",
                    va="center",
                    transform=axes[r, col].transAxes,
                )
            continue

        df_ab, cka = res["df"], res["cka"]

        ax_top = axes[0, col]
        ax_top.set_xlim(*X_LIM)
        ax_top.set_ylim(*Y_LIM)
        sub = df_ab[df_ab["model"] == ref]
        for b in sorted(sub["EF_bin"].dropna().unique()):
            sb = sub[sub["EF_bin"] == b]
            ax_top.scatter(
                sb["x"], sb["y"], c=[bin_to_color[b]], s=14, marker="o", alpha=0.85
            )
        ax_top.set_title("DINOv3", fontsize=12, fontweight="bold")

        ax_bot = axes[1, col]
        ax_bot.set_xlim(*X_LIM)
        ax_bot.set_ylim(*Y_LIM)
        sub = df_ab[df_ab["model"] == other]
        for b in sorted(sub["EF_bin"].dropna().unique()):
            sb = sub[sub["EF_bin"] == b]
            ax_bot.scatter(
                sb["x"], sb["y"], c=[bin_to_color[b]], s=14, marker="o", alpha=0.85
            )
        ax_bot.set_title(f"{other}\nCKA = {cka:.3f}", fontsize=12, fontweight="bold")

    fig.text(0.5, 0.01, "Canonical dim 1", ha="center")
    fig.text(0.01, 0.5, "Canonical dim 2", va="center", rotation="vertical")

    color_handles = [
        mpatches.Patch(color=mpl.colors.to_hex(bin_to_color[b]), label=str(b))
        for b in all_bins
    ]
    fig.legend(
        handles=color_handles,
        ncol=min(len(color_handles), 8),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        title="EF bins",
    )
    fig.suptitle(
        "CCA 2D Alignment: DINOv3 vs Other Models (Test Set)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0.03, 0.06, 1, 0.96])

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), dpi=240, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    return out_path


# ============== CKA ================
def rsm(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def _prep_for_cka(X):
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


def linear_cka(X, Y):
    X = _prep_for_cka(X)
    Y = _prep_for_cka(Y)
    K, L = X @ X.T, Y @ Y.T
    hsic = np.sum(K * L)
    return float(hsic / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-12))


def show_rsms(
    df_test,
    model_a,
    model_b,
    max_n=150,
    bw_diff=False,
    abs_diff=False,
    color=None,
    rsm_vmin=0.6,
    rsm_vmax=1.0,
    diff_range=0.2,
    title_prefix="RSMs: ",
    show_cka=True,
):
    Xa, Xb, meta, _ = paired_embeddings(df_test, model_a, model_b)
    if Xa is None or Xb is None:
        print("No overlap")
        return

    if Xa.shape[0] > max_n:
        idx = np.random.RandomState(42).choice(Xa.shape[0], max_n, replace=False)
        Xa, Xb = Xa[idx], Xb[idx]

    # RSMs
    Ra, Rb = rsm(Xa), rsm(Xb)
    diff = Ra - Rb
    D = np.abs(diff) if abs_diff else diff

    cka_val = linear_cka(Xa, Xb) if show_cka else None

    if color is None:
        color = mpl.cm.get_cmap("magma")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    imA = axes[0].imshow(Ra, vmin=rsm_vmin, vmax=rsm_vmax, cmap=color)

    axes[0].set_title(f"{model_a}", fontsize=12, fontweight="bold")
    axes[0].spines[:].set_visible(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    imB = axes[1].imshow(Rb, vmin=rsm_vmin, vmax=rsm_vmax, cmap=color)
    axes[1].set_title(f"{model_b}", fontsize=12, fontweight="bold")
    axes[1].spines[:].set_visible(False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    if bw_diff:
        if abs_diff:
            imD = axes[2].imshow(D, vmin=0.0, vmax=diff_range, cmap="gray_r")
            cb_label = "|A − B|"
        else:
            imD = axes[2].imshow(D, vmin=-diff_range, vmax=diff_range, cmap="gray")
            cb_label = "A − B "
    else:
        if abs_diff:
            imD = axes[2].imshow(D, vmin=0.0, vmax=diff_range, cmap="PuBuGn")
            cb_label = "|A − B|"
        else:
            imD = axes[2].imshow(D, vmin=-diff_range, vmax=diff_range, cmap="PuBuGn")
            cb_label = "A − B "

    axes[2].set_title("Difference", fontsize=12, fontweight="bold")
    axes[2].spines[:].set_visible(False)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.colorbar(imA, ax=axes[0], fraction=0.046, pad=0.04, label="Cosine similarity")
    fig.colorbar(imB, ax=axes[1], fraction=0.046, pad=0.04, label="Cosine similarity")
    cbar = fig.colorbar(imD, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label(cb_label, rotation=90)

    if cka_val is None:
        fig.suptitle(
            f"{title_prefix} {model_a} vs {model_b}", fontsize=13, fontweight="bold"
        )
    else:
        fig.suptitle(
            f"{title_prefix} DINOv3 vs {model_b} | CKA = {cka_val:.3f}",
            fontsize=16,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"rsm_{model_a}_vs_{model_b}.png"),
        dpi=240,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUT_DIR, f"rsm_{model_a}_vs_{model_b}.pdf"),
        dpi=240,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    for col in ["unique_id", "split", "EF", "view"]:
        if col not in df.columns:
            print(f"Missing column in CSV: {col}", file=sys.stderr)
            sys.exit(1)

    df = df.dropna(subset=["EF"])

    bins = np.arange(0, 101, 10)
    labels = [f"{b:02d}-{b+9:02d}" for b in bins[:-1]]

    df["EF_bin"] = pd.cut(
        df["EF"].clip(lower=0, upper=100),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    df_test = df[df["split"].str.lower() == "test"].copy()

    models_to_compare = ["SigLIP2", "BioMedClip", "EchoClip", "EchoPrime", "PanEcho"]
    run_cca(
        df_test,
        models_to_compare,
    )

    # ====== RUN CKA ========
    color = plt.get_cmap("PuBuGn")
    show_rsms(
        df_test,
        "DinoV3",
        "BioMedClip",
        abs_diff=True,
        color=color,
        rsm_vmin=0.6,
        rsm_vmax=1.0,
    )
    show_rsms(
        df_test,
        "DinoV3",
        "EchoClip",
        abs_diff=True,
        color=color,
        rsm_vmin=0.6,
        rsm_vmax=1.0,
    )
