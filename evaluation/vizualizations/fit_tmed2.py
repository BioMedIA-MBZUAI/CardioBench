import os, glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

CSV_PATH = ""  # path to gt, e.g. ./TMED_2/tmed2.csv
MODEL_DIRS = {
    "BioMedClip": "/embeddings/TMED_2/BioMedClip",
    "DinoV3": "/embeddings/TMED_2/DinoV3",
    "EchoClip": "/embeddings/TMED_2/EchoClip",
    "EchoPrime": "/embeddings/TMED_2/EchoPrime",
    "PanEcho": "/embeddings/TMED_2/PanEcho",
    "SigLip2": "/embeddings/TMED_2/SigLip2",
}
EMBED_KEYS_TRY = ["embedding", "emb", "feat", "features"]
VIEW_CLASS_NAMES = ["A2C", "A4C", "PLAX", "PSAX", "A4CorA2CorOther"]

USE_COSINE = True
CV_FOLDS = 5
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21]

OUT_DIR = "./knn_view_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def find_pt_path(model_dir: str, query_key: str, file_path: str) -> str | None:
    base_png = Path(file_path).name if file_path else Path(query_key).name
    if not base_png:
        base_png = query_key
    stem = Path(base_png).stem

    cands = [
        os.path.join(model_dir, f"{base_png}.pt"),
        os.path.join(model_dir, f"{stem}.pt"),
        os.path.join(model_dir, f"{Path(base_png).name}.pt"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c

    g1 = glob.glob(os.path.join(model_dir, f"**/*{stem}.pt"), recursive=True)
    if len(g1) == 1:
        return g1[0]
    if len(g1) > 1:
        ends = [p for p in g1 if p.endswith(f"{base_png}.pt")]
        if len(ends) == 1:
            return ends[0]
        return sorted(g1, key=len)[0]

    g2 = glob.glob(os.path.join(model_dir, f"**/*{base_png}.pt"), recursive=True)
    if len(g2) == 1:
        return g2[0]
    if len(g2) > 1:
        return sorted(g2, key=len)[0]
    return None


def load_tensor(pt_path: str) -> torch.Tensor | None:
    try:
        obj = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] load fail {pt_path}: {e}")
        return None
    t = None
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        for k in EMBED_KEYS_TRY:
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                break
        if t is None:
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    t = v
                    break
    if t is None:
        print(f"no tensor found in {pt_path}")
        return None
    if t.ndim > 1:
        t = t.view(-1)
    return t.float()


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def finite_mask(X: np.ndarray) -> np.ndarray:
    return np.isfinite(X).all(axis=1)


if __name__ == "__main__":
    gt = pd.read_csv(CSV_PATH)
    need = {"query_key", "view", "diagnosis_label", "split", "file_path"}
    miss = need - set(gt.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")
    gt = gt[gt["view"].isin(VIEW_CLASS_NAMES)].reset_index(drop=True)

    gt["split"] = gt["split"].astype(str)
    valid_splits = {"train", "val", "test"}
    if not set(gt["split"]).issubset(valid_splits):
        bad = sorted(set(gt["split"]) - valid_splits)
        raise ValueError(f"Unknown split values in CSV: {bad}")

    le = LabelEncoder()
    le.fit(VIEW_CLASS_NAMES)
    y_all = le.transform(gt["view"].values)

    results = []
    for model_name, model_dir in MODEL_DIRS.items():
        print(f"\n### Loading {model_name} from {model_dir}")
        embs = []
        miss_count = 0
        for _, row in tqdm(gt.iterrows(), total=len(gt)):
            pt_path = find_pt_path(
                model_dir, str(row["query_key"]), str(row["file_path"])
            )
            if pt_path is None:
                embs.append(None)
                miss_count += 1
                continue
            t = load_tensor(pt_path)
            if t is None:
                embs.append(None)
                miss_count += 1
                continue
            embs.append(t.numpy())

        dims = [e.shape[0] for e in embs if e is not None]
        if not dims:
            print(f"{model_name}: no embeddings found; skipping")
            continue
        dim = max(dims)
        X = np.full((len(embs), dim), np.nan, dtype=np.float32)
        for i, e in enumerate(embs):
            if e is None:
                continue
            d = min(dim, e.shape[0])
            X[i, :d] = e[:d]

        mask_ok = finite_mask(X)
        X = X[mask_ok]
        y = y_all[mask_ok]
        split_m = gt.loc[mask_ok, "split"].values
        is_trainval_m = np.isin(split_m, ["train", "val"])
        is_test_m = split_m == "test"

        X = l2_normalize_rows(X)

        X_tr, y_tr = X[is_trainval_m], y[is_trainval_m]
        X_te, y_te = X[is_test_m], y[is_test_m]

        metric = "cosine" if USE_COSINE else "euclidean"
        knn = KNeighborsClassifier(metric=metric, weights="uniform")
        binc = np.bincount(y_tr, minlength=len(VIEW_CLASS_NAMES))
        min_class = int(max(2, np.min(binc[binc > 0]) if np.any(binc > 0) else 2))
        folds = min(CV_FOLDS, min_class)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        valid_ks = [k for k in K_GRID if k <= len(y_tr)]
        if not valid_ks:
            valid_ks = [1]

        gs = GridSearchCV(
            knn,
            param_grid={"n_neighbors": valid_ks},
            scoring="accuracy",
            cv=cv,
            refit=True,
            n_jobs=-1,
        )
        gs.fit(X_tr, y_tr)
        best_k = gs.best_params_["n_neighbors"]
        clf = gs.best_estimator_

        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")

        labels_te = np.unique(y_te)
        names_te = [VIEW_CLASS_NAMES[i] for i in labels_te]

        print(f"\n=== {model_name} ===")
        print(
            f"Usable rows: {len(X)} | Missing rows: {int((~mask_ok).sum())} | "
            f"Train+Val: {len(y_tr)} | Test: {len(y_te)} | Best k: {best_k} | Metric: {metric}"
        )

        rep = classification_report(
            y_te,
            y_pred,
            labels=labels_te,
            target_names=names_te,
            digits=3,
            zero_division=0,
            output_dict=True,
        )
        print(pd.DataFrame(rep).T.to_string())
        cm = confusion_matrix(y_te, y_pred, labels=labels_te)
        print("Confusion matrix (rows=GT, cols=Pred):\n", cm)

        cm_path = os.path.join(OUT_DIR, f"confusion_{model_name}.npy")
        np.save(cm_path, cm)
        cm_rowsum = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / np.maximum(cm_rowsum, 1)
        cmn_path = os.path.join(OUT_DIR, f"confusion_{model_name}_norm.npy")
        np.save(cmn_path, cm_norm)

        rec_from_cm = {}
        for i, cname in enumerate(names_te):
            denom = cm[i].sum()
            rec_from_cm[cname] = (cm[i, i] / denom) if denom > 0 else np.nan

        row = {
            "model": model_name,
            "usable_rows": int(len(X)),
            "best_k": int(best_k),
            "metric": metric,
            "acc": float(acc),
            "f1_macro": float(f1m),
            "train_classes": ",".join([VIEW_CLASS_NAMES[i] for i in present_tr]),
            "test_classes": ",".join([VIEW_CLASS_NAMES[i] for i in present_te]),
        }

        for cls in VIEW_CLASS_NAMES:
            recall_val = rec_from_cm.get(cls, np.nan)
            f1_val = (rep.get(cls, {}) or {}).get("f1-score", np.nan)
            row[f"{cls}_acc"] = float(recall_val) if np.isfinite(recall_val) else np.nan
            row[f"{cls}_f1"] = float(f1_val) if np.isfinite(f1_val) else np.nan

        results.append(row)

    if results:
        df = pd.DataFrame(results)

        df_f1 = df[["model", "f1_macro"]].sort_values("f1_macro", ascending=False)
        print("\n=== Summary (Model, F1-macro) ===")
        print(df_f1.to_string(index=False))
        f1_csv = os.path.join(OUT_DIR, "knn_view_results.csv")
        df_f1.to_csv(f1_csv, index=False)
