import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import PAH_ROOT, PAH_OUT_DIR

PAH_OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_nii(p: Path) -> bool:
    return (p.suffix == ".nii") or ("".join(p.suffixes[-2:]) == ".nii.gz")


def patient_from_path(p: Path) -> str:
    base = p.stem  # e.g., "patient-2-4_image"
    if base.startswith("patient-"):
        parts = base.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])  # -> "patient-2"
    return base


rows = []
for cls in ["PAH", "Non-PAH"]:
    cls_dir = PAH_ROOT / cls
    if not cls_dir.exists():
        continue
    for p in cls_dir.rglob("*"):
        if p.is_file() and is_nii(p) and "label" not in p.name.lower():
            label = 1 if cls == "PAH" else 0
            rows.append(
                {
                    "path": str(p),
                    "PAH": label,
                    "patient_id": patient_from_path(p),
                }
            )

df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
if df.empty:
    raise RuntimeError(
        "No .nii or .nii.gz files found in PAH/Non-PAH folders (after filtering)."
    )

print("Found counts (files):\n", df["PAH"].value_counts(dropna=False))

per_patient_sets = df.groupby("patient_id")["PAH"].agg(lambda x: set(x))
mixed = per_patient_sets[per_patient_sets.apply(lambda s: len(s) > 1)]
if len(mixed) > 0:
    print(f"[WARN] Mixed-label patients: {len(mixed)}. Resolving via max (any 1 -> 1).")
    patient_labels = (
        df.groupby("patient_id")["PAH"].max().rename("PAH_patient").reset_index()
    )
else:
    patient_labels = (
        df.groupby("patient_id")["PAH"].first().rename("PAH_patient").reset_index()
    )

use_stratify = patient_labels["PAH_patient"].nunique() > 1

trainval_pat, test_pat = train_test_split(
    patient_labels,
    test_size=0.20,
    stratify=patient_labels["PAH_patient"] if use_stratify else None,
    random_state=42,
)

val_ratio_of_trainval = 0.10 / 0.80
train_pat, val_pat = train_test_split(
    trainval_pat,
    test_size=val_ratio_of_trainval,
    stratify=trainval_pat["PAH_patient"] if use_stratify else None,
    random_state=42,
)


def assign_split(d: pd.DataFrame, pats: pd.DataFrame, name: str) -> pd.DataFrame:
    out = d[d["patient_id"].isin(pats["patient_id"])].copy()
    out["split"] = name
    return out


df = df.merge(patient_labels, on="patient_id", how="left")

train_df = assign_split(df, train_pat, "train")
val_df = assign_split(df, val_pat, "val")
test_df = assign_split(df, test_pat, "test")

final = pd.concat([train_df, val_df, test_df], ignore_index=True)

overlap = (
    (set(train_df["patient_id"]) & set(val_df["patient_id"]))
    | (set(train_df["patient_id"]) & set(test_df["patient_id"]))
    | (set(val_df["patient_id"]) & set(test_df["patient_id"]))
)

final[["path", "PAH", "patient_id", "split"]].to_csv(
    PAH_OUT_DIR / "pah_split.csv", index=False
)


def report(d, name):
    print(f"\n=== {name} ===")
    print(d["PAH"].value_counts().rename({1: "PAH=1", 0: "PAH=0"}))
    print("Patients:", d["patient_id"].nunique(), " Files:", len(d))


print("Total patients:", patient_labels["patient_id"].nunique())
report(train_df, "Train (70%)")
report(val_df, "Val (10%)")
report(test_df, "Test (20%)")
