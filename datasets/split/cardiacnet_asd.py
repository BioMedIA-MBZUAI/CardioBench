import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import ASD_ROOT, ASD_OUT_DIR


def is_nii(p: Path) -> bool:
    return (p.suffix == ".nii") or ("".join(p.suffixes[-2:]) == ".nii.gz")


ASD_OUT_DIR.mkdir(parents=True, exist_ok=True)


def patient_from_path(p: Path) -> str:
    base = p.stem
    if base.startswith("patient-"):
        parts = base.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
    return base


rows = []
for cls in ["ASD", "Non-ASD"]:
    cls_dir = ASD_ROOT / cls
    if not cls_dir.exists():
        continue
    for p in cls_dir.rglob("*"):
        if p.is_file() and is_nii(p) and "label" not in p.name.lower():
            label = 1 if cls.lower() == "asd" else 0
            rows.append(
                {"path": str(p), "ASD": label, "patient_id": patient_from_path(p)}
            )

df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
if df.empty:
    raise RuntimeError(
        "No .nii or .nii.gz files found in ASD/Non-ASD folders (after filtering)."
    )

label_per_patient = df.groupby("patient_id")["ASD"].agg(lambda x: set(x))
mixed = label_per_patient[label_per_patient.apply(lambda s: len(s) > 1)]
if len(mixed) > 0:
    print(
        f"[WARN] Patients with mixed labels found ({len(mixed)}). "
        f"Resolving by setting ASD=1 if any file is ASD=1 for that patient."
    )
    resolved_label = df.groupby("patient_id")["ASD"].max().rename("ASD_patient")
else:
    resolved_label = df.groupby("patient_id")["ASD"].first().rename("ASD_patient")

patients_df = resolved_label.reset_index()

use_stratify = patients_df["ASD_patient"].nunique() > 1

trainval_pat, test_pat = train_test_split(
    patients_df,
    test_size=0.20,
    stratify=patients_df["ASD_patient"] if use_stratify else None,
    random_state=42,
)

val_ratio_of_trainval = 0.10 / 0.80
train_pat, val_pat = train_test_split(
    trainval_pat,
    test_size=val_ratio_of_trainval,
    stratify=trainval_pat["ASD_patient"] if use_stratify else None,
    random_state=42,
)

df = df.merge(patients_df, on="patient_id", how="left")


def assign_split(d: pd.DataFrame, pats: pd.DataFrame, name: str) -> pd.DataFrame:
    out = d[d["patient_id"].isin(pats["patient_id"])].copy()
    out["split"] = name
    return out


train_df = assign_split(df, train_pat, "train")
val_df = assign_split(df, val_pat, "val")
test_df = assign_split(df, test_pat, "test")

final = pd.concat([train_df, val_df, test_df], ignore_index=True)


final[["path", "ASD", "patient_id", "split"]].to_csv(
    ASD_OUT_DIR / "asd_split.csv", index=False
)


def report(d, name):
    print(f"\n=== {name} ===")
    print(d["ASD"].value_counts().rename({1: "ASD=1", 0: "ASD=0"}))
    print("Patients:", d["patient_id"].nunique(), " Files:", len(d))


print("\nAll files (pre-split) class counts:\n", df["ASD"].value_counts(dropna=False))
print("Total patients:", patients_df["patient_id"].nunique())

report(train_df, "Train (70%)")
report(val_df, "Val (10%)")
report(test_df, "Test (20%)")
