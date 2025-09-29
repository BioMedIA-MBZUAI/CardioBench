import os
import glob
import pandas as pd
from config import CAMUS_ROOT, CAMUS_SPLIT_DIR, CAMUS_OUT_CSV


def load_split(path, split_name):
    patients = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                patients.append(line)
    return {p: split_name for p in patients}


split_map = {}
split_map.update(
    load_split(os.path.join(CAMUS_SPLIT_DIR, "subgroup_training.txt"), "train")
)
split_map.update(
    load_split(os.path.join(CAMUS_SPLIT_DIR, "subgroup_validation.txt"), "val")
)
split_map.update(
    load_split(os.path.join(CAMUS_SPLIT_DIR, "subgroup_testing.txt"), "test")
)

rows = []
for patient_dir in sorted(glob.glob(os.path.join(CAMUS_ROOT, "patient*"))):
    patient_id = os.path.basename(patient_dir)
    split = split_map.get(patient_id, "unknown")

    for view in ["2CH", "4CH"]:
        nii_path = os.path.join(
            patient_dir, f"{patient_id}_{view}_half_sequence.nii.gz"
        )
        cfg_path = os.path.join(patient_dir, f"Info_{view}.cfg")

        if not os.path.exists(nii_path) or not os.path.exists(cfg_path):
            continue

        info = {}
        with open(cfg_path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = [x.strip() for x in line.split(":", 1)]
                info[key] = val

        rows.append(
            {
                "patient_id": patient_id,
                "view": view,
                "path": nii_path,
                "split": split,
                "Sex": info.get("Sex", ""),
                "Age": info.get("Age", ""),
                "EF": info.get("EF", ""),
                "FrameRate": info.get("FrameRate", ""),
                "ImageQuality": info.get("ImageQuality", ""),
            }
        )

df = pd.DataFrame(rows)
print(df.head())

df.to_csv(CAMUS_OUT_CSV, index=False)
