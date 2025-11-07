from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict

import pandas as pd

from cardiobench.workflow.config import ensure_parent_dir, expand_path, load_config


CAMUS_SPLIT_FILES = {
    "train": "subgroup_training.txt",
    "val": "subgroup_validation.txt",
    "test": "subgroup_testing.txt",
}


def _load_split(path: Path, split_name: str) -> Dict[str, str]:
    patients: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing CAMUS split file: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            patients[line] = split_name
    return patients


def _camus_info(cfg_path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        info[key] = val
    return info


def main(config_path: str | Path | None = None) -> Path:
    config = load_config(config_path)
    camus_cfg = config.get("camus", {})

    camus_root = expand_path(camus_cfg.get("root"))
    split_dir = expand_path(camus_cfg.get("split_dir"))
    out_csv = ensure_parent_dir(expand_path(camus_cfg.get("output_csv")))

    split_map: Dict[str, str] = {}
    for split_name, filename in CAMUS_SPLIT_FILES.items():
        split_map.update(_load_split(split_dir / filename, split_name))

    rows = []
    for patient_dir_str in sorted(glob.glob(str(camus_root / "patient*"))):
        patient_dir = Path(patient_dir_str)
        patient_id = patient_dir.name
        split = split_map.get(patient_id, "unknown")

        for view in ("2CH", "4CH"):
            nii_path = patient_dir / f"{patient_id}_{view}_half_sequence.nii.gz"
            cfg_path = patient_dir / f"Info_{view}.cfg"

            if not nii_path.exists() or not cfg_path.exists():
                continue

            info = _camus_info(cfg_path)

            rows.append(
                {
                    "patient_id": patient_id,
                    "view": view,
                    "path": str(nii_path),
                    "split": split,
                    "Sex": info.get("Sex", ""),
                    "Age": info.get("Age", ""),
                    "EF": info.get("EF", ""),
                    "FrameRate": info.get("FrameRate", ""),
                    "ImageQuality": info.get("ImageQuality", ""),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No CAMUS sequences found. Check CAMUS paths in the dataset config."
        )

    df["unique_id"] = df["patient_id"] + "_" + df["view"]

    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    written = main()
    print(f"Wrote CAMUS split to {written}")
