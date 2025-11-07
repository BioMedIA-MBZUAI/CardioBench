from __future__ import annotations

from pathlib import Path

import pandas as pd

from cardiobench.workflow.config import ensure_dir, expand_path, load_config

CALCS = ["LVIDd", "IVSd", "LVPWd"]


def make_video_path(videos_dir: Path, hashed_name: str) -> str:
    return str(videos_dir / f"{hashed_name}.avi")


def main(config_path: str | Path | None = None) -> Path:
    config = load_config(config_path)
    lvh_cfg = config.get("echonet", {}).get("lvh", {})

    measurement_csv = expand_path(lvh_cfg.get("measurement_csv"))
    videos_dir = expand_path(lvh_cfg.get("videos_dir"))
    output_dir = expand_path(lvh_cfg.get("output_dir"), allow_empty=True)
    out_dir = ensure_dir(output_dir or measurement_csv.parent)

    df = pd.read_csv(measurement_csv)

    required_cols = {
        "HashedFileName",
        "Calc",
        "CalcValue",
        "Frame",
        "X1",
        "X2",
        "Y1",
        "Y2",
        "Frames",
        "FPS",
        "Width",
        "Height",
        "split",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    df["video_path"] = df["HashedFileName"].astype(str).map(
        lambda name: make_video_path(videos_dir, name)
    )

    cols = [
        "Calc",
        "CalcValue",
        "HashedFileName",
        "video_path",
        "Frame",
        "X1",
        "X2",
        "Y1",
        "Y2",
        "Frames",
        "FPS",
        "Width",
        "Height",
        "split",
    ]
    df = df[cols]

    for calc in CALCS:
        subset = df[df["Calc"] == calc].copy()
        if subset.empty:
            print(f"[warn] No rows found for {calc}")
            continue

        out_csv = out_dir / f"{calc}_split.csv"
        subset.to_csv(out_csv, index=False)
        print(f"Wrote {len(subset):,} rows to {out_csv}")

    return out_dir


if __name__ == "__main__":
    output = main()
    print(f"LVH split files available in {output}")
