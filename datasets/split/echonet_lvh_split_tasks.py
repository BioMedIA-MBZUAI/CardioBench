import pandas as pd
from pathlib import Path
from config import LVH_MEASUREMENT_LIST, LVH_VIDEOS_DIR

CALCS = ["LVIDd", "IVSd", "LVPWd"]

OUT_DIR = Path(LVH_MEASUREMENT_LIST).parent


def make_video_path(hashed_name: str) -> str:
    return str(LVH_VIDEOS_DIR / f"{hashed_name}.avi")


def main():
    df = pd.read_csv(LVH_MEASUREMENT_LIST)

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

    df["video_path"] = df["HashedFileName"].astype(str).map(make_video_path)

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

    for c in CALCS:
        sub = df[df["Calc"] == c].copy()
        if sub.empty:
            print(f"[warn] No rows found for {c}")
            continue

        out_csv = OUT_DIR / f"{c}_.csv"
        sub.to_csv(out_csv, index=False)
        print(f"Wrote {len(sub):,} rows to {out_csv}")


if __name__ == "__main__":
    main()
