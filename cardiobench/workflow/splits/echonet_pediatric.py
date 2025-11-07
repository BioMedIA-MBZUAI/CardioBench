from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from cardiobench.workflow.config import ensure_parent_dir, expand_path, load_config


def main(config_path: str | Path | None = None) -> Path:
    config: Dict[str, Dict[str, dict]] = load_config(config_path)
    pediatric_cfg = config.get("echonet", {}).get("pediatric", {})

    a4c = expand_path(pediatric_cfg.get("a4c_filelist"))
    psax = expand_path(pediatric_cfg.get("psax_filelist"))
    out_csv = ensure_parent_dir(expand_path(pediatric_cfg.get("combined_output")))

    a4c_df = pd.read_csv(a4c)
    psax_df = pd.read_csv(psax)

    a4c_df["view"] = "A4C"
    psax_df["view"] = "PSAX"

    combined = pd.concat([a4c_df, psax_df], ignore_index=True)
    combined.to_csv(out_csv, index=False)

    print(f"Wrote combined pediatric filelist ({len(combined)} rows) to {out_csv}")
    return out_csv


if __name__ == "__main__":
    main()
