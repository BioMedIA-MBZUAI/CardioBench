"""Helpers for locating dataset assets required by the CardioBench workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_ENV_VAR = "CARDIOBENCH_DATA_CONFIG"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "datasets.json"
EXAMPLE_CONFIG_PATH = _REPO_ROOT / "configs" / "datasets.example.json"


def load_config(config_path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load the JSON dataset configuration used by the workflow scripts."""
    candidate = Path(
        config_path or os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    )

    if not candidate.exists():
        example_rel = _safe_relative(EXAMPLE_CONFIG_PATH)
        expected_rel = _safe_relative(candidate)
        raise FileNotFoundError(
            "Dataset config not found at "
            f"{expected_rel}. Copy `{example_rel}` to `{expected_rel}` and fill in your paths, "
            "or point the `CARDIOBENCH_DATA_CONFIG` environment variable at a JSON file."
        )

    with candidate.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def expand_path(value: str | None, allow_empty: bool = False) -> Path | None:
    """Turn a config value into a Path, expanding ~ and environment variables."""
    if value in (None, ""):
        if allow_empty:
            return None
        raise ValueError("Required path value missing from dataset config")

    expanded = os.path.expanduser(os.path.expandvars(value))
    return Path(expanded)


def ensure_parent_dir(path: Path) -> Path:
    """Ensure the parent directory for a file path exists and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)
