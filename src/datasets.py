from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re


@dataclass
class DatasetItem:
    path: Path
    key_frame: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    kind: str
    options: Dict[str, Any] = field(default_factory=dict)
    fixed_filters: Dict[str, Any] = field(default_factory=dict)


class DatasetLoader:
    def __init__(self, registry: Optional[Dict[str, DatasetSpec]] = None):
        self.registry = registry or DEFAULT_DATASET_REGISTRY
        self._loaders = {
            "Dynamic": self._load_echonet_filelist,
            "Pediatric": self._load_echonet_ped,
            "csv_table": self._load_csv_table,
            "echonet_lvh_measurements": self._load_echo_lvh,
            "tmed2_csv": self._load_tmed2,
        }

    def available(self) -> List[str]:
        return sorted(self.registry.keys())

    def load(
        self,
        dataset: str,
        *,
        root: str | Path,
        split: Optional[str] = None,
        view: Optional[str] = None,
        modality: Optional[str] = None,
        fold: Optional[int] = None,
        split_csv: Optional[str | Path] = None,
    ) -> List[DatasetItem]:
        if dataset not in self.registry:
            raise KeyError(f"Unknown dataset '{dataset}'. Available: {self.available()}")
        spec = self.registry[dataset]
        loader = self._loaders.get(spec.kind)
        if loader is None:
            raise KeyError(f"No loader registered for kind='{spec.kind}'")

        filters = dict(spec.fixed_filters)
        if split is not None:
            filters["split"] = split
        if view is not None:
            filters["view"] = view
        if modality is not None:
            filters["modality"] = modality
        if fold is not None:
            filters["fold"] = fold

        items = loader(spec, Path(root), filters, split_csv)
        if not items:
            raise FileNotFoundError(f"No items discovered for dataset='{dataset}' with filters={filters}")
        return items

    # ------------------------------------------------------------------
    # Loader implementations

    def _resolve_csv_path(
        self,
        root: Path,
        split_csv: Optional[str | Path],
        default_name: str,
    ) -> Path:
        if split_csv is not None:
            csv_path = Path(split_csv)
            if not csv_path.is_absolute():
                csv_path = root / csv_path
        else:
            csv_path = root / default_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find CSV at {csv_path}")
        return csv_path

    def _load_echonet_filelist(
        self,
        spec: DatasetSpec,
        root: Path,
        filters: Dict[str, Any],
        split_csv: Optional[str | Path],
    ) -> List[DatasetItem]:
        opts = spec.options
        csv_name = opts.get("csv_name", "FileList.csv")
        videos_subdir = opts.get("videos_subdir", "Videos")
        split_col = opts.get("split_column", "Split")
        fname_col = opts.get("filename_column", "FileName")
        split_value = (filters.get("split") or opts.get("default_split"))
        split_value_norm = split_value.upper() if isinstance(split_value, str) else None

        csv_path = self._resolve_csv_path(root, split_csv, csv_name)
        videos_dir = root / videos_subdir
        extensions = tuple(opts.get("extensions", [".avi", ".mp4"]))

        items: List[DatasetItem] = []
        seen: set[Path] = set()
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if split_value_norm is not None:
                    row_split = (row.get(split_col) or "").strip().upper()
                    if row_split != split_value_norm:
                        continue
                fname = (row.get(fname_col) or "").strip()
                if not fname:
                    continue
                path = videos_dir / fname
                if path.suffix.lower() not in extensions:
                    for ext in extensions:
                        candidate = videos_dir / f"{fname}{ext}"
                        if candidate.exists():
                            path = candidate
                            break
                if not path.exists():
                    continue
                path = path.resolve()
                if path in seen:
                    continue
                seen.add(path)
                metadata = {
                    "split": (row.get(split_col) or "").strip(),
                    "view": None,
                    "modality": None,
                }
                items.append(DatasetItem(path=path, key_frame=0, metadata={k: v for k, v in metadata.items() if v}))
        return items

    def _load_echonet_ped(
        self,
        spec: DatasetSpec,
        root: Path,
        filters: Dict[str, Any],
        split_csv: Optional[str | Path],
    ) -> List[DatasetItem]:
        opts = spec.options
        raw_view = filters.get("view") or spec.fixed_filters.get("view")
        if raw_view is None:
            candidate_views = opts.get("available_views") or []
            if not candidate_views:
                raise ValueError("echonet_ped loader requires a view (e.g., A4C, PSAX)")
            views = [str(v).upper() for v in candidate_views]
        else:
            if isinstance(raw_view, str):
                views = [v.strip().upper() for v in raw_view.split(",") if v and v.strip()]
            else:
                try:
                    views = [str(v).upper() for v in raw_view]
                except TypeError as exc:
                    raise ValueError("Invalid view specification for echonet_ped") from exc
            if not views:
                raise ValueError("echonet_ped loader requires at least one view value")

        csv_name = opts.get("csv_name", "FileList.csv")
        videos_subdir = opts.get("videos_subdir", "Videos")
        fold_col = opts.get("fold_column", "Split")
        fname_col = opts.get("filename_column", "FileName")
        fold_value = int(filters.get("fold", opts.get("default_fold", 0)))
        extensions = tuple(opts.get("extensions", [".avi", ".mp4"]))

        items: List[DatasetItem] = []
        seen: set[Path] = set()

        for view in views:
            view_root = root / view
            try:
                csv_path = self._resolve_csv_path(view_root, split_csv, csv_name)
            except FileNotFoundError:
                continue
            videos_dir = view_root / videos_subdir

            with open(csv_path, "r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    split_raw = (row.get(fold_col) or "").strip()
                    try:
                        if int(split_raw) != fold_value:
                            continue
                    except Exception:
                        continue
                    fname = (row.get(fname_col) or "").strip()
                    if not fname:
                        continue
                    path = videos_dir / fname
                    if path.suffix.lower() not in extensions:
                        for ext in extensions:
                            candidate = videos_dir / f"{fname}{ext}"
                            if candidate.exists():
                                path = candidate
                                break
                    if not path.exists():
                        continue
                    path = path.resolve()
                    if path in seen:
                        continue
                    seen.add(path)
                    metadata = {
                        "split": str(fold_value),
                        "view": view,
                        "modality": None,
                    }
                    items.append(DatasetItem(path=path, key_frame=0, metadata=metadata))
        if not items:
            raise FileNotFoundError(
                f"No pediatric items found under views {views} at root {root}."
            )
        return items

    def _load_csv_table(
        self,
        spec: DatasetSpec,
        root: Path,
        filters: Dict[str, Any],
        split_csv: Optional[str | Path],
    ) -> List[DatasetItem]:
        opts = spec.options
        csv_name = opts.get("csv_name", "split.csv")
        csv_path = self._resolve_csv_path(root, split_csv, csv_name)

        path_col = opts.get("path_column", "path")
        split_col = opts.get("split_column")
        view_col = opts.get("view_column")
        modality_col = opts.get("modality_column")
        metadata_cols: Iterable[str] = opts.get("metadata_columns", [])
        key_frame_cols: Iterable[str] = opts.get("key_frame_columns", ["first_annotated_frame_index", "Frame", "key_frame"])
        default_kf = int(opts.get("default_key_frame", 0))
        allowed_suffixes = tuple(opts.get("allowed_suffixes", [".avi", ".mp4", ".nii", ".nii.gz"]))
        exclude_tokens = tuple(opts.get("exclude_tokens", ["_mask", "_seg", "_label", "segmentation"]))
        required_values: Dict[str, Any] = opts.get("required_values", {})
        infer_view_from_parent = bool(opts.get("infer_view_from_parent", False))
        view_regex = opts.get("view_regex", r"(A2C|A3C|A4C|PSAX|PLAX)")
        known_views = tuple(opts.get("known_views", ["A2C", "A3C", "A4C", "PSAX", "PLAX"]))
        return_key_frame = bool(opts.get("return_key_frame", True))

        target_split = filters.get("split") or opts.get("default_split")
        target_split_norm = target_split.lower() if isinstance(target_split, str) else None
        target_view = (filters.get("view") or opts.get("default_view"))
        target_view_norm = target_view.upper() if isinstance(target_view, str) else None
        target_modality = (filters.get("modality") or opts.get("default_modality"))
        target_modality_norm = target_modality.upper() if isinstance(target_modality, str) else None

        def _infer_view(path: Path, existing: str) -> str:
            if existing:
                return existing
            parts = [p.upper() for p in path.parts]
            for part in parts[::-1]:
                if part in known_views:
                    return part
            if view_regex:
                match = re.search(view_regex, path.name.upper())
                if match:
                    return match.group(1)
            return existing

        items: List[DatasetItem] = []
        seen: set[Path] = set()
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if split_col:
                    row_split = (row.get(split_col) or "").strip().lower()
                    if target_split_norm and row_split != target_split_norm:
                        continue

                skip = False
                for key, allowed in required_values.items():
                    val = (row.get(key) or "").strip()
                    if isinstance(allowed, (list, tuple, set)):
                        allowed_set = {str(a).lower() for a in allowed}
                        if val.lower() not in allowed_set:
                            skip = True
                            break
                    else:
                        if val.lower() != str(allowed).lower():
                            skip = True
                            break
                if skip:
                    continue

                raw_path = (row.get(path_col) or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = root / path
                if not path.exists():
                    continue

                name_lower = path.name.lower()
                suffix_match = None
                if name_lower.endswith(".nii.gz"):
                    suffix_match = ".nii.gz"
                else:
                    suffix_match = path.suffix.lower()
                if allowed_suffixes:
                    if suffix_match not in allowed_suffixes:
                        continue
                if exclude_tokens and any(tok in name_lower for tok in exclude_tokens):
                    continue

                row_view = (row.get(view_col) or "").strip().upper() if view_col else ""
                if infer_view_from_parent:
                    row_view = _infer_view(path, row_view)
                if target_view_norm and row_view and row_view != target_view_norm:
                    continue
                if target_view_norm and not row_view:
                    continue

                row_modality = (row.get(modality_col) or "").strip().upper() if modality_col else ""
                if target_modality_norm and row_modality and row_modality != target_modality_norm:
                    continue
                if target_modality_norm and not row_modality:
                    continue

                key_frame = default_kf
                if return_key_frame:
                    for col in key_frame_cols:
                        if not col:
                            continue
                        val = row.get(col)
                        if val is None or str(val).strip() == "":
                            continue
                        try:
                            key_frame = int(float(val))
                            break
                        except Exception:
                            continue
                else:
                    key_frame = None

                metadata = {}
                if split_col:
                    metadata["split"] = (row.get(split_col) or "").strip()
                if row_view:
                    metadata["view"] = row_view
                if row_modality:
                    metadata["modality"] = row_modality
                for col in metadata_cols:
                    if col in row and row[col] not in (None, ""):
                        metadata[col] = row[col]

                abs_path = path.resolve()
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                items.append(DatasetItem(path=abs_path, key_frame=key_frame, metadata=metadata))
        return items

    def _load_echo_lvh(
        self,
        spec: DatasetSpec,
        root: Path,
        filters: Dict[str, Any],
        split_csv: Optional[str | Path],
    ) -> List[DatasetItem]:
        opts = spec.options
        csv_name = opts.get("csv_name", "MeasurementsList.csv")
        videos_subdir = opts.get("videos_subdir", "videos")
        filename_col = opts.get("filename_column", "HashedFileName")
        key_frame_col = opts.get("key_frame_column", "Frame")
        split_col = opts.get("split_column", "split")
        split_value = (filters.get("split") or opts.get("default_split", "test"))
        split_value_norm = split_value.lower() if isinstance(split_value, str) else None
        extensions = tuple(opts.get("extensions", [".avi", ".mp4"]))

        csv_path = self._resolve_csv_path(root, split_csv, csv_name)
        videos_dir = root / videos_subdir

        items: List[DatasetItem] = []
        seen: set[Path] = set()
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_split = (row.get(split_col) or "").strip().lower()
                if split_value_norm and row_split != split_value_norm:
                    continue
                fname = (row.get(filename_col) or "").strip()
                if not fname:
                    continue
                path = videos_dir / fname
                if path.suffix.lower() not in extensions:
                    for ext in extensions:
                        candidate = videos_dir / f"{fname}{ext}"
                        if candidate.exists():
                            path = candidate
                            break
                if not path.exists():
                    continue
                try:
                    key_frame = int(float(row.get(key_frame_col, 0)))
                except Exception:
                    key_frame = 0
                abs_path = path.resolve()
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                metadata = {
                    "split": (row.get(split_col) or "").strip(),
                    "view": None,
                    "modality": None,
                }
                items.append(DatasetItem(path=abs_path, key_frame=key_frame, metadata=metadata))
        return items

    def _load_tmed2(
        self,
        spec: DatasetSpec,
        root: Path,
        filters: Dict[str, Any],
        split_csv: Optional[str | Path],
    ) -> List[DatasetItem]:
        if split_csv is None:
            raise ValueError("tmed2 loader requires --split_csv pointing to the TMED2 CSV file")
        opts = spec.options
        csv_path = self._resolve_csv_path(root, split_csv, Path(split_csv).name)
        split_col = opts.get("split_column", "split")
        path_col = opts.get("path_column", "file_path")
        split_value = (filters.get("split") or opts.get("default_split", "test"))
        split_value_norm = split_value.lower() if isinstance(split_value, str) else None
        allowed_suffixes = tuple(opts.get("allowed_suffixes", [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]))

        items: List[DatasetItem] = []
        seen: set[Path] = set()
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_split = (row.get(split_col) or "").strip().lower()
                if split_value_norm and row_split != split_value_norm:
                    continue
                raw_path = (row.get(path_col) or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = root / path
                if not path.exists():
                    continue
                suffix = path.suffix.lower()
                if suffix not in allowed_suffixes:
                    continue
                abs_path = path.resolve()
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                metadata = {
                    "split": (row.get(split_col) or "").strip(),
                    "view": (row.get(opts.get("view_column", "view")) or "").strip(),
                    "modality": None,
                }
                items.append(DatasetItem(path=abs_path, key_frame=0, metadata={k: v for k, v in metadata.items() if v}))
        return items


# ----------------------------------------------------------------------
# Registry setup


def _spec(name: str, kind: str, *, options: Optional[Dict[str, Any]] = None, fixed_filters: Optional[Dict[str, Any]] = None) -> DatasetSpec:
    return DatasetSpec(name=name, kind=kind, options=options or {}, fixed_filters=fixed_filters or {})


ECHONET_DYNAMIC_SPEC = {
    "csv_name": "FileList.csv",
    "videos_subdir": "Videos",
    "split_column": "Split",
    "filename_column": "FileName",
    "default_split": "TEST",
    "extensions": [".avi", ".mp4"],
}
ECHONET_PED_SPEC = {
    "csv_name": "FileList.csv",
    "videos_subdir": "Videos",
    "fold_column": "Split",
    "filename_column": "FileName",
    "default_fold": 0,
    "extensions": [".avi", ".mp4"],
    "available_views": ["A4C", "PSAX"],
}
HMC_QU_SPEC = {
    "csv_name": "split.csv",
    "path_column": "path",
    "split_column": "split",
    "view_column": "view",
    "metadata_columns": ["patient_id", "STEMI", "unique_id"],
    "infer_view_from_parent": True,
    "known_views": ["A2C", "A3C", "A4C", "PSAX", "PLAX"],
    "allowed_suffixes": [".avi", ".mp4"],
    "exclude_tokens": [],
    "default_split": "test",
    "return_key_frame": False,
}
CAMUS_SPEC = {
    "csv_name": "camus_split.csv",
    "path_column": "path",
    "split_column": "split",
    "view_column": "view",
    "metadata_columns": ["unique_id"],
    "allowed_suffixes": [".nii", ".nii.gz", ".avi", ".mp4"],
    "exclude_tokens": ["_mask", "_seg", "_label", "segmentation"],
    "default_split": "test",
    "return_key_frame": True,
}
REGIONAL_WALL_SPEC = {
    "csv_name": "split.csv",
    "path_column": "path",
    "split_column": "split",
    "view_column": "view",
    "modality_column": "modality",
    "key_frame_columns": ["first_annotated_frame_index", "Frame", "key_frame"],
    "required_values": {"modality": ["2D"]},
    "view_regex": r"(A2C|A3C|A4C)",
    "allowed_suffixes": [".nii", ".nii.gz", ".avi", ".mp4"],
    "default_split": "test",
    "return_key_frame": True,
    "metadata_columns": ["unique_id"],
}
ECHONET_LVH_SPEC = {
    "csv_name": "MeasurementsList.csv",
    "videos_subdir": "videos",
    "filename_column": "HashedFileName",
    "key_frame_column": "Frame",
    "split_column": "split",
    "default_split": "test",
    "extensions": [".avi", ".mp4"],
}
SIMPLE_CSV_SPEC = {
    "csv_name": "split.csv",
    "path_column": "path",
    "split_column": "split",
    "key_frame_columns": ["first_annotated_frame_index", "Frame", "key_frame"],
    "allowed_suffixes": [".nii", ".nii.gz", ".avi", ".mp4"],
    "return_key_frame": True,
    "metadata_columns": ["unique_id"],
}
TMED2_SPEC = {
    "path_column": "file_path",
    "split_column": "split",
    "view_column": "view",
    "allowed_suffixes": [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"],
    "default_split": "test",
    "metadata_columns": ["diagnosis_label", "unique_id"],
}

DEFAULT_DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "Dynamic": _spec("echonet", "Dynamic", options=ECHONET_DYNAMIC_SPEC),
    "pediatric": _spec("echonet_ped", "Pediatric", options=ECHONET_PED_SPEC.copy()),
    "echonet_ped_a4c": _spec("echonet_ped_a4c", "Pediatric", options=ECHONET_PED_SPEC.copy(), fixed_filters={"view": "A4C"}),
    "echonet_ped_psax": _spec("echonet_ped_psax", "Pediatric", options=ECHONET_PED_SPEC.copy(), fixed_filters={"view": "PSAX"}),
    "hmc_qu": _spec("hmc_qu", "csv_table", options=HMC_QU_SPEC.copy()),
    "hmc_qu_a2c": _spec("hmc_qu_a2c", "csv_table", options=HMC_QU_SPEC.copy(), fixed_filters={"view": "A2C"}),
    "hmc_qu_a4c": _spec("hmc_qu_a4c", "csv_table", options=HMC_QU_SPEC.copy(), fixed_filters={"view": "A4C"}),
    "camus": _spec("camus", "csv_table", options=CAMUS_SPEC.copy()),
    "camus_2ch": _spec("camus_2ch", "csv_table", options=CAMUS_SPEC.copy(), fixed_filters={"view": "2CH"}),
    "camus_4ch": _spec("camus_4ch", "csv_table", options=CAMUS_SPEC.copy(), fixed_filters={"view": "4CH"}),
    "regional_wall": _spec("regional_wall", "csv_table", options=REGIONAL_WALL_SPEC.copy()),
    "regional_wall_a2c": _spec("regional_wall_a2c", "csv_table", options=REGIONAL_WALL_SPEC.copy(), fixed_filters={"view": "A2C"}),
    "regional_wall_a3c": _spec("regional_wall_a3c", "csv_table", options=REGIONAL_WALL_SPEC.copy(), fixed_filters={"view": "A3C"}),
    "regional_wall_a4c": _spec("regional_wall_a4c", "csv_table", options=REGIONAL_WALL_SPEC.copy(), fixed_filters={"view": "A4C"}),
    "echo_lvh": _spec("echo_lvh", "echo_lvh_measurements", options=ECHONET_LVH_SPEC.copy()),
    "asd_csv": _spec("asd_csv", "csv_table", options=SIMPLE_CSV_SPEC.copy()),
    "pah_csv": _spec("pah_csv", "csv_table", options=SIMPLE_CSV_SPEC.copy()),
    "tmed2_csv": _spec("tmed2_csv", "tmed2_csv", options=TMED2_SPEC.copy()),
}


def _cli():
    loader = DatasetLoader()
    parser = argparse.ArgumentParser(description="Inspect dataset items using the modular loader")
    parser.add_argument("--config", default=None, help="Path to JSON config providing default arguments")
    parser.add_argument("--dataset", choices=loader.available(), default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--modality", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Print at most this many items")
    args = parser.parse_args()

    config_data: Dict[str, Any] = {}
    config_path: Optional[Path] = None
    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        default_candidate = Path(__file__).resolve().parent / "config" / "dataset_loader.json"
        if default_candidate.exists():
            config_path = default_candidate

    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                config_data = data
            else:
                raise ValueError(f"Config file {config_path} must contain a JSON object")

    def _resolve(key: str) -> Any:
        value = getattr(args, key, None)
        if value is not None:
            return value
        return config_data.get(key)

    dataset = _resolve("dataset")
    if not dataset:
        raise ValueError("Provide a dataset name either via CLI or config file")
    if dataset not in loader.available():
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {loader.available()}")

    root = _resolve("root")
    if not root:
        raise ValueError("Provide a dataset root either via CLI or config file")

    split = _resolve("split")
    view = _resolve("view")
    modality = _resolve("modality")
    fold = _resolve("fold")
    split_csv = _resolve("split_csv")

    limit = _resolve("limit")
    if limit is None:
        limit = 10
    else:
        limit = int(limit)

    items = loader.load(
        str(dataset),
        root=root,
        split=split,
        view=view,
        modality=modality,
        fold=None if fold is None else int(fold),
        split_csv=split_csv,
    )
    limit = max(0, limit)
    for item in items[:limit or None]:
        meta = ", ".join(f"{k}={v}" for k, v in item.metadata.items()) if item.metadata else ""
        print(f"{item.path} | key_frame={item.key_frame}{(' | ' + meta) if meta else ''}")
    print(f"Total items: {len(items)}")


if __name__ == "__main__":
    _cli()


__all__ = ["DatasetLoader", "DatasetItem", "DatasetSpec", "DEFAULT_DATASET_REGISTRY"]
