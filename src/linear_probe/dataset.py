from pathlib import Path
from typing import Optional, Tuple, List, Dict
import csv
import torch
from torch.utils.data import Dataset
import cv2

from .utils import (
    read_video,
    preprocess_frames,
    preprocess_one_frame,
    crop_and_scale,
    get_num_frames,
    read_one_avi_frame,
    read_one_nii_frame,
)


def _indices_after_keyframe(n: int, start: int, max_frames: int, stride: int) -> List[int]:
    if n <= 0:
        return []
    start = max(0, min(start, n - 1))
    idxs = list(range(start, n, max(1, stride)))
    return idxs[:max_frames]


class EchoVideoCSV(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        split: str = "test",
        root: str | Path | None = None,
        view: Optional[str] = None,
        modality: Optional[str] = None,
        res: Tuple[int, int] = (224, 224),
        max_frames: int = 1,
        stride: int = 1,
        default_key_frame: int = 0,
        preprocess_val=None,
        key_frame_col: str = "key_frame",
        label_col: Optional[str] = None,
        random_single_frame: bool = False,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self.root = None if root is None else Path(root)
        self.want_split = split.lower()
        self.want_view = None if view is None else view.upper()
        self.want_modality = None if modality is None else str(modality).upper()
        self.res = res
        self.max_frames = max_frames
        self.stride = stride
        self.default_kf = default_key_frame
        self.preprocess_val = preprocess_val
        self.key_frame_col = key_frame_col
        self.label_col = label_col
        self.random_single_frame = bool(random_single_frame)

        self.rows: List[Dict[str, str]] = []
        with open(self.csv_path, "r", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("split") or "").strip().lower() != self.want_split:
                    continue
                path_value = (row.get("path") or "").strip()
                if not path_value:
                    continue
                path = Path(path_value)
                if self.root is not None and not path.is_absolute():
                    path = self.root / path
                if not path.exists():
                    continue
                if self.want_view:
                    rv = (row.get("view") or "").strip().upper()
                    if rv and rv != self.want_view:
                        continue
                if self.want_modality:
                    rm = (row.get("modality") or "").strip().upper()
                    if rm and rm != self.want_modality:
                        continue
                row["_abs_path"] = str(path.resolve())
                self.rows.append(row)

        if not self.rows:
            raise FileNotFoundError(
                f"No rows for split={self.want_split} (view={self.want_view}) in {self.csv_path}"
            )
        if self.preprocess_val is None:
            raise RuntimeError("Pass preprocess_val from create_model_and_transforms(...)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["_abs_path"])
        try:
            key_frame = int(row.get(self.key_frame_col, self.default_kf))
        except Exception:
            key_frame = self.default_kf

        suf = path.suffix.lower()
        is_image = suf in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

        if is_image:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            tensor = preprocess_one_frame(img, self.preprocess_val)
            video = tensor.unsqueeze(0).contiguous().clone()
            selected = [0]
            n_raw = 1
            key_frame = 0
        else:
            if suf in {".avi", ".mp4"} and int(self.max_frames) == 1:
                n_raw = get_num_frames(path)
                if self.random_single_frame and self.want_split == "train" and n_raw > 0:
                    import random

                    idx = random.randint(0, max(n_raw - 1, 0))
                else:
                    idx = max(0, min(int(key_frame), max(n_raw - 1, 0)))
                frame = read_one_avi_frame(path, idx)
                tensor = preprocess_one_frame(frame, self.preprocess_val)
                video = tensor.unsqueeze(0).contiguous().clone()
                selected = [idx]
            elif (path.name.lower().endswith(".nii.gz") or suf == ".nii") and int(self.max_frames) == 1:
                n_raw = get_num_frames(path)
                if self.random_single_frame and self.want_split == "train" and n_raw > 0:
                    import random

                    idx = random.randint(0, max(n_raw - 1, 0))
                else:
                    idx = max(0, min(int(key_frame), max(n_raw - 1, 0)))
                frame = read_one_nii_frame(path, idx)
                tensor = preprocess_one_frame(frame, self.preprocess_val)
                video = tensor.unsqueeze(0).contiguous().clone()
                selected = [idx]
            else:
                frames = read_video(path, res=None)
                n_raw = int(frames.shape[0])
                if int(self.max_frames) == 1:
                    if self.random_single_frame and self.want_split == "train" and n_raw > 0:
                        import random

                        idx = random.randint(0, n_raw - 1)
                    else:
                        idx = max(0, min(int(key_frame), n_raw - 1))
                    tensor = preprocess_one_frame(frames[idx], self.preprocess_val)
                    video = tensor.unsqueeze(0).contiguous().clone()
                    selected = [idx]
                else:
                    selected = _indices_after_keyframe(n_raw, key_frame, self.max_frames, self.stride)
                    if not selected:
                        raise RuntimeError(
                            f"No frames selected for {path.name} (n_raw={n_raw}, key_frame={key_frame})"
                        )
                    video = preprocess_frames(frames[selected], self.preprocess_val)

        label = None
        if self.label_col:
            val = row.get(self.label_col)
            if val is not None:
                text = str(val).strip()
                try:
                    label = int(text)
                except Exception:
                    try:
                        label = float(text)
                    except Exception:
                        label = text

        meta = {
            "path": str(path),
            "video_id": path.stem,
            "frame_indices": selected,
            "key_frame": key_frame,
            "view": (row.get("view") or "").strip().upper(),
            "n_frames_raw": n_raw,
        }
        if label is not None:
            meta["label"] = label
        return video, meta
