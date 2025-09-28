from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


def crop_and_scale(img: np.ndarray, res: Tuple[int, int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.1) -> np.ndarray:
    """
    Preprocess an image by center-cropping and resizing to the target resolution.
    """
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding] if padding > 0 else img
    elif r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding] if padding > 0 else img
    if zoom > 0:
        pad_x = int(round(img.shape[1] * zoom))
        pad_y = int(round(img.shape[0] * zoom))
        if pad_x > 0 and pad_y > 0 and pad_y * 2 < img.shape[0] and pad_x * 2 < img.shape[1]:
            img = img[pad_y:-pad_y, pad_x:-pad_x]

    return cv2.resize(img, res, interpolation=interpolation)


def read_avi(path: Path, res: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Read echocardiography videos in AVI format and retunr as a numpy array of shape (T, H, W, C).
    """
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice to the range [0, 255] as uint8.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        mid = arr.shape[-1] // 2
        arr = arr[..., mid]
    if arr.ndim == 1:
        arr = arr[:, None]
    smin = float(arr.min()) if arr.size else 0.0
    smax = float(arr.max()) if arr.size else 0.0
    if smax > smin:
        arr = (arr - smin) / (smax - smin)
    else:
        arr = arr * 0.0
    arr = (arr * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def read_nii(path: Path, res: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Read echocardiography videos in NIfTI format and return as a numpy array of shape (T, H, W, C).
    """
    img = nib.load(str(path))
    dataobj = img.dataobj
    shp = img.shape

    frames: List[np.ndarray] = []
    if len(shp) == 2:
        frame = _normalize_slice(np.asarray(dataobj))
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    else:
        T = int(shp[-1])
        for idx in range(T):
            frame = _normalize_slice(np.asarray(dataobj[..., idx]))
            if res is not None:
                frame = crop_and_scale(frame, res)
            frames.append(frame)
    return np.stack(frames, axis=0)


def read_video(path: Path, res: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Main video reading function that supports NIfTI, AVI, and image files.
    """
    name = path.name.lower()
    suf = path.suffix.lower()
    if name.endswith(".nii.gz") or suf == ".nii":
        return read_nii(path, res=res)
    if suf in {".avi", ".mp4"}:
        return read_avi(path, res=res)
    if suf in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if res is not None:
            img = crop_and_scale(img, res)
        if img.ndim == 2:
            frames = np.expand_dims(img, axis=0)
        else:
            frames = np.expand_dims(img, axis=0)
        return frames
    raise ValueError(f"Unsupported video type for {path}")


def preprocess_frames(frames: np.ndarray, preprocess_val) -> torch.Tensor:
    to_pil = T.ToPILImage()
    tensors = [preprocess_val(to_pil(frame)) for frame in frames]
    return torch.stack(tensors, dim=0)


def encode_video_clip_batched(
    model,
    frames_tensor: torch.Tensor,
    *,
    device: str = "cuda",
    precision: str = "bf16",
    batch_size: int = 128,
    use_channels_last: bool = True,
    pin_memory: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    if pin_memory and frames_tensor.device.type == "cpu":
        frames_tensor = frames_tensor.pin_memory()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    outputs: List[torch.Tensor] = []
    stream = torch.cuda.Stream(device=device) if "cuda" in device else None

    for start in range(0, frames_tensor.shape[0], batch_size):
        chunk = frames_tensor[start:start + batch_size]
        if stream is not None:
            with torch.cuda.stream(stream):
                x = chunk.to(device, dtype=dtype, non_blocking=True)
                if use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                x = model.encode_image(x)
                if normalize:
                    x = F.normalize(x, dim=-1)
        else:
            x = chunk.to(device, dtype=dtype)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            x = model.encode_image(x)
            if normalize:
                x = F.normalize(x, dim=-1)
        outputs.append(x)

    if stream is not None:
        torch.cuda.current_stream(device).wait_stream(stream)

    feats = torch.cat(outputs, dim=0).to("cpu").to(torch.float16)
    return feats


def indices_after_keyframe(n_frames: int, start: int, max_frames: int, stride: int) -> List[int]:
    if n_frames <= 0:
        return []
    start = max(0, min(start, n_frames - 1))
    idxs = list(range(start, n_frames, max(1, stride)))
    return idxs[:max_frames]


def select_indices(n_frames: int, max_frames: int, stride: int) -> List[int]:
    end = min(max_frames, n_frames)
    return list(range(0, end, max(1, stride)))


def save_sample_frames(
    frames: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    indices: Optional[Sequence[int]] = None,
    max_frames: int = 3,
) -> List[Path]:
    """
    Save sample frames from a video to the specified output directory for visual inspection
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if indices is None:
        take = min(max_frames, int(frames.shape[0]))
        indices = list(range(take))

    saved: List[Path] = []
    for idx in indices:
        if idx < 0 or idx >= frames.shape[0]:
            continue
        frame = np.squeeze(frames[idx])
        if frame.dtype != np.uint8:
            arr = frame.astype(np.float32)
            arr -= arr.min() if arr.size else 0.0
            max_val = arr.max() if arr.size else 0.0
            if max_val > 0:
                arr /= max_val
            frame_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_u8 = frame
        if frame_u8.ndim == 3 and frame_u8.shape[-1] == 1:
            frame_u8 = frame_u8[..., 0]
        out_path = out_dir / f"{stem}_frame{idx:04d}.png"
        cv2.imwrite(str(out_path), frame_u8)
        saved.append(out_path)
    return saved


__all__ = [
    "crop_and_scale",
    "read_video",
    "preprocess_frames",
    "encode_video_clip_batched",
    "indices_after_keyframe",
    "select_indices",
    "save_sample_frames",
]
