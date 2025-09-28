import cv2
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import torch
from functools import lru_cache


def preprocess_one_frame(frame_hwC: np.ndarray, preprocess_val) -> torch.Tensor:
    if frame_hwC.ndim == 2:
        frame_hwC = frame_hwC[..., None]
    C = frame_hwC.shape[-1]
    if C == 1:
        frame_hwC = np.repeat(frame_hwC, 3, axis=-1)
    elif C == 3:
        frame_hwC = frame_hwC[..., ::-1].copy()
    frame_hwC = np.ascontiguousarray(frame_hwC, dtype=np.uint8)
    ten = preprocess_val(Image.fromarray(frame_hwC))
    return ten.contiguous().to(torch.float32).clone()


def preprocess_frames(frames: np.ndarray, preprocess_val) -> torch.Tensor:
    if frames.ndim == 3:
        frames = frames[..., None]
    assert frames.ndim == 4
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    T, H, W, C = frames.shape
    if C == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif C == 3:
        frames = frames[..., ::-1].copy()
    out = []
    for t in range(T):
        img = Image.fromarray(frames[t])
        ten = preprocess_val(img).contiguous().clone()
        out.append(ten)
    return torch.stack(out, dim=0)


def crop_and_scale(img, res=(640, 480), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]
    img = cv2.resize(img, res, interpolation=interpolation)
    return img


def read_avi(p: Path, res=None):
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def get_num_frames(p: Path) -> int:
    name = p.name.lower()
    suf = p.suffix.lower()
    if name.endswith(".nii.gz") or suf == ".nii":
        img = _load_nifti_cached(str(p))
        shp = img.shape
        if len(shp) == 3:
            return int(shp[-1])
        if len(shp) == 4:
            return int(shp[-1])
        return 1
    if suf in {".avi", ".mp4"}:
        cap = cv2.VideoCapture(str(p))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return max(n, 0)
    return 1


def read_one_avi_frame(p: Path, idx: int, res=None) -> np.ndarray:
    cap = cv2.VideoCapture(str(p))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {p}")
        if res is not None:
            frame = crop_and_scale(frame, res)
        return frame
    finally:
        cap.release()


def read_nii(p: Path, res=None) -> np.ndarray:
    img = _load_nifti_cached(str(p))
    dataobj = img.dataobj
    shp = img.shape
    outs = []
    if len(shp) == 2:
        slice_ = np.asarray(dataobj, dtype=np.float32)
        slice_ = np.squeeze(slice_)
        if slice_.ndim != 2:
            if slice_.ndim > 2:
                mid = slice_.shape[-1] // 2
                slice_ = slice_[..., mid]
            elif slice_.ndim == 1:
                slice_ = slice_[:, None]
        smin = float(slice_.min()) if slice_.size else 0.0
        smax = float(slice_.max()) if slice_.size else 0.0
        slice_ = (slice_ - smin) / (smax - smin) if smax > smin else slice_ * 0.0
        slice_u8 = (slice_ * 255.0).astype(np.uint8)
        if res is not None:
            slice_u8 = crop_and_scale(slice_u8, res=res)
        if slice_u8.ndim == 2:
            slice_u8 = slice_u8[..., None]
        outs.append(slice_u8)
    else:
        T = int(shp[-1])
        for idx in range(T):
            slice_ = np.asarray(dataobj[..., idx], dtype=np.float32)
            slice_ = np.squeeze(slice_)
            if slice_.ndim > 2:
                mid = slice_.shape[-1] // 2
                slice_ = slice_[..., mid]
            if slice_.ndim == 1:
                slice_ = slice_[:, None]
            smin = float(slice_.min()) if slice_.size else 0.0
            smax = float(slice_.max()) if slice_.size else 0.0
            slice_ = (slice_ - smin) / (smax - smin) if smax > smin else slice_ * 0.0
            slice_u8 = (slice_ * 255.0).astype(np.uint8)
            if res is not None:
                slice_u8 = crop_and_scale(slice_u8, res=res)
            if slice_u8.ndim == 2:
                slice_u8 = slice_u8[..., None]
            outs.append(slice_u8)
    return np.stack(outs, axis=0)


def read_one_nii_frame(p: Path, idx: int, res=None) -> np.ndarray:
    img = _load_nifti_cached(str(p))
    dataobj = img.dataobj
    shp = img.shape
    if len(shp) >= 3:
        T = shp[-1]
    else:
        T = 1
    idx = max(0, min(int(idx), int(T) - 1))
    slice_ = np.asarray(dataobj[..., idx], dtype=np.float32)
    if slice_.ndim > 2:
        slice_ = np.squeeze(slice_)
        if slice_.ndim > 2:
            mid = slice_.shape[-1] // 2
            slice_ = slice_[..., mid]
    if slice_.ndim == 1:
        slice_ = slice_[None, :]
    smin = float(slice_.min())
    smax = float(slice_.max())
    if smax > smin:
        slice_ = (slice_ - smin) / (smax - smin)
    else:
        slice_ = slice_ * 0.0
    slice_u8 = (slice_ * 255.0).astype(np.uint8)
    if res is not None:
        slice_u8 = crop_and_scale(slice_u8, res=res)
    if slice_u8.ndim == 2:
        slice_u8 = slice_u8[..., None]
    return slice_u8


@lru_cache(maxsize=32)
def _load_nifti_cached(path: str):
    return nib.load(path)


__all__ = [
    "preprocess_one_frame",
    "preprocess_frames",
    "crop_and_scale",
    "read_avi",
    "get_num_frames",
    "read_one_avi_frame",
    "read_nii",
    "read_one_nii_frame",
]
