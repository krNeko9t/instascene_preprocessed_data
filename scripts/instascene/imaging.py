"""Shared image IO and JPEG/base64 encoding used by rendering and API clients."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "encode_image_path_to_jpeg_b64",
    "encode_rgb_to_jpeg_b64",
    "read_image_rgb",
]


def read_image_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_rgb_to_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    q = max(1, min(100, quality))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=q)
        return base64.b64encode(bio.getvalue()).decode("ascii")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_image_path_to_jpeg_b64(path: str | Path, *, quality: int = 90) -> str:
    return encode_rgb_to_jpeg_b64(read_image_rgb(path), quality=quality)
