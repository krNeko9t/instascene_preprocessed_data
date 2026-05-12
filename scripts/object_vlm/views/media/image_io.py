from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np


def read_image_rgb(image_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_rgb_to_jpeg_base64(image_rgb: np.ndarray, quality: int = 90) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
