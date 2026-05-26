from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["load_mask", "load_npy_id_map"]


def load_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
    return arr


def load_npy_id_map(id_map_path: Path) -> np.ndarray:
    arr = np.load(id_map_path)
    if arr.ndim != 2:
        raise ValueError(f"ID map should be 2D, got shape {arr.shape} from {id_map_path}")
    return arr
