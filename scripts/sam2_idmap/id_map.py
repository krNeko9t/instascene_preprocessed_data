from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np


def masks_to_id_map(masks: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Merge SAM2 mask records into a single HxW instance id map.

    Background pixels are 0. Instance ids are 1..N. Masks are painted from
    largest area to smallest so smaller instances win on overlaps.
    """
    id_map = np.zeros((height, width), dtype=np.int32)
    if not masks:
        return id_map

    sorted_masks = sorted(masks, key=lambda ann: ann["area"], reverse=True)
    for inst_id, ann in enumerate(sorted_masks, start=1):
        id_map[ann["segmentation"]] = inst_id
    return id_map


def atomic_npy_save(path: str, arr: np.ndarray) -> None:
    """Write a .npy file atomically via a temporary file."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_dir = out_dir or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npy", dir=tmp_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, arr)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
