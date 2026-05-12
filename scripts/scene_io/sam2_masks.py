"""Decode SAM2 ``auto_masks.json`` masklet RLEs into 2D instance id maps (pycocotools)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_util

__all__ = [
    "decode_masklet_frame_to_id_map",
    "load_auto_masks_document",
    "pair_sorted_rgb_with_masklet",
]


def decode_masklet_frame_to_id_map(rles: Sequence[dict[str, Any]]) -> np.ndarray:
    """Merge one frame's binary RLE masks into a single id map; id ``k+1`` for RLE index ``k``, 0 background."""
    if not rles:
        raise ValueError("empty RLE list for masklet frame")
    first = rles[0]
    size = first.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"invalid RLE size: {first!r}")
    h, w = int(size[0]), int(size[1])
    out = np.zeros((h, w), dtype=np.int32)
    for k, rle in enumerate(rles):
        binary = mask_util.decode(rle)
        if binary.ndim == 3:
            binary = np.squeeze(binary, axis=-1)
        if binary.shape[:2] != (h, w):
            raise ValueError(
                f"decoded mask shape {binary.shape[:2]} != RLE size ({h}, {w}) for instance index {k}"
            )
        out[binary > 0] = k + 1
    return out


def load_auto_masks_document(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"auto_masks JSON must be an object: {path}")
    if "masklet" not in data:
        raise ValueError(f"auto_masks JSON missing masklet: {path}")
    return data


def pair_sorted_rgb_with_masklet(
    image_dir: Path,
    masklet: List[List[dict[str, Any]]],
    *,
    supported_suffixes: set[str],
) -> List[Tuple[str, Path, int, np.ndarray]]:
    """Sort rgb files under ``image_dir``, align with ``masklet`` by index, return per-frame id maps.

    Returns tuples ``(view_stem, image_path, frame_index, id_map)``.
    """
    image_paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix in supported_suffixes
    )
    n_img = len(image_paths)
    n_mask = len(masklet)
    n_pair = min(n_img, n_mask)
    if n_img != n_mask:
        warnings.warn(
            f"{image_dir}: image count {n_img} != masklet frames {n_mask}; using first {n_pair} pairs",
            stacklevel=2,
        )
    out: List[Tuple[str, Path, int, np.ndarray]] = []
    for i in range(n_pair):
        frame_rles = masklet[i]
        if not isinstance(frame_rles, list):
            raise TypeError(f"masklet[{i}] must be a list, got {type(frame_rles).__name__}")
        id_map = decode_masklet_frame_to_id_map(frame_rles)
        ip = image_paths[i]
        out.append((ip.stem, ip, i, id_map))
    return out
