from __future__ import annotations

from typing import Any

import numpy as np

from . import geometry


def compose_panels(rendered_panels: list[np.ndarray], layout_spec: dict[str, Any], fill_value: int) -> np.ndarray:
    if not rendered_panels:
        raise ValueError("Cannot compose empty panels")

    layout_type = str(layout_spec.get("type", "horizontal")).strip().lower()
    if layout_type not in {"horizontal", "vertical", "grid"}:
        raise ValueError("layout.type must be one of: horizontal, vertical, grid")

    if layout_type == "horizontal":
        target_h = max(img.shape[0] for img in rendered_panels)
        padded = [geometry.pad_to_size_center(img, target_h, img.shape[1], fill_value) for img in rendered_panels]
        return np.hstack(padded)

    if layout_type == "vertical":
        target_w = max(img.shape[1] for img in rendered_panels)
        padded = [geometry.pad_to_size_center(img, img.shape[0], target_w, fill_value) for img in rendered_panels]
        return np.vstack(padded)

    n = len(rendered_panels)
    cols = layout_spec.get("cols")
    rows = layout_spec.get("rows")
    if cols is None and rows is None:
        cols = 2
    if cols is not None:
        cols = int(cols)
        if cols <= 0:
            raise ValueError("layout.cols must be positive")
    if rows is not None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("layout.rows must be positive")
    if cols is None:
        cols = int(np.ceil(n / rows))
    if rows is None:
        rows = int(np.ceil(n / cols))
    if rows * cols < n:
        raise ValueError("layout.rows * layout.cols is smaller than number of panels")

    tile_h = max(img.shape[0] for img in rendered_panels)
    tile_w = max(img.shape[1] for img in rendered_panels)
    blank = np.full((tile_h, tile_w, 3), fill_value, dtype=np.uint8)
    tiles = [geometry.pad_to_size_center(img, tile_h, tile_w, fill_value) for img in rendered_panels]
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    row_images = []
    for r in range(rows):
        start = r * cols
        row_images.append(np.hstack(tiles[start : start + cols]))
    return np.vstack(row_images)
