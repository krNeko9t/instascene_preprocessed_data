from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import PipelineConfig
from data_loader import SceneData, ViewRecord


@dataclass(slots=True)
class ViewInfo:
    view_name: str
    image_path: str
    mask_path: str
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    pixel_count: int
    bbox_area_ratio: float
    pixel_ratio: float
    quality_score: float


def _compute_bbox(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(binary_mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("binary_mask has no positive pixels")
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def _build_view_info(record: ViewRecord, obj_id: int) -> ViewInfo | None:
    binary_mask = record.mask == obj_id
    pixel_count = int(binary_mask.sum())
    if pixel_count <= 0:
        return None

    x1, y1, x2, y2 = _compute_bbox(binary_mask)
    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1
    bbox_area = bbox_w * bbox_h
    image_area = record.mask.shape[0] * record.mask.shape[1]
    bbox_area_ratio = float(bbox_area / max(image_area, 1))
    pixel_ratio = float(pixel_count / max(bbox_area, 1))

    # Favor large, compact observations.
    quality_score = (pixel_count / max(image_area, 1)) * 0.7 + pixel_ratio * 0.3
    return ViewInfo(
        view_name=record.view_name,
        image_path=str(record.image_path),
        mask_path=str(record.mask_path),
        mask=binary_mask,
        bbox_xyxy=(x1, y1, x2, y2),
        pixel_count=pixel_count,
        bbox_area_ratio=bbox_area_ratio,
        pixel_ratio=pixel_ratio,
        quality_score=quality_score,
    )


def select_views_for_object(obj_id: int, scene_data: SceneData, config: PipelineConfig) -> List[ViewInfo]:
    candidates: List[ViewInfo] = []
    for record in scene_data.object_to_views.get(obj_id, []):
        info = _build_view_info(record, obj_id)
        if info is None:
            continue
        if info.pixel_count < config.object_filter.min_pixel_count:
            continue
        if info.bbox_area_ratio < config.object_filter.min_bbox_area_ratio:
            continue
        if info.pixel_ratio < config.object_filter.min_pixel_ratio:
            continue
        candidates.append(info)

    candidates.sort(
        key=lambda x: (x.quality_score, x.pixel_count, x.pixel_ratio, x.bbox_area_ratio),
        reverse=True,
    )

    if config.view_render.max_views <= 0:
        return candidates
    return candidates[: config.view_render.max_views]
