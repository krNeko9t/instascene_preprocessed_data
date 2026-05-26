from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

__all__ = ["ViewRecord", "SceneData", "ScenePathRecord"]


@dataclass(slots=True, frozen=True)
class ScenePathRecord:
    """One scene worth of absolute paths before serializing to manifest JSON."""

    partition: str
    scene_name: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path | None
    id_map_json: Path | None = None


@dataclass(slots=True)
class ViewRecord:
    view_name: str
    image_path: Path
    mask_path: Path
    mask: np.ndarray
    frame_index: int | None = None


@dataclass(slots=True)
class SceneData:
    dataset: str
    scene: str
    scene_root: Path
    image_dir: Path
    mask_dir: Path
    views: List[ViewRecord]
    object_ids: List[int]
    object_to_views: Dict[int, List[ViewRecord]]
