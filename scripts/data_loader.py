from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass(slots=True)
class ViewRecord:
    view_name: str
    image_path: Path
    mask_path: Path
    mask: np.ndarray


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


def _load_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
    return arr


def _build_image_index(image_dir: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        image_index[path.stem] = path
    return image_index


def load_scene(data_root: Path, dataset: str, scene: str, mask_subdir: str = "mask") -> SceneData:
    scene_root = data_root / dataset / scene
    image_dir = scene_root / "images"
    mask_dir = scene_root / "sam" / mask_subdir

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_root}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    image_index = _build_image_index(image_dir)
    mask_files = sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not mask_files:
        raise RuntimeError(f"No mask files found in: {mask_dir}")

    views: List[ViewRecord] = []
    object_id_set: set[int] = set()
    object_to_views: Dict[int, List[ViewRecord]] = {}

    for mask_path in mask_files:
        view_name = mask_path.stem
        image_path = image_index.get(view_name)
        if image_path is None:
            # Skip masks without corresponding image.
            continue

        mask = _load_mask(mask_path)
        record = ViewRecord(
            view_name=view_name,
            image_path=image_path,
            mask_path=mask_path,
            mask=mask,
        )
        views.append(record)

        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            int_id = int(obj_id)
            if int_id == 0:
                continue
            object_id_set.add(int_id)
            object_to_views.setdefault(int_id, []).append(record)

    if not views:
        raise RuntimeError(f"No valid image-mask pairs found under: {scene_root}")

    object_ids = sorted(object_id_set)
    return SceneData(
        dataset=dataset,
        scene=scene,
        scene_root=scene_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        views=views,
        object_ids=object_ids,
        object_to_views=object_to_views,
    )
