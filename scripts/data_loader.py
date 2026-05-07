from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from sam2_auto_masks import load_auto_masks_document, pair_sorted_rgb_with_masklet
from view_pairing import SUPPORTED_IMAGE_SUFFIXES, list_mask_paths, pair_image_and_masks

__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "SceneData",
    "ViewRecord",
    "build_views_from_sam2_json",
    "load_scene",
    "load_scene_from_npy",
    "load_scene_from_png",
    "_load_mask",
    "_load_npy_id_map",
]


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


def _load_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
    return arr


def _load_npy_id_map(id_map_path: Path) -> np.ndarray:
    arr = np.load(id_map_path)
    if arr.ndim != 2:
        raise ValueError(f"ID map should be 2D, got shape {arr.shape} from {id_map_path}")
    return arr


def _resize_mask_to_image(mask: np.ndarray, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    mask_h, mask_w = mask.shape[:2]
    if (mask_w, mask_h) == (img_w, img_h):
        return mask
    resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(mask.dtype, copy=False)


def load_scene_from_png(scene_root: Path, mask_subdir: str) -> tuple[Path, List[Path]]:
    mask_dir = scene_root / "sam" / mask_subdir
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    mask_files = list_mask_paths(mask_dir, "png")
    if not mask_files:
        raise RuntimeError(f"No mask files found in: {mask_dir}")
    return mask_dir, mask_files


def load_scene_from_npy(scene_root: Path) -> tuple[Path, List[Path]]:
    id_map_dir = scene_root / "id_maps"
    if not id_map_dir.exists():
        raise FileNotFoundError(f"ID map directory does not exist: {id_map_dir}")
    id_map_files = list_mask_paths(id_map_dir, "npy")
    if not id_map_files:
        raise RuntimeError(f"No npy id maps found in: {id_map_dir}")
    return id_map_dir, id_map_files


def default_sam2_auto_masks_path(scene_root: Path, scene_name: str) -> Path:
    """``processed_re10k/<scene>/`` -> ``processed_re10k/sam2_results/<scene>/auto_masks.json``."""
    return scene_root.parent / "sam2_results" / scene_name / "auto_masks.json"


def build_views_from_sam2_json(
    image_dir: Path,
    auto_masks_json: Path,
) -> Tuple[Path, List[ViewRecord]]:
    """Decode ``auto_masks.json`` and pair sorted RGB images with ``masklet`` frames."""
    if not auto_masks_json.is_file():
        raise FileNotFoundError(f"auto_masks JSON not found: {auto_masks_json}")
    doc = load_auto_masks_document(auto_masks_json)
    masklet = doc["masklet"]
    if not isinstance(masklet, list):
        raise TypeError(f"masklet must be a list in {auto_masks_json}")
    paired = pair_sorted_rgb_with_masklet(
        image_dir,
        masklet,
        supported_suffixes=SUPPORTED_IMAGE_SUFFIXES,
    )
    mask_dir = auto_masks_json.parent
    views: List[ViewRecord] = []
    for view_name, image_path, frame_index, id_map in paired:
        id_map = _resize_mask_to_image(id_map, image_path)
        views.append(
            ViewRecord(
                view_name=view_name,
                image_path=image_path,
                mask_path=auto_masks_json,
                mask=id_map,
                frame_index=frame_index,
            )
        )
    return mask_dir, views


def load_scene(
    data_root: Path,
    dataset: str,
    scene: str,
    mask_subdir: str = "mask",
    id_map_source: str = "npy",
    image_subdir: str = "images",
    sam2_json_path: Path | None = None,
) -> SceneData:
    scene_root = data_root / dataset / scene
    image_dir = scene_root / image_subdir

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_root}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if id_map_source == "sam2_json":
        json_path = (
            sam2_json_path.expanduser().resolve()
            if sam2_json_path is not None
            else default_sam2_auto_masks_path(scene_root, scene)
        )
        map_dir, views = build_views_from_sam2_json(image_dir, json_path)
    elif id_map_source == "npy":
        map_dir, map_files = load_scene_from_npy(scene_root)
        pairs = pair_image_and_masks(image_dir, map_files, strategy="stem")
        views = []
        for view_name, image_path, map_path in pairs:
            mask = _load_npy_id_map(map_path)
            mask = _resize_mask_to_image(mask, image_path)
            views.append(
                ViewRecord(
                    view_name=view_name,
                    image_path=image_path,
                    mask_path=map_path,
                    mask=mask,
                    frame_index=None,
                )
            )
    elif id_map_source == "png":
        map_dir, map_files = load_scene_from_png(scene_root, mask_subdir)
        pairs = pair_image_and_masks(image_dir, map_files, strategy="stem")
        views = []
        for view_name, image_path, map_path in pairs:
            mask = _load_mask(map_path)
            mask = _resize_mask_to_image(mask, image_path)
            views.append(
                ViewRecord(
                    view_name=view_name,
                    image_path=image_path,
                    mask_path=map_path,
                    mask=mask,
                    frame_index=None,
                )
            )
    else:
        raise ValueError(f"Unsupported id_map_source: {id_map_source}")

    object_id_set: set[int] = set()
    object_to_views: Dict[int, List[ViewRecord]] = {}

    for record in views:
        unique_ids = np.unique(record.mask)
        for obj_id in unique_ids:
            int_id = int(obj_id)
            if int_id < 0:
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
        mask_dir=map_dir,
        views=views,
        object_ids=object_ids,
        object_to_views=object_to_views,
    )
