from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from instascene.manifest.models import ResolvedScenePaths
from instascene.scene.loaders.masks import load_mask, load_npy_id_map
from instascene.scene.loaders.sam2 import load_auto_masks_document, pair_sorted_rgb_with_masklet
from instascene.scene.models import SceneData, ViewRecord
from instascene.scene.pairing import list_mask_paths, pair_image_and_masks
from instascene.types import IdMapSource, PairingStrategy, SUPPORTED_IMAGE_SUFFIXES

__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "SceneData",
    "ViewRecord",
    "build_views_from_sam2_json",
    "load_scene_from_paths",
]


def _resize_mask_to_image(mask: np.ndarray, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    mask_h, mask_w = mask.shape[:2]
    if (mask_w, mask_h) == (img_w, img_h):
        return mask
    resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(mask.dtype, copy=False)


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
    paired = pair_sorted_rgb_with_masklet(image_dir, masklet)
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


def _build_object_index(views: List[ViewRecord]) -> tuple[list[int], Dict[int, List[ViewRecord]]]:
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
    return sorted(object_id_set), object_to_views


def load_scene_from_paths(
    dataset_id: str,
    resolved: ResolvedScenePaths,
    *,
    id_map_source: IdMapSource,
    pair_by: PairingStrategy,
) -> SceneData:
    """Load ``SceneData`` from absolute paths in a resolved manifest entry."""
    image_dir = resolved.image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if resolved.id_map_json is not None:
        if id_map_source != "sam2_json":
            raise ValueError(
                f"Manifest scene {resolved.scene_id!r} has id_map_json but id_map_source={id_map_source!r}"
            )
        map_dir, views = build_views_from_sam2_json(image_dir, resolved.id_map_json)
    elif resolved.id_map_dir is not None:
        map_dir = resolved.id_map_dir
        if not map_dir.is_dir():
            raise FileNotFoundError(f"ID map directory does not exist: {map_dir}")
        if id_map_source == "sam2_json":
            raise ValueError(
                f"Manifest scene {resolved.scene_id!r} has id_map_dir but id_map_source=sam2_json"
            )
        map_files = list_mask_paths(map_dir, id_map_source)
        pairs = pair_image_and_masks(image_dir, map_files, strategy=pair_by)
        views = []
        for view_name, image_path, map_path in pairs:
            mask = load_npy_id_map(map_path) if id_map_source == "npy" else load_mask(map_path)
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
        raise ValueError(f"Manifest scene {resolved.scene_id!r} has neither id_map_dir nor id_map_json")

    if not views:
        raise RuntimeError(f"No valid image-mask pairs found for scene: {resolved.scene_id}")

    object_ids, object_to_views = _build_object_index(views)
    return SceneData(
        dataset=dataset_id,
        scene=resolved.scene_id,
        scene_root=resolved.scene_root,
        image_dir=image_dir,
        mask_dir=map_dir,
        views=views,
        object_ids=object_ids,
        object_to_views=object_to_views,
    )
