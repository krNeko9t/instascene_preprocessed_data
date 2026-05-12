"""Load paired RGB / instance maps from disk (``SceneData``)."""

from .loader import (
    SceneData,
    ViewRecord,
    build_views_from_sam2_json,
    load_scene,
    load_scene_from_npy,
    load_scene_from_png,
)
from .pairing import (
    SUPPORTED_IMAGE_SUFFIXES,
    IdMapSource,
    PairingStrategy,
    build_image_index,
    list_mask_paths,
    pair_image_and_masks,
)

__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "IdMapSource",
    "PairingStrategy",
    "SceneData",
    "ViewRecord",
    "build_image_index",
    "build_views_from_sam2_json",
    "list_mask_paths",
    "load_scene",
    "load_scene_from_npy",
    "load_scene_from_png",
    "pair_image_and_masks",
]
