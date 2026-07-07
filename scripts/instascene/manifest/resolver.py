"""Resolve dataset scene_id values into absolute paths for loaders."""

from __future__ import annotations

from pathlib import Path

from instascene.manifest.models import ResolvedScenePaths

__all__ = [
    "resolve_infinigen_scene",
    "resolve_local_scene",
    "resolve_re10k_scene",
    "resolve_scannetpp_v2_scene",
    "resolve_scene_ref",
]


def resolve_local_scene(dataset_root: Path, scene_id: str) -> ResolvedScenePaths:
    scene_root = (dataset_root / scene_id).resolve()
    image_dir = scene_root / "images"
    id_maps = scene_root / "id_maps"
    sam_mask = scene_root / "sam" / "mask"
    if id_maps.is_dir():
        id_map_dir = id_maps.resolve()
    elif sam_mask.is_dir():
        id_map_dir = sam_mask.resolve()
    else:
        id_map_dir = id_maps.resolve()
    return ResolvedScenePaths(
        scene_id=scene_id,
        scene_root=scene_root,
        image_dir=image_dir.resolve(),
        id_map_dir=id_map_dir,
        id_map_json=None,
    )


def resolve_infinigen_scene(dataset_root: Path, scene_id: str) -> ResolvedScenePaths:
    if "/" not in scene_id:
        raise ValueError(f"infinigen scene_id must be '<partition>/<name>', got {scene_id!r}")
    partition, scene_name = scene_id.split("/", 1)
    scene_root = (dataset_root / partition / scene_name).resolve()
    frames = scene_root / "frames"
    return ResolvedScenePaths(
        scene_id=scene_id,
        scene_root=scene_root,
        image_dir=(frames / "Image" / "camera_0").resolve(),
        id_map_dir=(frames / "ObjectSegmentation" / "camera_0").resolve(),
        id_map_json=None,
    )


def resolve_re10k_scene(dataset_root: Path, scene_id: str) -> ResolvedScenePaths:
    scene_root = (dataset_root / scene_id).resolve()
    sam2_json = dataset_root / "sam2_results" / scene_id / "auto_masks.json"
    id_map_json = sam2_json.resolve() if sam2_json.is_file() else None
    return ResolvedScenePaths(
        scene_id=scene_id,
        scene_root=scene_root,
        image_dir=(scene_root / "rgb").resolve(),
        id_map_dir=None,
        id_map_json=id_map_json,
    )


def resolve_scannetpp_v2_scene(dataset_root: Path, scene_id: str) -> ResolvedScenePaths:
    scene_root = (dataset_root / scene_id).resolve()
    return ResolvedScenePaths(
        scene_id=scene_id,
        scene_root=scene_root,
        image_dir=(scene_root / "images").resolve(),
        id_map_dir=(scene_root / "refined_ins_ids").resolve(),
        id_map_json=None,
    )


_RESOLVERS = {
    "3dovs": resolve_local_scene,
    "lerf": resolve_local_scene,
    "zipnerf": resolve_local_scene,
    "infinigen": resolve_infinigen_scene,
    "re10k": resolve_re10k_scene,
    "scannetpp_v2": resolve_scannetpp_v2_scene,
}


def resolve_scene_ref(dataset_id: str, scene_id: str, dataset_root: Path) -> ResolvedScenePaths:
    try:
        resolver = _RESOLVERS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unknown dataset_id {dataset_id!r}") from exc
    return resolver(dataset_root.resolve(), scene_id)
