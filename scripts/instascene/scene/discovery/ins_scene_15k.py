"""Discover valid scene directories in InsScene-15K dataset layouts."""

from __future__ import annotations

from pathlib import Path

from instascene.scene.ref import SceneRef

__all__ = [
    "discover_infinigen_scenes",
    "discover_re10k_scenes",
    "discover_scannetpp_v2_scenes",
]


def discover_infinigen_scenes(dataset_root: Path) -> list[SceneRef]:
    """``scene_* / <name> / frames/Image/camera_0`` + ``.../ObjectSegmentation/camera_0``."""
    out: list[SceneRef] = []
    scene_groups = sorted(
        p for p in dataset_root.glob("scene_*") if p.is_dir() and not p.name.startswith(".")
    )
    for scene_group in scene_groups:
        for child in sorted(scene_group.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            frames = child / "frames"
            image_dir = frames / "Image" / "camera_0"
            id_map_dir = frames / "ObjectSegmentation" / "camera_0"
            if not image_dir.is_dir() or not id_map_dir.is_dir():
                continue
            out.append(SceneRef(scene_id=f"{scene_group.name}/{child.name}"))
    out.sort(key=lambda e: e.scene_id)
    return out


def discover_re10k_scenes(scenes_root: Path) -> list[SceneRef]:
    """One directory per scene with ``rgb/`` and ``cam/``."""
    out: list[SceneRef] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        rgb = child / "rgb"
        cam = child / "cam"
        if not rgb.is_dir() or not cam.is_dir():
            continue
        out.append(SceneRef(scene_id=child.name))
    out.sort(key=lambda e: e.scene_id)
    return out


def discover_scannetpp_v2_scenes(scenes_root: Path) -> list[SceneRef]:
    """``images/`` + ``refined_ins_ids/`` per scene (jpg vs png, same stem)."""
    out: list[SceneRef] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = child / "images"
        ins_ids = child / "refined_ins_ids"
        if not images.is_dir() or not ins_ids.is_dir():
            continue
        out.append(SceneRef(scene_id=child.name))
    out.sort(key=lambda e: e.scene_id)
    return out
