"""Discover valid scene directories under InsScene-15K processed extracts."""

from __future__ import annotations

from pathlib import Path

from instascene.scene.models import ScenePathRecord

__all__ = [
    "discover_infinigen_extracted",
    "discover_re10k_extracted",
    "discover_scannetpp_v2_extracted",
]


def discover_infinigen_extracted(dataset_root: Path) -> list[ScenePathRecord]:
    """``scene_* / <name> / frames/Image/camera_0`` + ``.../ObjectSegmentation/camera_0``."""
    out: list[ScenePathRecord] = []
    partitions = sorted(
        p for p in dataset_root.glob("scene_*") if p.is_dir() and not p.name.startswith(".")
    )
    for partition in partitions:
        for child in sorted(partition.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            frames = child / "frames"
            image_dir = frames / "Image" / "camera_0"
            id_map_dir = frames / "ObjectSegmentation" / "camera_0"
            if not image_dir.is_dir() or not id_map_dir.is_dir():
                continue
            out.append(
                ScenePathRecord(
                    partition=partition.name,
                    scene_name=child.name,
                    scene_root=child.resolve(),
                    image_dir=image_dir.resolve(),
                    id_map_dir=id_map_dir.resolve(),
                )
            )
    out.sort(key=lambda e: (e.partition, e.scene_name))
    return out


def discover_re10k_extracted(scenes_root: Path) -> list[ScenePathRecord]:
    """One directory per scene with ``rgb/`` and ``cam/``; optional ``sam2_results/<id>/auto_masks.json``."""
    out: list[ScenePathRecord] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        rgb = child / "rgb"
        cam = child / "cam"
        if not rgb.is_dir() or not cam.is_dir():
            continue
        sam2_json = scenes_root / "sam2_results" / child.name / "auto_masks.json"
        id_map_json = sam2_json.resolve() if sam2_json.is_file() else None
        out.append(
            ScenePathRecord(
                partition="processed_re10k",
                scene_name=child.name,
                scene_root=child.resolve(),
                image_dir=rgb.resolve(),
                id_map_dir=None,
                id_map_json=id_map_json,
            )
        )
    out.sort(key=lambda e: e.scene_name)
    return out


def discover_scannetpp_v2_extracted(scenes_root: Path) -> list[ScenePathRecord]:
    """``images/`` + ``refined_ins_ids/`` per scene (jpg vs png, same stem)."""
    out: list[ScenePathRecord] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = child / "images"
        ins_ids = child / "refined_ins_ids"
        if not images.is_dir() or not ins_ids.is_dir():
            continue
        out.append(
            ScenePathRecord(
                partition="processed_scannetpp_v2",
                scene_name=child.name,
                scene_root=child.resolve(),
                image_dir=images.resolve(),
                id_map_dir=ins_ids.resolve(),
            )
        )
    out.sort(key=lambda e: e.scene_name)
    return out
