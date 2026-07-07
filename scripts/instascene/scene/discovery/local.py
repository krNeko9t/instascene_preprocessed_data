"""Discover scenes under the local ``<dataset>/<scene>/images + mask`` layout."""

from __future__ import annotations

from pathlib import Path

from instascene.scene.models import ScenePathRecord

__all__ = ["discover_local_extracted"]


def discover_local_extracted(scenes_root: Path) -> list[ScenePathRecord]:
    """One directory per scene with ``images/`` and ``id_maps/`` or ``sam/mask/``."""
    out: list[ScenePathRecord] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = child / "images"
        if not images.is_dir():
            continue
        id_maps = child / "id_maps"
        sam_mask = child / "sam" / "mask"
        if id_maps.is_dir():
            id_map_dir = id_maps.resolve()
        elif sam_mask.is_dir():
            id_map_dir = sam_mask.resolve()
        else:
            continue
        out.append(
            ScenePathRecord(
                partition="",
                scene_name=child.name,
                scene_root=child.resolve(),
                image_dir=images.resolve(),
                id_map_dir=id_map_dir,
            )
        )
    out.sort(key=lambda e: e.scene_name)
    return out
