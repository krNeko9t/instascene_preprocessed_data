"""Discover scenes under the local ``<dataset>/<scene>/images + mask`` layout."""

from __future__ import annotations

from pathlib import Path

from instascene.scene.ref import SceneRef

__all__ = ["discover_local_extracted"]


def discover_local_extracted(scenes_root: Path) -> list[SceneRef]:
    """One directory per scene with ``images/`` and ``id_maps/`` or ``sam/mask/``."""
    out: list[SceneRef] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = child / "images"
        if not images.is_dir():
            continue
        id_maps = child / "id_maps"
        sam_mask = child / "sam" / "mask"
        if not id_maps.is_dir() and not sam_mask.is_dir():
            continue
        out.append(SceneRef(scene_id=child.name))
    out.sort(key=lambda e: e.scene_id)
    return out
