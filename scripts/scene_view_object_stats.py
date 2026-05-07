from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from data_loader import _load_mask, _load_npy_id_map
from infinigen_manifest import ResolvedScenePaths
from view_pairing import IdMapSource, PairingStrategy, list_mask_paths, pair_image_and_masks


@dataclass(slots=True)
class ViewObjectStats:
    """Per-scene counts aligned with load_scene pairing and object id rules (non-negative ids only)."""

    scene_key: str
    n_views: int
    n_objects: int
    error: str | None = None


def summarize_from_paired_masks(
    scene_key: str,
    pairs: Sequence[Tuple[str, Path, Path]],
    id_map_source: IdMapSource,
) -> ViewObjectStats:
    """Union object ids across paired masks; no resize (stats-only, see plan)."""
    object_ids: set[int] = set()
    for _, _, mask_path in pairs:
        if id_map_source == "npy":
            mask = _load_npy_id_map(mask_path)
        else:
            mask = _load_mask(mask_path)
        for v in np.unique(mask):
            oid = int(v)
            if oid >= 0:
                object_ids.add(oid)
    return ViewObjectStats(scene_key=scene_key, n_views=len(pairs), n_objects=len(object_ids), error=None)


def summarize_manifest_scene(
    resolved: ResolvedScenePaths,
    id_map_source: IdMapSource,
    *,
    pairing: PairingStrategy = "infinigen",
) -> ViewObjectStats:
    """Pair dirs from a scene-path manifest entry, then count views and distinct object ids."""
    try:
        if not resolved.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir is not a directory: {resolved.image_dir}")
        if not resolved.id_map_dir.is_dir():
            raise FileNotFoundError(f"id_map_dir is not a directory: {resolved.id_map_dir}")
        mask_paths = list_mask_paths(resolved.id_map_dir, id_map_source)
        pairs = pair_image_and_masks(resolved.image_dir, mask_paths, strategy=pairing)
        return summarize_from_paired_masks(resolved.scene_key, pairs, id_map_source)
    except Exception as exc:  # noqa: BLE001 — record per-scene failure without aborting the batch
        return ViewObjectStats(
            scene_key=resolved.scene_key,
            n_views=0,
            n_objects=0,
            error=str(exc),
        )
