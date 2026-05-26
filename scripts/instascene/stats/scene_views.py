from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from instascene.manifest.models import ResolvedScenePaths
from instascene.scene.loaders.masks import load_mask, load_npy_id_map
from instascene.scene.loaders.scene import build_views_from_sam2_json
from instascene.scene.pairing import build_image_index, list_mask_paths, pair_image_and_masks
from instascene.types import IdMapSource, PairingStrategy

__all__ = ["ViewObjectStats", "summarize_manifest_scene"]


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
    object_ids: set[int] = set()
    for _, _, mask_path in pairs:
        if id_map_source == "npy":
            mask = load_npy_id_map(mask_path)
        elif id_map_source == "png":
            mask = load_mask(mask_path)
        else:
            raise ValueError(f"id_map_source {id_map_source!r} requires id_map_json, not id_map_dir")
        for v in np.unique(mask):
            oid = int(v)
            if oid >= 0:
                object_ids.add(oid)
    return ViewObjectStats(scene_key=scene_key, n_views=len(pairs), n_objects=len(object_ids), error=None)


def summarize_from_sam2_json(scene_key: str, image_dir: Path, auto_masks_json: Path) -> ViewObjectStats:
    """Same decode + pairing as build_views_from_sam2_json; union distinct ids >= 0."""
    _, views = build_views_from_sam2_json(image_dir, auto_masks_json)
    object_ids: set[int] = set()
    for v in views:
        for u in np.unique(v.mask):
            oid = int(u)
            if oid >= 0:
                object_ids.add(oid)
    return ViewObjectStats(scene_key=scene_key, n_views=len(views), n_objects=len(object_ids), error=None)


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

        if resolved.id_map_dir is not None:
            if id_map_source == "sam2_json":
                raise ValueError(
                    "Manifest entry has id_map_dir: use --id-map-source npy or png for directory masks, "
                    "not sam2_json."
                )
            if not resolved.id_map_dir.is_dir():
                raise FileNotFoundError(f"id_map_dir is not a directory: {resolved.id_map_dir}")
            mask_paths = list_mask_paths(resolved.id_map_dir, id_map_source)
            pairs = pair_image_and_masks(resolved.image_dir, mask_paths, strategy=pairing)
            return summarize_from_paired_masks(resolved.scene_key, pairs, id_map_source)

        if resolved.id_map_json is not None:
            if not resolved.id_map_json.is_file():
                raise FileNotFoundError(f"id_map_json is not a file: {resolved.id_map_json}")
            return summarize_from_sam2_json(resolved.scene_key, resolved.image_dir, resolved.id_map_json)

        n_views = len(build_image_index(resolved.image_dir))
        return ViewObjectStats(
            scene_key=resolved.scene_key,
            n_views=n_views,
            n_objects=0,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — record per-scene failure without aborting the batch
        return ViewObjectStats(
            scene_key=resolved.scene_key,
            n_views=0,
            n_objects=0,
            error=str(exc),
        )
