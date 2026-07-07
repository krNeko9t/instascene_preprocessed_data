from __future__ import annotations

from dataclasses import dataclass

from instascene.manifest.models import ResolvedScenePaths
from instascene.scene.loaders.scene import load_scene_from_paths
from instascene.types import IdMapSource, PairingStrategy

__all__ = ["ViewObjectStats", "summarize_manifest_scene"]


@dataclass(slots=True)
class ViewObjectStats:
    """Per-scene counts from the shared scene loader (non-negative ids only)."""

    scene_id: str
    n_views: int
    n_objects: int
    error: str | None = None


def summarize_manifest_scene(
    resolved: ResolvedScenePaths,
    dataset_id: str,
    id_map_source: IdMapSource,
    *,
    pair_by: PairingStrategy,
) -> ViewObjectStats:
    """Load one manifest scene and count views and distinct object ids."""
    try:
        scene_data = load_scene_from_paths(
            dataset_id,
            resolved,
            id_map_source=id_map_source,
            pair_by=pair_by,
        )
        return ViewObjectStats(
            scene_id=resolved.scene_id,
            n_views=len(scene_data.views),
            n_objects=len(scene_data.object_ids),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — record per-scene failure without aborting the batch
        return ViewObjectStats(
            scene_id=resolved.scene_id,
            n_views=0,
            n_objects=0,
            error=str(exc),
        )
