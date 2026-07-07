from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ResolvedScenePaths",
    "SamplingMetadata",
    "SceneSelectionEntry",
    "SceneSelectionManifest",
    "SCHEMA_VERSION",
    "MANIFEST_KIND",
]

SCHEMA_VERSION = 2
MANIFEST_KIND = "scene_selection"


@dataclass(slots=True, frozen=True)
class SceneSelectionEntry:
    """One selected scene: dataset identity plus stable scene id."""

    dataset_id: str
    scene_id: str


@dataclass(slots=True, frozen=True)
class SamplingMetadata:
    num_candidates: int
    num_selected: int
    n_requested: int
    seed: int | None
    shuffle: bool
    strict: bool


@dataclass(slots=True, frozen=True)
class SceneSelectionManifest:
    """Scene selection lockfile: fixed random sample, resolved at runtime."""

    schema_version: int
    kind: str
    generated_at_utc: str
    sampling: SamplingMetadata
    dataset_roots: dict[str, Path]
    scenes: list[SceneSelectionEntry]


@dataclass(slots=True, frozen=True)
class ResolvedScenePaths:
    """Absolute paths for one scene, ready for pairing / IO."""

    scene_id: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path | None
    id_map_json: Path | None = None
