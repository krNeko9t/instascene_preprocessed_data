from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from instascene.scene.models import ScenePathRecord

__all__ = [
    "ScenePathRecord",
    "ScenePathsManifestEntry",
    "ScenePathsManifest",
    "ResolvedScenePaths",
]


@dataclass(slots=True, frozen=True)
class ScenePathsManifestEntry:
    """One element of the manifest ``scenes[]`` array."""

    partition: str
    scene_name: str
    scene_root: str
    image_dir: str
    id_map_dir: str | None
    id_map_json: str | None = None


@dataclass(slots=True, frozen=True)
class ScenePathsManifest:
    """Top-level manifest: ``dataset_root`` plus a list of scene path bundles."""

    dataset_root: Path
    scenes: list[ScenePathsManifestEntry]
    metadata: Mapping[str, Any]


@dataclass(slots=True, frozen=True)
class ResolvedScenePaths:
    """Absolute paths for one scene, ready for pairing / IO."""

    partition: str
    scene_name: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path | None
    id_map_json: Path | None = None

    @property
    def scene_key(self) -> str:
        if self.partition:
            return f"{self.partition}/{self.scene_name}"
        return self.scene_name
