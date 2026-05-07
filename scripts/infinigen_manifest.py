"""Scene path manifest: JSON with ``dataset_root`` and explicit per-scene paths.

Any tool that knows how to walk a dataset's on-disk layout can emit this shape; readers
only resolve paths and do not guess directory structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class ScenePathsManifestEntry:
    """One element of the manifest ``scenes[]`` array."""

    partition: str
    scene_name: str
    scene_root: str
    image_dir: str
    id_map_dir: str


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
    id_map_dir: Path

    @property
    def scene_key(self) -> str:
        return f"{self.partition}/{self.scene_name}"


def _resolve_path(raw: str, dataset_root: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (dataset_root / p).resolve()


def resolve_scene_paths(entry: ScenePathsManifestEntry, dataset_root: Path) -> ResolvedScenePaths:
    root = dataset_root.resolve()
    return ResolvedScenePaths(
        partition=entry.partition,
        scene_name=entry.scene_name,
        scene_root=_resolve_path(entry.scene_root, root),
        image_dir=_resolve_path(entry.image_dir, root),
        id_map_dir=_resolve_path(entry.id_map_dir, root),
    )


def load_scene_paths_manifest(path: Path) -> ScenePathsManifest:
    """Load and validate a scene-path manifest JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    if "dataset_root" not in data or "scenes" not in data:
        raise ValueError(f"Manifest missing dataset_root or scenes: {path}")
    if not isinstance(data["scenes"], list):
        raise ValueError(f"Manifest scenes must be a list: {path}")

    dataset_root = Path(str(data["dataset_root"])).expanduser().resolve()
    scenes: list[ScenePathsManifestEntry] = []
    for i, item in enumerate(data["scenes"]):
        if not isinstance(item, dict):
            raise ValueError(f"scenes[{i}] must be an object")
        required = ("partition", "scene_name", "scene_root", "image_dir", "id_map_dir")
        for key in required:
            if key not in item:
                raise ValueError(f"scenes[{i}] missing key {key!r}")
        scenes.append(
            ScenePathsManifestEntry(
                partition=str(item["partition"]),
                scene_name=str(item["scene_name"]),
                scene_root=str(item["scene_root"]),
                image_dir=str(item["image_dir"]),
                id_map_dir=str(item["id_map_dir"]),
            )
        )

    metadata = {k: v for k, v in data.items() if k not in {"dataset_root", "scenes"}}
    return ScenePathsManifest(dataset_root=dataset_root, scenes=scenes, metadata=metadata)
