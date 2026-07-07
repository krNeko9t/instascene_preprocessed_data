"""Scene path manifest IO: load, resolve, and validate JSON manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from instascene.manifest.models import (
    ResolvedScenePaths,
    ScenePathsManifest,
    ScenePathsManifestEntry,
)
from instascene.types import IdMapSource, PairingStrategy

__all__ = [
    "load_scene_paths_manifest",
    "manifest_dataset_id",
    "manifest_id_map_source",
    "manifest_pair_by",
    "resolve_scene_paths",
]


def _resolve_path(raw: str, dataset_root: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (dataset_root / p).resolve()


def resolve_scene_paths(entry: ScenePathsManifestEntry, dataset_root: Path) -> ResolvedScenePaths:
    root = dataset_root.resolve()
    id_map: Path | None = None
    if entry.id_map_dir:
        id_map = _resolve_path(entry.id_map_dir, root)
    id_json: Path | None = None
    if entry.id_map_json:
        id_json = _resolve_path(entry.id_map_json, root)
    return ResolvedScenePaths(
        partition=entry.partition,
        scene_name=entry.scene_name,
        scene_root=_resolve_path(entry.scene_root, root),
        image_dir=_resolve_path(entry.image_dir, root),
        id_map_dir=id_map,
        id_map_json=id_json,
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
        for key in ("partition", "scene_name", "scene_root", "image_dir"):
            if key not in item:
                raise ValueError(f"scenes[{i}] missing key {key!r}")

        def _optional_str(raw: object) -> str | None:
            if raw is None or raw == "": return None
            return str(raw)
            
        scenes.append(
            ScenePathsManifestEntry(
                partition=str(item["partition"]),
                scene_name=str(item["scene_name"]),
                scene_root=str(item["scene_root"]),
                image_dir=str(item["image_dir"]),
                id_map_dir=_optional_str(item.get("id_map_dir")),
                id_map_json=_optional_str(item.get("id_map_json")),
            )
        )

    metadata = {k: v for k, v in data.items() if k not in {"dataset_root", "scenes"}}
    return ScenePathsManifest(dataset_root=dataset_root, scenes=scenes, metadata=metadata)


def manifest_dataset_id(doc: ScenePathsManifest) -> str:
    raw = doc.metadata.get("dataset_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "unknown"


def manifest_id_map_source(doc: ScenePathsManifest) -> IdMapSource:
    raw = doc.metadata.get("id_map_source")
    if raw in ("npy", "png", "sam2_json"):
        return cast(IdMapSource, raw)
    for entry in doc.scenes:
        if entry.id_map_json:
            return "sam2_json"
        if entry.id_map_dir:
            suffix = Path(entry.id_map_dir).suffix.lower()
            if suffix == ".npy" or "id_maps" in entry.id_map_dir:
                return "npy"
            return "png"
    return "png"


def manifest_pair_by(doc: ScenePathsManifest, *, default: PairingStrategy = "stem") -> PairingStrategy:
    raw = doc.metadata.get("pair_by")
    if raw in ("stem", "infinigen"):
        return cast(PairingStrategy, raw)
    return default
