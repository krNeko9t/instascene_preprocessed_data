"""Scene selection lockfile IO."""

from __future__ import annotations

import json
from pathlib import Path

from instascene.manifest.models import (
    MANIFEST_KIND,
    SCHEMA_VERSION,
    SamplingMetadata,
    SceneSelectionEntry,
    SceneSelectionManifest,
)

__all__ = ["load_scene_selection_manifest"]


def _require_int(data: dict[str, object], key: str) -> int:
    raw = data[key]
    if not isinstance(raw, int):
        raise ValueError(f"Manifest field {key!r} must be an integer")
    return raw


def _require_bool(data: dict[str, object], key: str) -> bool:
    raw = data[key]
    if not isinstance(raw, bool):
        raise ValueError(f"Manifest field {key!r} must be a boolean")
    return raw


def load_scene_selection_manifest(path: Path) -> SceneSelectionManifest:
    """Load and validate a v2 scene selection lockfile."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")

    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema_version must be {SCHEMA_VERSION}, got {schema_version!r}: {path}"
        )
    if data.get("kind") != MANIFEST_KIND:
        raise ValueError(f"Manifest kind must be {MANIFEST_KIND!r}: {path}")

    for key in ("generated_at_utc", "sampling", "dataset_roots", "scenes"):
        if key not in data:
            raise ValueError(f"Manifest missing key {key!r}: {path}")

    sampling_raw = data["sampling"]
    if not isinstance(sampling_raw, dict):
        raise ValueError(f"Manifest sampling must be an object: {path}")
    for key in ("num_candidates", "num_selected", "n_requested", "shuffle", "strict"):
        if key not in sampling_raw:
            raise ValueError(f"Manifest sampling missing key {key!r}: {path}")

    roots_raw = data["dataset_roots"]
    if not isinstance(roots_raw, dict) or not roots_raw:
        raise ValueError(f"Manifest dataset_roots must be a non-empty object: {path}")

    dataset_roots: dict[str, Path] = {}
    for dataset_id, root_raw in roots_raw.items():
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError(f"Manifest dataset_roots keys must be non-empty strings: {path}")
        dataset_roots[dataset_id] = Path(str(root_raw)).expanduser().resolve()

    scenes_raw = data["scenes"]
    if not isinstance(scenes_raw, list):
        raise ValueError(f"Manifest scenes must be a list: {path}")

    scenes: list[SceneSelectionEntry] = []
    for i, item in enumerate(scenes_raw):
        if not isinstance(item, dict):
            raise ValueError(f"scenes[{i}] must be an object")
        for key in ("dataset_id", "scene_id"):
            if key not in item:
                raise ValueError(f"scenes[{i}] missing key {key!r}")
        dataset_id = str(item["dataset_id"]).strip()
        scene_id = str(item["scene_id"]).strip()
        if not dataset_id or not scene_id:
            raise ValueError(f"scenes[{i}] dataset_id and scene_id must be non-empty")
        if dataset_id not in dataset_roots:
            raise ValueError(
                f"scenes[{i}] references unknown dataset_id {dataset_id!r}; "
                f"known: {sorted(dataset_roots)}"
            )
        scenes.append(SceneSelectionEntry(dataset_id=dataset_id, scene_id=scene_id))

    seed_raw = sampling_raw.get("seed")
    seed = seed_raw if isinstance(seed_raw, int) else None

    return SceneSelectionManifest(
        schema_version=SCHEMA_VERSION,
        kind=MANIFEST_KIND,
        generated_at_utc=str(data["generated_at_utc"]),
        sampling=SamplingMetadata(
            num_candidates=_require_int(sampling_raw, "num_candidates"),
            num_selected=_require_int(sampling_raw, "num_selected"),
            n_requested=_require_int(sampling_raw, "n_requested"),
            seed=seed,
            shuffle=_require_bool(sampling_raw, "shuffle"),
            strict=_require_bool(sampling_raw, "strict"),
        ),
        dataset_roots=dataset_roots,
        scenes=scenes,
    )
