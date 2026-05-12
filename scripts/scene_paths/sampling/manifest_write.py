from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..paths.record import ScenePathRecord
from ..paths.roots import path_for_manifest_json


def records_to_manifest_scenes(
    records: Sequence[ScenePathRecord],
    relative_to: Path | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for e in records:
        row: dict[str, str] = {
            "partition": e.partition,
            "scene_name": e.scene_name,
            "scene_root": path_for_manifest_json(e.scene_root, relative_to=relative_to),
            "image_dir": path_for_manifest_json(e.image_dir, relative_to=relative_to),
        }
        if e.id_map_dir is not None:
            row["id_map_dir"] = path_for_manifest_json(e.id_map_dir, relative_to=relative_to)
        if e.id_map_json is not None:
            row["id_map_json"] = path_for_manifest_json(e.id_map_json, relative_to=relative_to)
        rows.append(row)
    return rows


def build_manifest_payload(
    dataset_root: Path,
    scenes: list[dict[str, str]],
    *,
    n_candidates: int,
    n_selected: int,
    n_requested: int,
    seed: int | None,
    shuffle: bool,
    strict: bool,
    pair_by: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_candidates": n_candidates,
        "num_selected": n_selected,
        "n_requested": n_requested,
        "seed": seed,
        "shuffle": shuffle,
        "strict": strict,
        "scenes": scenes,
    }
    if pair_by is not None:
        payload["pair_by"] = pair_by
    return payload


def write_manifest(path: Path, payload: dict[str, object], indent: int) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ind = None if indent <= 0 else indent
    path.write_text(json.dumps(payload, indent=ind, ensure_ascii=False) + "\n", encoding="utf-8")
