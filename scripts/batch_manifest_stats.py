from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from instascene.manifest.io import load_scene_selection_manifest
from instascene.manifest.resolver import resolve_scene_ref
from instascene.manifest.sample_registry import get_dataset_spec
from instascene.stats.scene_views import summarize_manifest_scene
from instascene.types import IdMapSource, PairingStrategy


def effective_id_map_source(dataset_id: str, cli_value: str) -> IdMapSource:
    if cli_value != "auto":
        return cast(IdMapSource, cli_value)
    return get_dataset_spec(dataset_id).id_map_source


def effective_pairing(dataset_id: str, cli_value: str) -> PairingStrategy:
    if cli_value != "auto":
        return cast(PairingStrategy, cli_value)
    return get_dataset_spec(dataset_id).pair_by


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count views and distinct object ids per scene from a scene selection lockfile "
            "(from sample_scenes.py --dataset ...)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to scene selection lockfile (from sample_scenes.py)",
    )
    parser.add_argument(
        "--id-map-source",
        type=str,
        default="auto",
        choices=["auto", "npy", "png", "sam2_json"],
        help="Mask format; auto reads dataset registry (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Write aggregated stats JSON to this path",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first scene that raises an error",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a short human-readable summary to stdout",
    )
    parser.add_argument(
        "--pair-by",
        type=str,
        default="auto",
        choices=["auto", "stem", "infinigen"],
        help="Pairing strategy; auto reads dataset registry (default: auto).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()

    doc = load_scene_selection_manifest(manifest_path)
    scene_rows: list[dict[str, Any]] = []
    n_errors = 0

    for entry in doc.scenes:
        dataset_root = doc.dataset_roots[entry.dataset_id]
        id_map_source = effective_id_map_source(entry.dataset_id, args.id_map_source)
        pairing = effective_pairing(entry.dataset_id, args.pair_by)
        resolved = resolve_scene_ref(entry.dataset_id, entry.scene_id, dataset_root)
        stats = summarize_manifest_scene(
            resolved,
            entry.dataset_id,
            id_map_source,
            pair_by=pairing,
        )
        row: dict[str, Any] = {
            "dataset_id": entry.dataset_id,
            "scene_id": entry.scene_id,
            "scene_key": stats.scene_key,
            "id_map_source": id_map_source,
            "pair_by": pairing,
            "n_views": stats.n_views,
            "n_objects": stats.n_objects,
            "resolved_scene_root": str(resolved.scene_root),
            "resolved_image_dir": str(resolved.image_dir),
            "resolved_id_map_dir": str(resolved.id_map_dir) if resolved.id_map_dir is not None else None,
            "resolved_id_map_json": str(resolved.id_map_json) if resolved.id_map_json is not None else None,
        }
        if stats.error is not None:
            row["error"] = stats.error
            n_errors += 1
            scene_rows.append(row)
            if args.fail_fast:
                print(f"fail-fast: {stats.scene_key}: {stats.error}", file=sys.stderr)
                break
            continue
        scene_rows.append(row)

    n_scenes = len(scene_rows)
    n_ok = n_scenes - n_errors
    sum_views = sum(r["n_views"] for r in scene_rows)
    n_scenes_with_views = sum(1 for r in scene_rows if r.get("error") is None and r["n_views"] > 0)

    payload: dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "schema_version": doc.schema_version,
        "kind": doc.kind,
        "dataset_roots": {k: str(v) for k, v in doc.dataset_roots.items()},
        "sampling": {
            "num_candidates": doc.sampling.num_candidates,
            "num_selected": doc.sampling.num_selected,
            "n_requested": doc.sampling.n_requested,
            "seed": doc.sampling.seed,
            "shuffle": doc.sampling.shuffle,
            "strict": doc.sampling.strict,
        },
        "scenes": scene_rows,
        "totals": {
            "n_scenes": n_scenes,
            "n_ok": n_ok,
            "n_errors": n_errors,
            "n_scenes_with_views": n_scenes_with_views,
            "sum_views": sum_views,
        },
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.print_summary:
        print(f"scenes={n_scenes} ok={n_ok} errors={n_errors} sum_views={sum_views}")
    print(f"wrote {out_path}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
