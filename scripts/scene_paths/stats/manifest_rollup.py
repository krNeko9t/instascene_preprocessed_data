from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

from scripts.scene_io.pairing import PairingStrategy

from ..manifest.paths_manifest import ScenePathsManifest, load_scene_paths_manifest, resolve_scene_paths
from .view_object_counts import summarize_manifest_scene


def effective_pairing(doc: ScenePathsManifest, cli_value: str) -> PairingStrategy:
    if cli_value != "auto":
        return cast(PairingStrategy, cli_value)
    meta = doc.metadata.get("pair_by")
    if meta in ("stem", "infinigen"):
        return cast(PairingStrategy, meta)
    return "infinigen"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count views and distinct object ids per scene from a scene-path manifest "
            "(from scene_paths.sampling.datasets: infinigen, re10k, scannetpp_v2)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest JSON from sampling tools",
    )
    parser.add_argument(
        "--id-map-source",
        type=str,
        default="png",
        choices=["npy", "png", "sam2_json"],
        help=(
            "Mask format under id_map_dir (npy|png), or sam2_json when manifest scenes only have "
            "id_map_json (auto_masks.json). Ignored for scenes that only specify id_map_json (RE10K)."
        ),
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
        help=(
            "auto: use manifest top-level pair_by if set (stem|infinigen), else infinigen. "
            "infinigen=Image_* vs ObjectSegmentation_*; stem=same basename (e.g. ScanNet++ jpg/png)."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    id_map_source: Literal["npy", "png", "sam2_json"] = args.id_map_source  # type: ignore[assignment]

    doc = load_scene_paths_manifest(manifest_path)
    pairing = effective_pairing(doc, args.pair_by)
    scene_rows: list[dict[str, Any]] = []
    n_errors = 0

    for entry in doc.scenes:
        resolved = resolve_scene_paths(entry, doc.dataset_root)
        stats = summarize_manifest_scene(resolved, id_map_source, pairing=pairing)
        row: dict[str, Any] = {
            "partition": entry.partition,
            "scene_name": entry.scene_name,
            "scene_key": stats.scene_key,
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
        "id_map_source": id_map_source,
        "pair_by": pairing,
        "manifest_metadata": dict(doc.metadata),
        "manifest_dataset_root": str(doc.dataset_root),
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
