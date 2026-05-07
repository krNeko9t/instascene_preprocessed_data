from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ins_scene_15k_roots import SCANNETPPV2_ROOT, path_for_manifest_json


@dataclass(slots=True, frozen=True)
class SceneEntry:
    partition: str
    scene_name: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path


def discover_scannetpp_v2_scenes(scenes_root: Path) -> list[SceneEntry]:
    """Each scene has ``images/`` (jpg) and ``refined_ins_ids/`` (png instance maps, same stem)."""
    out: list[SceneEntry] = []
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = child / "images"
        ins_ids = child / "refined_ins_ids"
        if not images.is_dir() or not ins_ids.is_dir():
            continue
        out.append(
            SceneEntry(
                partition="processed_scannetpp_v2",
                scene_name=child.name,
                scene_root=child.resolve(),
                image_dir=images.resolve(),
                id_map_dir=ins_ids.resolve(),
            )
        )
    out.sort(key=lambda e: e.scene_name)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample scenes under InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2 "
            "and write a scene-path manifest (pair images/*.jpg with refined_ins_ids/*.png by stem)."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help=f"Scenes root. If omitted: {SCANNETPPV2_ROOT}",
    )
    parser.add_argument("-n", "--num-scenes", type=int, default=-1, help="-1 = all; must not be 0")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--relative-to", type=str, default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_scenes == 0:
        print("error: -n must not be 0", file=sys.stderr)
        return 2

    scenes_root = (
        Path(args.root).expanduser().resolve()
        if args.root.strip()
        else SCANNETPPV2_ROOT.expanduser().resolve()
    )
    if not scenes_root.is_dir():
        print(f"error: not a directory: {scenes_root}", file=sys.stderr)
        return 1

    relative_to = Path(args.relative_to).expanduser().resolve() if args.relative_to.strip() else None

    candidates = discover_scannetpp_v2_scenes(scenes_root)
    n_candidates = len(candidates)
    if n_candidates == 0:
        print(f"error: no valid scenes under: {scenes_root}", file=sys.stderr)
        return 1

    n_requested = args.num_scenes
    if n_requested != -1 and args.strict and n_requested > n_candidates:
        print(f"error: --strict: N={n_requested} > {n_candidates}", file=sys.stderr)
        return 3
    if args.seed is not None:
        random.seed(args.seed)

    if n_requested == -1:
        selected = list(candidates)
        if args.shuffle:
            random.shuffle(selected)
    else:
        k = min(n_requested, n_candidates)
        if k < n_requested:
            print(f"warning: only {n_candidates} scenes; selecting {k}", file=sys.stderr)
        selected = random.sample(candidates, k)

    base = relative_to
    payload = {
        "dataset_root": str(scenes_root),
        "pair_by": "stem",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_candidates": n_candidates,
        "num_selected": len(selected),
        "n_requested": n_requested,
        "seed": args.seed,
        "shuffle": bool(args.shuffle),
        "strict": bool(args.strict),
        "scenes": [
            {
                "partition": e.partition,
                "scene_name": e.scene_name,
                "scene_root": path_for_manifest_json(e.scene_root, relative_to=base),
                "image_dir": path_for_manifest_json(e.image_dir, relative_to=base),
                "id_map_dir": path_for_manifest_json(e.id_map_dir, relative_to=base),
            }
            for e in selected
        ],
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent <= 0 else args.indent
    out_path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} scene(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
