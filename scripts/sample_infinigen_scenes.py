from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INFINIGEN_ROOT = (
    "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/dataset/"
    "InsScene-15K/processed_infinigen_extracted"
)


@dataclass(slots=True, frozen=True)
class SceneEntry:
    partition: str
    scene_name: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path


def discover_infinigen_scenes(dataset_root: Path) -> list[SceneEntry]:
    """List scenes under ``scene_*`` with ``frames/Image/camera_0`` and ``frames/ObjectSegmentation/camera_0``."""
    out: list[SceneEntry] = []
    partitions = sorted(
        p for p in dataset_root.glob("scene_*") if p.is_dir() and not p.name.startswith(".")
    )
    for partition in partitions:
        for child in sorted(partition.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue
            frames = child / "frames"
            image_dir = frames / "Image" / "camera_0"
            id_map_dir = frames / "ObjectSegmentation" / "camera_0"
            if not image_dir.is_dir() or not id_map_dir.is_dir():
                continue
            out.append(
                SceneEntry(
                    partition=partition.name,
                    scene_name=child.name,
                    scene_root=child.resolve(),
                    image_dir=image_dir.resolve(),
                    id_map_dir=id_map_dir.resolve(),
                )
            )
    out.sort(key=lambda e: (e.partition, e.scene_name))
    return out


def _path_for_json(path: Path, *, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    try:
        return str(path.relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample scenes under processed_infinigen_extracted layout and write a scene-path manifest JSON. "
            "re10k / spp2 need their own layout-specific scripts when you define those trees."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help=f"Dataset root (scene_* partitions). If omitted, uses: {DEFAULT_INFINIGEN_ROOT}",
    )
    parser.add_argument(
        "-n",
        "--num-scenes",
        type=int,
        default=-1,
        help="Number of scenes to sample; -1 means all (default: -1). Must not be 0.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (ignored when -n is -1 and --shuffle is not set)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2); use 0 for compact single-line output",
    )
    parser.add_argument(
        "--relative-to",
        type=str,
        default="",
        help="If set, emit paths relative to this directory instead of absolute",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle order (--n=-1 yields all scenes in random order; uses --seed if set)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, exit with error when N is greater than the number of candidates",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_scenes == 0:
        print("error: --num-scenes / -n must not be 0 (use -1 for all scenes)", file=sys.stderr)
        return 2

    dataset_root = (
        Path(args.root).expanduser().resolve()
        if args.root.strip()
        else Path(DEFAULT_INFINIGEN_ROOT).expanduser().resolve()
    )

    if not dataset_root.is_dir():
        print(f"error: dataset root is not a directory: {dataset_root}", file=sys.stderr)
        return 1

    relative_to = None
    if args.relative_to.strip():
        relative_to = Path(args.relative_to).expanduser().resolve()

    candidates = discover_infinigen_scenes(dataset_root)
    n_candidates = len(candidates)
    if n_candidates == 0:
        print(f"error: no valid scenes found under: {dataset_root}", file=sys.stderr)
        return 1

    n_requested = args.num_scenes
    seed = args.seed

    if n_requested != -1 and args.strict and n_requested > n_candidates:
        print(
            f"error: --strict: requested N={n_requested} exceeds candidate count={n_candidates}",
            file=sys.stderr,
        )
        return 3

    if seed is not None:
        random.seed(seed)

    if n_requested == -1:
        selected = list(candidates)
        if args.shuffle:
            random.shuffle(selected)
    else:
        k = min(n_requested, n_candidates)
        if k < n_requested:
            print(
                f"warning: requested N={n_requested} but only {n_candidates} candidates; "
                f"selecting {k}",
                file=sys.stderr,
            )
        selected = random.sample(candidates, k)

    base = relative_to if relative_to is not None else None
    payload = {
        "dataset_root": str(dataset_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_candidates": n_candidates,
        "num_selected": len(selected),
        "n_requested": n_requested,
        "seed": seed,
        "shuffle": bool(args.shuffle),
        "strict": bool(args.strict),
        "scenes": [
            {
                "partition": e.partition,
                "scene_name": e.scene_name,
                "scene_root": _path_for_json(e.scene_root, relative_to=base),
                "image_dir": _path_for_json(e.image_dir, relative_to=base),
                "id_map_dir": _path_for_json(e.id_map_dir, relative_to=base),
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
