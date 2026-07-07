from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from instascene.manifest.roots import path_for_manifest_json
from instascene.scene.models import ScenePathRecord
from instascene.types import IdMapSource, PairingStrategy

__all__ = [
    "SampleJobConfig",
    "build_sample_argparser",
    "run_sample_job",
]


@dataclass(frozen=True, slots=True)
class SampleJobConfig:
    """Bind one dataset: default root, discover function, manifest pair_by, and stderr messages."""

    dataset_id: str
    description: str
    default_root: Path
    discover: Callable[[Path], list[ScenePathRecord]]
    id_map_source: IdMapSource
    pair_by: PairingStrategy | None
    empty_candidates_template: str
    root_not_dir_template: str = "error: not a directory: {root}"


def build_sample_argparser(description: str, *, default_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help=f"Dataset / scenes root. If omitted, uses: {default_root}",
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


def resolve_root(cli_root: str, default_root: Path) -> Path:
    return Path(cli_root).expanduser().resolve() if cli_root.strip() else default_root.expanduser().resolve()


def select_scene_subset(
    candidates: Sequence[ScenePathRecord],
    *,
    n_requested: int,
    seed: int | None,
    shuffle: bool,
) -> list[ScenePathRecord]:
    rng = random.Random(seed)
    if n_requested == -1:
        selected = list(candidates)
        if shuffle:
            rng.shuffle(selected)
        return selected
    k = min(n_requested, len(candidates))
    if k < n_requested:
        print(
            f"warning: requested N={n_requested} but only {len(candidates)} candidates; selecting {k}",
            file=sys.stderr,
        )
    return rng.sample(list(candidates), k)


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
    dataset_id: str,
    id_map_source: IdMapSource,
    n_candidates: int,
    n_selected: int,
    n_requested: int,
    seed: int | None,
    shuffle: bool,
    strict: bool,
    pair_by: PairingStrategy | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "id_map_source": id_map_source,
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


def run_sample_job(config: SampleJobConfig, argv: list[str] | None = None) -> int:
    parser = build_sample_argparser(config.description, default_root=config.default_root)
    args = parser.parse_args(argv)

    if args.num_scenes == 0:
        print("error: --num-scenes / -n must not be 0 (use -1 for all scenes)", file=sys.stderr)
        return 2

    dataset_root = resolve_root(args.root, config.default_root)
    if not dataset_root.is_dir():
        print(config.root_not_dir_template.format(root=dataset_root), file=sys.stderr)
        return 1

    relative_to = None
    if args.relative_to.strip():
        relative_to = Path(args.relative_to).expanduser().resolve()

    candidates = config.discover(dataset_root)
    n_candidates = len(candidates)
    if n_candidates == 0:
        print(config.empty_candidates_template.format(root=dataset_root), file=sys.stderr)
        return 1

    n_requested = args.num_scenes
    if n_requested != -1 and args.strict and n_requested > n_candidates:
        print(
            f"error: --strict: requested N={n_requested} exceeds candidate count={n_candidates}",
            file=sys.stderr,
        )
        return 3

    selected = select_scene_subset(
        candidates,
        n_requested=n_requested,
        seed=args.seed,
        shuffle=bool(args.shuffle),
    )

    scenes = records_to_manifest_scenes(selected, relative_to)
    payload = build_manifest_payload(
        dataset_root,
        scenes,
        dataset_id=config.dataset_id,
        id_map_source=config.id_map_source,
        n_candidates=n_candidates,
        n_selected=len(selected),
        n_requested=n_requested,
        seed=args.seed,
        shuffle=bool(args.shuffle),
        strict=bool(args.strict),
        pair_by=config.pair_by,
    )

    out_path = Path(args.output)
    write_manifest(out_path, payload, args.indent)
    print(f"wrote {len(selected)} scene(s) to {out_path.expanduser().resolve()}")
    return 0
