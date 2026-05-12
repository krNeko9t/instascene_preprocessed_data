from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..paths.record import ScenePathRecord
from .cli_parser import build_sample_argparser, resolve_cli_root
from .manifest_write import build_manifest_payload, records_to_manifest_scenes, write_manifest
from .selection import select_scene_subset


@dataclass(frozen=True, slots=True)
class SampleJobConfig:
    """Bind one dataset: default root, discover function, manifest pair_by, and stderr messages."""

    description: str
    default_root: Path
    discover: Callable[[Path], list[ScenePathRecord]]
    pair_by: str | None
    empty_candidates_template: str
    root_not_dir_template: str = "error: not a directory: {root}"


def run_sample_job(config: SampleJobConfig, argv: list[str] | None = None) -> int:
    parser = build_sample_argparser(config.description, default_root=config.default_root)
    args = parser.parse_args(argv)

    if args.num_scenes == 0:
        print("error: --num-scenes / -n must not be 0 (use -1 for all scenes)", file=sys.stderr)
        return 2

    dataset_root = resolve_cli_root(args.root, config.default_root)
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

    if args.seed is not None:
        random.seed(args.seed)

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
