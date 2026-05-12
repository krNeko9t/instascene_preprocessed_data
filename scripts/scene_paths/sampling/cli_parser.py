from __future__ import annotations

import argparse
from pathlib import Path


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


def resolve_cli_root(cli_root: str, default_root: Path) -> Path:
    return Path(cli_root).expanduser().resolve() if cli_root.strip() else default_root.expanduser().resolve()
