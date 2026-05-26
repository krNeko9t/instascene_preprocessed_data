from __future__ import annotations

import argparse
import sys

from instascene.manifest.sample import build_sample_argparser, run_sample_job
from instascene.manifest.sample_registry import SAMPLE_JOB_CONFIGS, get_sample_job_config

_DESCRIPTION = (
    "Sample scenes from an InsScene-15K processed extract and write a scene-path manifest JSON."
)

_DATASET_HELP = "\n".join(
    f"  {name}: {cfg.description.strip()}"
    for name, cfg in sorted(SAMPLE_JOB_CONFIGS.items())
)


def _print_combined_help() -> None:
    pre = argparse.ArgumentParser(description=_DESCRIPTION, add_help=False)
    pre.add_argument(
        "--dataset",
        choices=tuple(SAMPLE_JOB_CONFIGS),
        required=True,
        help="Dataset extract to sample",
    )
    print(pre.format_help())
    print("datasets:")
    print(_DATASET_HELP)
    print()
    print("Additional arguments (after --dataset):")
    example_config = next(iter(SAMPLE_JOB_CONFIGS.values()))
    build_sample_argparser(
        "Default --root depends on --dataset when omitted.",
        default_root=example_config.default_root,
    ).print_help()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "-h" in argv or "--help" in argv:
        if "--dataset" not in argv:
            _print_combined_help()
            return 0

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", choices=tuple(SAMPLE_JOB_CONFIGS), required=True)
    pre_args, rest = pre.parse_known_args(argv)
    config = get_sample_job_config(pre_args.dataset)
    return run_sample_job(config, rest)


if __name__ == "__main__":
    raise SystemExit(main())
