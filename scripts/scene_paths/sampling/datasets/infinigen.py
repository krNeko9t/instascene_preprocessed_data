from __future__ import annotations

from ...paths.discover import discover_infinigen_extracted
from ...paths.roots import INFINIGEN_ROOT
from ..job import SampleJobConfig, run_sample_job

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_infinigen_extracted and write a scene-path manifest JSON "
        "(see also re10k, scannetpp_v2 under sampling.datasets)."
    ),
    default_root=INFINIGEN_ROOT,
    discover=discover_infinigen_extracted,
    pair_by="infinigen",
    empty_candidates_template="error: no valid scenes found under: {root}",
    root_not_dir_template="error: dataset root is not a directory: {root}",
)


def main() -> int:
    return run_sample_job(_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
