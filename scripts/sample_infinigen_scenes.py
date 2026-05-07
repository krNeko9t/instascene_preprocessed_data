from __future__ import annotations

from ins_scene_15k_roots import INFINIGEN_ROOT
from scene_path_discover_ins_scene_15k import discover_infinigen_extracted
from scene_path_sample_job import SampleJobConfig, run_sample_job

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_infinigen_extracted and write a scene-path manifest JSON "
        "(see also sample_re10k_scenes.py, sample_scannetpp_v2_scenes.py)."
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
