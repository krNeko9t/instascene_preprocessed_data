from __future__ import annotations

from ins_scene_15k_roots import SCANNETPPV2_ROOT
from scene_path_discover_ins_scene_15k import discover_scannetpp_v2_extracted
from scene_path_sample_job import SampleJobConfig, run_sample_job

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2 "
        "and write a scene-path manifest (pair images/*.jpg with refined_ins_ids/*.png by stem)."
    ),
    default_root=SCANNETPPV2_ROOT,
    discover=discover_scannetpp_v2_extracted,
    pair_by="stem",
    empty_candidates_template="error: no valid scenes under: {root}",
    root_not_dir_template="error: not a directory: {root}",
)


def main() -> int:
    return run_sample_job(_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
