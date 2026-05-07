from __future__ import annotations

from ins_scene_15k_roots import RE10K_ROOT
from scene_path_discover_ins_scene_15k import discover_re10k_extracted
from scene_path_sample_job import SampleJobConfig, run_sample_job

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_re10k_extracted/processed_re10k and write a scene-path manifest. "
        "Entries omit id_map_dir (rgb-only); batch stats will count views and set n_objects=0."
    ),
    default_root=RE10K_ROOT,
    discover=discover_re10k_extracted,
    pair_by=None,
    empty_candidates_template="error: no scenes with rgb+cam under: {root}",
    root_not_dir_template="error: not a directory: {root}",
)


def main() -> int:
    return run_sample_job(_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
