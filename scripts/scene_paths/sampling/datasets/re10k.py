from __future__ import annotations

from ...paths.discover import discover_re10k_extracted
from ...paths.roots import RE10K_ROOT
from ..job import SampleJobConfig, run_sample_job

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_re10k_extracted/processed_re10k and write a scene-path manifest. "
        "Includes optional id_map_json when sam2_results/<scene>/auto_masks.json exists (RLE masklet)."
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
