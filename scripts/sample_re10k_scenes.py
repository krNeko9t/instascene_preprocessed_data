from __future__ import annotations

from instascene.manifest.roots import RE10K_ROOT
from instascene.manifest.sample import SampleJobConfig, run_sample_job
from instascene.scene.discovery.ins_scene_15k import discover_re10k_extracted

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_re10k_extracted/processed_re10k "
        "and write a scene-path manifest JSON."
    ),
    default_root=RE10K_ROOT,
    discover=discover_re10k_extracted,
    pair_by=None,
    empty_candidates_template="error: no valid scenes found under: {root}",
)


def main() -> int:
    return run_sample_job(_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
