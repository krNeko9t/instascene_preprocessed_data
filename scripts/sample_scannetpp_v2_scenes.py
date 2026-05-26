from __future__ import annotations

from instascene.manifest.roots import SCANNETPPV2_ROOT
from instascene.manifest.sample import SampleJobConfig, run_sample_job
from instascene.scene.discovery.ins_scene_15k import discover_scannetpp_v2_extracted

_CONFIG = SampleJobConfig(
    description=(
        "Sample scenes under InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2 "
        "and write a scene-path manifest JSON."
    ),
    default_root=SCANNETPPV2_ROOT,
    discover=discover_scannetpp_v2_extracted,
    pair_by="stem",
    empty_candidates_template="error: no valid scenes found under: {root}",
)


def main() -> int:
    return run_sample_job(_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
