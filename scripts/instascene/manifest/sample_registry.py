from __future__ import annotations

import sys
from typing import Literal, cast

from instascene.manifest.roots import INFINIGEN_ROOT, RE10K_ROOT, SCANNETPPV2_ROOT
from instascene.manifest.sample import SampleJobConfig
from instascene.scene.discovery.ins_scene_15k import (
    discover_infinigen_extracted,
    discover_re10k_extracted,
    discover_scannetpp_v2_extracted,
)

__all__ = [
    "SAMPLE_JOB_CONFIGS",
    "SampleDatasetId",
    "get_sample_job_config",
]

SampleDatasetId = Literal["infinigen", "re10k", "scannetpp_v2"]

SAMPLE_JOB_CONFIGS: dict[SampleDatasetId, SampleJobConfig] = {
    "infinigen": SampleJobConfig(
        description=(
            "Sample scenes under InsScene-15K/processed_infinigen_extracted "
            "and write a scene-path manifest JSON."
        ),
        default_root=INFINIGEN_ROOT,
        discover=discover_infinigen_extracted,
        pair_by="infinigen",
        empty_candidates_template="error: no valid scenes found under: {root}",
        root_not_dir_template="error: dataset root is not a directory: {root}",
    ),
    "re10k": SampleJobConfig(
        description=(
            "Sample scenes under InsScene-15K/processed_re10k_extracted/processed_re10k "
            "and write a scene-path manifest JSON."
        ),
        default_root=RE10K_ROOT,
        discover=discover_re10k_extracted,
        pair_by=None,
        empty_candidates_template="error: no valid scenes found under: {root}",
    ),
    "scannetpp_v2": SampleJobConfig(
        description=(
            "Sample scenes under InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2 "
            "and write a scene-path manifest JSON."
        ),
        default_root=SCANNETPPV2_ROOT,
        discover=discover_scannetpp_v2_extracted,
        pair_by="stem",
        empty_candidates_template="error: no valid scenes found under: {root}",
    ),
}


def get_sample_job_config(dataset: str) -> SampleJobConfig:
    if dataset not in SAMPLE_JOB_CONFIGS:
        allowed = ", ".join(sorted(SAMPLE_JOB_CONFIGS))
        print(
            f"error: unknown dataset {dataset!r}; choose one of: {allowed}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return SAMPLE_JOB_CONFIGS[cast(SampleDatasetId, dataset)]
