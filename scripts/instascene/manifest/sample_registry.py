from __future__ import annotations

import sys
from typing import Literal, cast

from instascene.manifest.roots import (
    INFINIGEN_ROOT,
    LOCAL_3DOVS_ROOT,
    LOCAL_LERF_ROOT,
    LOCAL_ZIPNERF_ROOT,
    RE10K_ROOT,
    SCANNETPPV2_ROOT,
)
from instascene.manifest.sample import SampleJobConfig
from instascene.scene.discovery.ins_scene_15k import (
    discover_infinigen_extracted,
    discover_re10k_extracted,
    discover_scannetpp_v2_extracted,
)
from instascene.scene.discovery.local import discover_local_extracted

__all__ = [
    "SAMPLE_JOB_CONFIGS",
    "SampleDatasetId",
    "get_sample_job_config",
]

SampleDatasetId = Literal["3dovs", "lerf", "zipnerf", "infinigen", "re10k", "scannetpp_v2"]

SAMPLE_JOB_CONFIGS: dict[SampleDatasetId, SampleJobConfig] = {
    "3dovs": SampleJobConfig(
        dataset_id="3dovs",
        description="Local 3DOVS scenes (images/ + id_maps/ or sam/mask/).",
        default_root=LOCAL_3DOVS_ROOT,
        discover=discover_local_extracted,
        id_map_source="npy",
        pair_by="stem",
        empty_candidates_template="error: no valid scenes found under: {root}",
    ),
    "lerf": SampleJobConfig(
        dataset_id="lerf",
        description="Local LERF scenes (images/ + sam/mask/).",
        default_root=LOCAL_LERF_ROOT,
        discover=discover_local_extracted,
        id_map_source="png",
        pair_by="stem",
        empty_candidates_template="error: no valid scenes found under: {root}",
    ),
    "zipnerf": SampleJobConfig(
        dataset_id="zipnerf",
        description="Local Zip-NeRF scenes (images/ + sam/mask/).",
        default_root=LOCAL_ZIPNERF_ROOT,
        discover=discover_local_extracted,
        id_map_source="png",
        pair_by="stem",
        empty_candidates_template="error: no valid scenes found under: {root}",
    ),
    "infinigen": SampleJobConfig(
        dataset_id="infinigen",
        description=(
            "Sample scenes under InsScene-15K/processed_infinigen_extracted "
            "and write a scene-path manifest JSON."
        ),
        default_root=INFINIGEN_ROOT,
        discover=discover_infinigen_extracted,
        id_map_source="png",
        pair_by="infinigen",
        empty_candidates_template="error: no valid scenes found under: {root}",
        root_not_dir_template="error: dataset root is not a directory: {root}",
    ),
    "re10k": SampleJobConfig(
        dataset_id="re10k",
        description=(
            "Sample scenes under InsScene-15K/processed_re10k_extracted/processed_re10k "
            "and write a scene-path manifest JSON."
        ),
        default_root=RE10K_ROOT,
        discover=discover_re10k_extracted,
        id_map_source="sam2_json",
        pair_by=None,
        empty_candidates_template="error: no valid scenes found under: {root}",
    ),
    "scannetpp_v2": SampleJobConfig(
        dataset_id="scannetpp_v2",
        description=(
            "Sample scenes under InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2 "
            "and write a scene-path manifest JSON."
        ),
        default_root=SCANNETPPV2_ROOT,
        discover=discover_scannetpp_v2_extracted,
        id_map_source="png",
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
