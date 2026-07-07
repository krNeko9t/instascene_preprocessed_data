from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

from instascene.manifest.roots import (
    DEFAULT_3DOVS_ROOT,
    DEFAULT_LERF_ROOT,
    DEFAULT_ZIPNERF_ROOT,
    INFINIGEN_ROOT,
    RE10K_ROOT,
    SCANNETPPV2_ROOT,
)
from instascene.manifest.sample import SampleJobConfig
from instascene.scene.discovery.ins_scene_15k import (
    discover_infinigen_scenes,
    discover_re10k_scenes,
    discover_scannetpp_v2_scenes,
)
from instascene.scene.discovery.instascene import discover_instascene_scenes
from instascene.scene.ref import SceneRef
from instascene.types import IdMapSource, PairingStrategy

__all__ = [
    "DATASET_SPECS",
    "DatasetSpec",
    "SAMPLE_JOB_CONFIGS",
    "SampleDatasetId",
    "get_dataset_spec",
    "get_sample_job_config",
]

SampleDatasetId = Literal["3dovs", "lerf", "zipnerf", "infinigen", "re10k", "scannetpp_v2"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    description: str
    default_root: Path
    discover: Callable[[Path], list[SceneRef]]
    id_map_source: IdMapSource
    pair_by: PairingStrategy


DATASET_SPECS: dict[SampleDatasetId, DatasetSpec] = {
    "3dovs": DatasetSpec(
        dataset_id="3dovs",
        description="3DOVS scenes (images/ + id_maps/ or sam/mask/).",
        default_root=DEFAULT_3DOVS_ROOT,
        discover=discover_instascene_scenes,
        id_map_source="npy",
        pair_by="stem",
    ),
    "lerf": DatasetSpec(
        dataset_id="lerf",
        description="LERF scenes (images/ + sam/mask/).",
        default_root=DEFAULT_LERF_ROOT,
        discover=discover_instascene_scenes,
        id_map_source="png",
        pair_by="stem",
    ),
    "zipnerf": DatasetSpec(
        dataset_id="zipnerf",
        description="Zip-NeRF scenes (images/ + sam/mask/).",
        default_root=DEFAULT_ZIPNERF_ROOT,
        discover=discover_instascene_scenes,
        id_map_source="png",
        pair_by="stem",
    ),
    "infinigen": DatasetSpec(
        dataset_id="infinigen",
        description=(
            "Sample Infinigen scenes from InsScene-15K and write a scene selection lockfile."
        ),
        default_root=INFINIGEN_ROOT,
        discover=discover_infinigen_scenes,
        id_map_source="png",
        pair_by="infinigen",
    ),
    "re10k": DatasetSpec(
        dataset_id="re10k",
        description=(
            "Sample RE10K scenes from InsScene-15K and write a scene selection lockfile."
        ),
        default_root=RE10K_ROOT,
        discover=discover_re10k_scenes,
        id_map_source="sam2_json",
        pair_by="stem",
    ),
    "scannetpp_v2": DatasetSpec(
        dataset_id="scannetpp_v2",
        description=(
            "Sample ScanNet++ v2 scenes from InsScene-15K and write a scene selection lockfile."
        ),
        default_root=SCANNETPPV2_ROOT,
        discover=discover_scannetpp_v2_scenes,
        id_map_source="png",
        pair_by="stem",
    ),
}

SAMPLE_JOB_CONFIGS: dict[SampleDatasetId, SampleJobConfig] = {
    dataset_id: SampleJobConfig(
        dataset_id=spec.dataset_id,
        description=spec.description,
        default_root=spec.default_root,
        discover=spec.discover,
        id_map_source=spec.id_map_source,
        pair_by=spec.pair_by,
        empty_candidates_template="error: no valid scenes found under: {root}",
        root_not_dir_template=(
            "error: dataset root is not a directory: {root}"
            if dataset_id == "infinigen"
            else "error: not a directory: {root}"
        ),
    )
    for dataset_id, spec in DATASET_SPECS.items()
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASET_SPECS:
        allowed = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"unknown dataset_id {dataset_id!r}; choose one of: {allowed}")
    return DATASET_SPECS[cast(SampleDatasetId, dataset_id)]


def get_sample_job_config(dataset: str) -> SampleJobConfig:
    if dataset not in SAMPLE_JOB_CONFIGS:
        allowed = ", ".join(sorted(SAMPLE_JOB_CONFIGS))
        print(
            f"error: unknown dataset {dataset!r}; choose one of: {allowed}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return SAMPLE_JOB_CONFIGS[cast(SampleDatasetId, dataset)]
