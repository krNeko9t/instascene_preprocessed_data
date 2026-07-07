"""Shared dataset root paths."""

from __future__ import annotations

from pathlib import Path

INS_SCENE_15K_ROOT = Path(
    "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/dataset/InsScene-15K"
)

INFINIGEN_ROOT = INS_SCENE_15K_ROOT / "processed_infinigen_extracted"
RE10K_ROOT = INS_SCENE_15K_ROOT / "processed_re10k_extracted" / "processed_re10k"
SCANNETPPV2_ROOT = (
    INS_SCENE_15K_ROOT / "processed_scannetpp_v2_extracted" / "processed_scannetpp_v2"
)

# Repo root: scripts/instascene/manifest/roots.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_3DOVS_ROOT = REPO_ROOT / "3dovs"
DEFAULT_LERF_ROOT = REPO_ROOT / "lerf"
DEFAULT_ZIPNERF_ROOT = REPO_ROOT / "zipnerf"
