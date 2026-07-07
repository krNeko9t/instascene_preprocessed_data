"""Shared roots under the InsScene-15K release tree (multiple processed extracts)."""

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
LOCAL_DATA_ROOT = Path(__file__).resolve().parents[3]
LOCAL_3DOVS_ROOT = LOCAL_DATA_ROOT / "3dovs"
LOCAL_LERF_ROOT = LOCAL_DATA_ROOT / "lerf"
LOCAL_ZIPNERF_ROOT = LOCAL_DATA_ROOT / "zipnerf"


def path_for_manifest_json(path: Path, *, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    try:
        return str(path.relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)
