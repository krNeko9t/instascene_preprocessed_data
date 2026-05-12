from __future__ import annotations

from pathlib import Path
from typing import List


def list_scenes(data_root: Path, dataset: str) -> List[str]:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_dir}")
    scenes = [p.name for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    return scenes


def list_datasets(data_root: Path) -> List[str]:
    datasets: List[str] = []
    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        has_scene = False
        for scene_dir in ds_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "images").exists() and (scene_dir / "sam").exists():
                has_scene = True
                break
        if has_scene:
            datasets.append(ds_dir.name)
    return datasets
