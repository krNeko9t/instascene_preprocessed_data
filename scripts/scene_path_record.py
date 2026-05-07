from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ScenePathRecord:
    """One scene worth of absolute paths before serializing to manifest JSON."""

    partition: str
    scene_name: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path | None
