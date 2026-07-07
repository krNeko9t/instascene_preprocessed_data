from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SceneRef"]


@dataclass(slots=True, frozen=True)
class SceneRef:
    """Stable scene identity within one dataset root."""

    scene_id: str
