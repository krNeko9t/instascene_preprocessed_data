from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass(slots=True)
class PreparedImage:
    view_name: str
    variant: str
    image_base64: str
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class ImageBatch:
    images: List[PreparedImage]
    mode: str
    meta: dict[str, Any] | None = None
