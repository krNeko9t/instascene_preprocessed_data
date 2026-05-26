from __future__ import annotations

from typing import Literal

IdMapSource = Literal["npy", "png", "sam2_json"]
PairingStrategy = Literal["stem", "infinigen"]
OverlayStyle = str

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

__all__ = [
    "IdMapSource",
    "OverlayStyle",
    "PairingStrategy",
    "SUPPORTED_IMAGE_SUFFIXES",
]
