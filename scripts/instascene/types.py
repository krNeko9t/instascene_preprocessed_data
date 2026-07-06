from __future__ import annotations

from typing import Literal

IdMapSource = Literal["npy", "png", "sam2_json"]
PairingStrategy = Literal["stem", "infinigen"]
OverlayStyle = str

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

ALLOWED_OVERLAY_STYLES = {"contour", "bbox", "semitransparent", "all"}


def normalize_overlay_styles(style: OverlayStyle) -> set[str]:
    """Parse a comma-separated overlay style string into the effective style set."""
    tokens = [token.strip() for token in style.split(",") if token.strip()]
    if not tokens:
        raise ValueError("overlay-style cannot be empty")
    invalid = sorted({token for token in tokens if token not in ALLOWED_OVERLAY_STYLES})
    if invalid:
        raise ValueError(
            f"Invalid overlay style(s): {', '.join(invalid)}; allowed: contour,bbox,semitransparent,all"
        )
    if "all" in tokens:
        return {"contour", "bbox", "semitransparent"}
    return set(tokens)


__all__ = [
    "ALLOWED_OVERLAY_STYLES",
    "IdMapSource",
    "OverlayStyle",
    "PairingStrategy",
    "SUPPORTED_IMAGE_SUFFIXES",
    "normalize_overlay_styles",
]
