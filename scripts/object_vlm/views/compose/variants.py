from __future__ import annotations

from typing import Any

import numpy as np

from ...config import PipelineConfig
from . import geometry


def render_origin(image_rgb: np.ndarray, *_: object) -> np.ndarray:
    return image_rgb.copy()


def render_mask_bw(mask: np.ndarray, *, foreground: int = 255, background: int = 0, invert: bool = False) -> np.ndarray:
    m = mask.astype(bool)
    if invert:
        m = ~m
    fg = np.uint8(np.clip(foreground, 0, 255))
    bg = np.uint8(np.clip(background, 0, 255))
    out = np.where(m, fg, bg).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=2)


def render_highlight_outside_dark(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    outside_factor: float = 0.35,
) -> np.ndarray:
    out = image_rgb.copy()
    factor = float(outside_factor)
    factor = max(0.0, min(1.0, factor))
    outside = ~mask.astype(bool)
    out[outside] = np.clip(out[outside].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def render_variant(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    variant: str,
    config: PipelineConfig,
    spec: dict[str, Any],
) -> np.ndarray:
    if variant == "origin":
        return render_origin(image_rgb)
    if variant == "mask_bw":
        opts = spec.get("mask_bw") if isinstance(spec.get("mask_bw"), dict) else {}
        return render_mask_bw(
            mask,
            foreground=int(opts.get("foreground", 255)),
            background=int(opts.get("background", 0)),
            invert=bool(opts.get("invert", False)),
        )
    if variant == "highlight_outside_dark":
        opts = spec.get("highlight_outside_dark") if isinstance(spec.get("highlight_outside_dark"), dict) else {}
        return render_highlight_outside_dark(
            image_rgb,
            mask,
            outside_factor=float(opts.get("outside_factor", 0.35)),
        )
    if variant == "overlay":
        return geometry.draw_overlay(image_rgb, mask, bbox_xyxy, config.view_render.overlay_style)
    if variant == "crop":
        return geometry.make_crop(image_rgb, mask, bbox_xyxy, config.view_render.crop_padding_ratio)
    raise ValueError(f"Unsupported panel variant: {variant}")
