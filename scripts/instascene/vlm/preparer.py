from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List

import cv2
import numpy as np

from instascene.imaging import encode_rgb_to_jpeg_b64, read_image_rgb
from instascene.types import normalize_overlay_styles
from instascene.vlm.compose import (
    ComposeSpec,
    LayoutSpec,
    TitleSpec,
    parse_compose_spec,
)
from instascene.vlm.config import PipelineConfig
from instascene.vlm.selector import ViewInfo


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


def _expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(img_w - 1, x2 + pad_x)
    ey2 = min(img_h - 1, y2 + pad_y)
    return ex1, ey1, ex2, ey2


def _make_crop(image_rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], padding_ratio: float) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(*bbox_xyxy, w, h, padding_ratio)
    cropped_img = image_rgb[y1 : y2 + 1, x1 : x2 + 1].copy()
    cropped_mask = mask[y1 : y2 + 1, x1 : x2 + 1]

    white_bg = np.full_like(cropped_img, 255)
    out = np.where(cropped_mask[..., None], cropped_img, white_bg)
    return out


def _draw_overlay(image_rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], style: str) -> np.ndarray:
    out = image_rgb.copy()
    x1, y1, x2, y2 = bbox_xyxy
    styles = normalize_overlay_styles(style)

    if "semitransparent" in styles:
        color = np.array([40, 220, 70], dtype=np.uint8)
        alpha = 0.35
        out[mask] = (out[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    mask_u8 = (mask.astype(np.uint8) * 255).copy()
    if "contour" in styles:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 50, 50), 2)

    if "bbox" in styles:
        cv2.rectangle(out, (x1, y1), (x2, y2), (50, 120, 255), 2)
    return out


def _render_mask_bw(mask: np.ndarray, spec: ComposeSpec) -> np.ndarray:
    m = mask.astype(bool)
    if spec.mask_bw.invert:
        m = ~m
    out = np.where(m, np.uint8(spec.mask_bw.foreground), np.uint8(spec.mask_bw.background)).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=2)


def _render_highlight_outside_dark(image_rgb: np.ndarray, mask: np.ndarray, spec: ComposeSpec) -> np.ndarray:
    out = image_rgb.copy()
    outside = ~mask.astype(bool)
    out[outside] = np.clip(out[outside].astype(np.float32) * spec.highlight_outside_dark.outside_factor, 0, 255).astype(np.uint8)
    return out


def _render_variant(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    variant: str,
    config: PipelineConfig,
    spec: ComposeSpec,
) -> np.ndarray:
    if variant == "origin":
        return image_rgb.copy()
    if variant == "mask_bw":
        return _render_mask_bw(mask, spec)
    if variant == "highlight_outside_dark":
        return _render_highlight_outside_dark(image_rgb, mask, spec)
    if variant == "overlay":
        return _draw_overlay(image_rgb, mask, bbox_xyxy, config.view_render.overlay_style)
    if variant == "crop":
        return _make_crop(image_rgb, mask, bbox_xyxy, config.view_render.crop_padding_ratio)
    raise ValueError(f"Unsupported panel variant: {variant}")


def _pad_to_size_center(image_rgb: np.ndarray, target_h: int, target_w: int, fill_value: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h > target_h or w > target_w:
        raise ValueError(f"Target size too small: current=({h},{w}), target=({target_h},{target_w})")
    if h == target_h and w == target_w:
        return image_rgb
    pad_t = (target_h - h) // 2
    pad_b = target_h - h - pad_t
    pad_l = (target_w - w) // 2
    pad_r = target_w - w - pad_l
    return np.pad(
        image_rgb,
        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )


def _add_panel_title(panel_rgb: np.ndarray, title: str, titles: TitleSpec) -> np.ndarray:
    bar_h = titles.height
    h, w = panel_rgb.shape[:2]
    bar = np.full((bar_h, w, 3), titles.bg_value, dtype=np.uint8)
    out = np.vstack([bar, panel_rgb])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.2, bar_h / 36.0))
    thickness = max(1, int(round(bar_h / 18.0)))
    text = title.strip() or "panel"
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    text_x = max(6, (w - text_w) // 2)
    text_y = max(text_h + 3, (bar_h + text_h) // 2 - baseline // 2)
    fg = titles.fg_value
    cv2.putText(out, text, (text_x, text_y), font, font_scale, (fg, fg, fg), thickness, cv2.LINE_AA)
    return out


def _compose_panels(rendered_panels: list[np.ndarray], layout: LayoutSpec, fill_value: int) -> np.ndarray:
    if not rendered_panels:
        raise ValueError("Cannot compose empty panels")

    if layout.type == "horizontal":
        target_h = max(img.shape[0] for img in rendered_panels)
        padded = [_pad_to_size_center(img, target_h, img.shape[1], fill_value) for img in rendered_panels]
        return np.hstack(padded)

    if layout.type == "vertical":
        target_w = max(img.shape[1] for img in rendered_panels)
        padded = [_pad_to_size_center(img, img.shape[0], target_w, fill_value) for img in rendered_panels]
        return np.vstack(padded)

    # grid: rows/cols resolved and validated at parse time
    rows, cols = layout.rows, layout.cols
    assert rows is not None and cols is not None
    tile_h = max(img.shape[0] for img in rendered_panels)
    tile_w = max(img.shape[1] for img in rendered_panels)
    blank = np.full((tile_h, tile_w, 3), fill_value, dtype=np.uint8)
    tiles = [_pad_to_size_center(img, tile_h, tile_w, fill_value) for img in rendered_panels]
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    row_images = []
    for r in range(rows):
        start = r * cols
        row_images.append(np.hstack(tiles[start : start + cols]))
    return np.vstack(row_images)


def prepare_image_batches(views: List[ViewInfo], config: PipelineConfig) -> List[ImageBatch]:
    if not views:
        return []

    spec = parse_compose_spec(config.view_render.view_compose_spec)

    images: List[PreparedImage] = []
    for view in views:
        image_rgb = read_image_rgb(view.image_path)
        rendered = [
            _render_variant(
                image_rgb,
                view.mask,
                view.bbox_xyxy,
                variant=panel.variant,
                config=config,
                spec=spec,
            )
            for panel in spec.panels
        ]

        if spec.titles.enabled:
            max_h = max(img.shape[0] for img in rendered)
            max_w = max(img.shape[1] for img in rendered)
            rendered = [
                _add_panel_title(
                    _pad_to_size_center(img, max_h, max_w, spec.fill_value),
                    panel.title,
                    spec.titles,
                )
                for img, panel in zip(rendered, spec.panels)
            ]

        composed_rgb = _compose_panels(rendered, spec.layout, spec.fill_value)
        images.append(
            PreparedImage(
                view_name=view.view_name,
                variant="compose",
                image_base64=encode_rgb_to_jpeg_b64(composed_rgb),
                meta={
                    "compose": {
                        "panels": [{"variant": p.variant, "title": p.title} for p in spec.panels],
                        "layout": asdict(spec.layout),
                        "titles": asdict(spec.titles),
                    }
                },
            )
        )

    return [ImageBatch(images=images, mode="compose", meta={"compose_spec": spec.as_json_dict()})]
