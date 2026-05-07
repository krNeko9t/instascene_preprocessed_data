from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from config import PipelineConfig
from view_selector import ViewInfo


def _normalize_overlay_styles(style: str) -> set[str]:
    allowed = {"contour", "bbox", "semitransparent", "all"}
    tokens = [token.strip() for token in style.split(",") if token.strip()]
    if not tokens:
        return {"bbox"}
    invalid = sorted({token for token in tokens if token not in allowed})
    if invalid:
        raise ValueError(
            f"Invalid overlay style(s): {', '.join(invalid)}; allowed: contour,bbox,semitransparent,all"
        )
    if "all" in tokens:
        return {"contour", "bbox", "semitransparent"}
    return set(tokens)


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


def _read_image_rgb(image_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _encode_rgb_to_jpeg_base64(image_rgb: np.ndarray, quality: int = 90) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


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
    styles = _normalize_overlay_styles(style)

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


def _render_origin(image_rgb: np.ndarray, *_: object) -> np.ndarray:
    return image_rgb.copy()


def _render_mask_bw(mask: np.ndarray, *, foreground: int = 255, background: int = 0, invert: bool = False) -> np.ndarray:
    m = mask.astype(bool)
    if invert:
        m = ~m
    fg = np.uint8(np.clip(foreground, 0, 255))
    bg = np.uint8(np.clip(background, 0, 255))
    out = np.where(m, fg, bg).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=2)


def _render_highlight_outside_dark(image_rgb: np.ndarray, mask: np.ndarray, *, outside_factor: float = 0.35) -> np.ndarray:
    out = image_rgb.copy()
    factor = float(outside_factor)
    factor = max(0.0, min(1.0, factor))
    outside = ~mask.astype(bool)
    out[outside] = np.clip(out[outside].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def _render_variant(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    variant: str,
    config: PipelineConfig,
    spec: dict[str, Any],
) -> np.ndarray:
    if variant == "origin":
        return _render_origin(image_rgb)
    if variant == "mask_bw":
        opts = spec.get("mask_bw") if isinstance(spec.get("mask_bw"), dict) else {}
        return _render_mask_bw(
            mask,
            foreground=int(opts.get("foreground", 255)),
            background=int(opts.get("background", 0)),
            invert=bool(opts.get("invert", False)),
        )
    if variant == "highlight_outside_dark":
        opts = spec.get("highlight_outside_dark") if isinstance(spec.get("highlight_outside_dark"), dict) else {}
        return _render_highlight_outside_dark(
            image_rgb,
            mask,
            outside_factor=float(opts.get("outside_factor", 0.35)),
        )
    if variant == "overlay":
        return _draw_overlay(image_rgb, mask, bbox_xyxy, config.view_render.overlay_style)
    if variant == "crop":
        return _make_crop(image_rgb, mask, bbox_xyxy, config.view_render.crop_padding_ratio)
    raise ValueError(f"Unsupported panel variant: {variant}")


def _default_title(variant: str) -> str:
    aliases = {
        "origin": "origin",
        "mask_bw": "mask",
        "highlight_outside_dark": "highlight",
        "overlay": "overlay",
        "crop": "crop",
    }
    return aliases.get(variant, variant)


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


def _add_panel_title(panel_rgb: np.ndarray, title: str, title_spec: dict[str, Any]) -> np.ndarray:
    bar_h = int(title_spec.get("height", 28))
    bar_h = max(16, min(96, bar_h))
    bg_value = int(title_spec.get("bg_value", 245))
    fg_value = int(title_spec.get("fg_value", 20))
    bg_value = max(0, min(255, bg_value))
    fg_value = max(0, min(255, fg_value))

    h, w = panel_rgb.shape[:2]
    bar = np.full((bar_h, w, 3), bg_value, dtype=np.uint8)
    out = np.vstack([bar, panel_rgb])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.2, bar_h / 36.0))
    thickness = max(1, int(round(bar_h / 18.0)))
    text = title.strip() or "panel"
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    text_x = max(6, (w - text_w) // 2)
    text_y = max(text_h + 3, (bar_h + text_h) // 2 - baseline // 2)
    cv2.putText(out, text, (text_x, text_y), font, font_scale, (fg_value, fg_value, fg_value), thickness, cv2.LINE_AA)
    return out


def _compose_panels(rendered_panels: list[np.ndarray], layout_spec: dict[str, Any], fill_value: int) -> np.ndarray:
    if not rendered_panels:
        raise ValueError("Cannot compose empty panels")

    layout_type = str(layout_spec.get("type", "horizontal")).strip().lower()
    if layout_type not in {"horizontal", "vertical", "grid"}:
        raise ValueError("layout.type must be one of: horizontal, vertical, grid")

    if layout_type == "horizontal":
        target_h = max(img.shape[0] for img in rendered_panels)
        padded = [_pad_to_size_center(img, target_h, img.shape[1], fill_value) for img in rendered_panels]
        return np.hstack(padded)

    if layout_type == "vertical":
        target_w = max(img.shape[1] for img in rendered_panels)
        padded = [_pad_to_size_center(img, img.shape[0], target_w, fill_value) for img in rendered_panels]
        return np.vstack(padded)

    # grid
    n = len(rendered_panels)
    cols = layout_spec.get("cols")
    rows = layout_spec.get("rows")
    if cols is None and rows is None:
        cols = 2
    if cols is not None:
        cols = int(cols)
        if cols <= 0:
            raise ValueError("layout.cols must be positive")
    if rows is not None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("layout.rows must be positive")
    if cols is None:
        cols = int(np.ceil(n / rows))
    if rows is None:
        rows = int(np.ceil(n / cols))
    if rows * cols < n:
        raise ValueError("layout.rows * layout.cols is smaller than number of panels")

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


def _parse_compose_spec(raw_spec: str) -> dict[str, Any]:
    try:
        spec = json.loads(raw_spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --view-compose-spec JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("--view-compose-spec must be a JSON object")
    panels = spec.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("--view-compose-spec.panels must be a non-empty list")
    layout = spec.get("layout", {"type": "horizontal"})
    if not isinstance(layout, dict):
        raise ValueError("--view-compose-spec.layout must be an object")
    titles = spec.get("titles", {"enabled": False})
    if not isinstance(titles, dict):
        raise ValueError("--view-compose-spec.titles must be an object")
    align = spec.get("align", {"mode": "pad", "fill_value": 255})
    if not isinstance(align, dict):
        raise ValueError("--view-compose-spec.align must be an object")
    mode = str(align.get("mode", "pad")).strip().lower()
    if mode != "pad":
        raise ValueError("Only align.mode='pad' is supported currently")
    fill_value = int(align.get("fill_value", 255))
    fill_value = max(0, min(255, fill_value))

    spec["layout"] = layout
    spec["titles"] = titles
    spec["align"] = {"mode": mode, "fill_value": fill_value}
    return spec


def prepare_image_batches(views: List[ViewInfo], config: PipelineConfig) -> List[ImageBatch]:
    if not views:
        return []

    spec = _parse_compose_spec(config.view_render.view_compose_spec)
    panels = spec.get("panels")
    assert isinstance(panels, list)
    title_spec = spec["titles"]
    layout_spec = spec["layout"]
    fill_value = int(spec["align"]["fill_value"])
    titles_enabled = bool(title_spec.get("enabled", False))

    images: List[PreparedImage] = []
    for view in views:
        image_rgb = _read_image_rgb(view.image_path)
        rendered_panels: list[dict[str, Any]] = []
        for p in panels:
            if not isinstance(p, dict):
                raise ValueError("--view-compose-spec.panels items must be objects")
            variant = p.get("variant")
            if not isinstance(variant, str) or not variant.strip():
                raise ValueError("Each panel must have a non-empty string field: variant")
            title = p.get("title")
            if title is not None and not isinstance(title, str):
                raise ValueError("panel.title must be a string when provided")

            panel_rgb = _render_variant(
                image_rgb,
                view.mask,
                view.bbox_xyxy,
                variant=variant.strip(),
                config=config,
                spec=spec,
            )
            panel_title = title if isinstance(title, str) else _default_title(variant.strip())
            rendered_panels.append({"variant": variant.strip(), "title": panel_title, "image_rgb": panel_rgb})

        if titles_enabled:
            max_h = max(rp["image_rgb"].shape[0] for rp in rendered_panels)
            max_w = max(rp["image_rgb"].shape[1] for rp in rendered_panels)
            for rp in rendered_panels:
                padded = _pad_to_size_center(rp["image_rgb"], max_h, max_w, fill_value)
                rp["image_rgb"] = _add_panel_title(padded, rp["title"], title_spec)

        composed_rgb = _compose_panels([rp["image_rgb"] for rp in rendered_panels], layout_spec, fill_value)
        panel_meta = [
            {"variant": rp["variant"], "title": (rp["title"] if isinstance(rp["title"], str) else _default_title(rp["variant"]))}
            for rp in rendered_panels
        ]
        images.append(
            PreparedImage(
                view_name=view.view_name,
                variant="compose",
                image_base64=_encode_rgb_to_jpeg_base64(composed_rgb),
                meta={
                    "compose": {
                        "panels": panel_meta,
                        "layout": layout_spec,
                        "titles": title_spec,
                    }
                },
            )
        )

    return [ImageBatch(images=images, mode="compose", meta={"compose_spec": spec})]
