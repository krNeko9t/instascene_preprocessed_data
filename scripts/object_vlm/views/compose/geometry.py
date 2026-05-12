from __future__ import annotations

import cv2
import numpy as np


def normalize_overlay_styles(style: str) -> set[str]:
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


def expand_bbox(
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


def make_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    padding_ratio: float,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = expand_bbox(*bbox_xyxy, w, h, padding_ratio)
    cropped_img = image_rgb[y1 : y2 + 1, x1 : x2 + 1].copy()
    cropped_mask = mask[y1 : y2 + 1, x1 : x2 + 1]

    white_bg = np.full_like(cropped_img, 255)
    out = np.where(cropped_mask[..., None], cropped_img, white_bg)
    return out


def draw_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    style: str,
) -> np.ndarray:
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


def pad_to_size_center(image_rgb: np.ndarray, target_h: int, target_w: int, fill_value: int) -> np.ndarray:
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


def add_panel_title(panel_rgb: np.ndarray, title: str, title_spec: dict) -> np.ndarray:
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
