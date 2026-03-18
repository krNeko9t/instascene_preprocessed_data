from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config import PipelineConfig
from view_selector import ViewInfo


@dataclass(slots=True)
class PreparedImage:
    view_name: str
    variant: str
    image_base64: str


@dataclass(slots=True)
class ImageBatch:
    images: List[PreparedImage]
    mode: str


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

    if style in {"semitransparent", "all"}:
        color = np.array([40, 220, 70], dtype=np.uint8)
        alpha = 0.35
        out[mask] = (out[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    mask_u8 = (mask.astype(np.uint8) * 255).copy()
    if style in {"contour", "all"}:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 50, 50), 2)

    if style in {"bbox", "all"}:
        cv2.rectangle(out, (x1, y1), (x2, y2), (50, 120, 255), 2)
    return out


def prepare_image_batches(views: List[ViewInfo], config: PipelineConfig) -> List[ImageBatch]:
    if not views:
        return []

    if config.image_mode == "crop_only":
        images: List[PreparedImage] = []
        for view in views:
            image_rgb = _read_image_rgb(view.image_path)
            crop = _make_crop(image_rgb, view.mask, view.bbox_xyxy, config.crop_padding_ratio)
            images.append(
                PreparedImage(
                    view_name=view.view_name,
                    variant="crop",
                    image_base64=_encode_rgb_to_jpeg_base64(crop),
                )
            )
        return [ImageBatch(images=images, mode="crop_only")]

    if config.image_mode == "overlay_only":
        images = []
        for view in views:
            image_rgb = _read_image_rgb(view.image_path)
            overlay = _draw_overlay(image_rgb, view.mask, view.bbox_xyxy, config.overlay_style)
            images.append(
                PreparedImage(
                    view_name=view.view_name,
                    variant="overlay",
                    image_base64=_encode_rgb_to_jpeg_base64(overlay),
                )
            )
        return [ImageBatch(images=images, mode="overlay_only")]

    if config.image_mode == "pair":
        images = []
        for view in views:
            image_rgb = _read_image_rgb(view.image_path)
            overlay = _draw_overlay(image_rgb, view.mask, view.bbox_xyxy, config.overlay_style)
            crop = _make_crop(image_rgb, view.mask, view.bbox_xyxy, config.crop_padding_ratio)
            images.append(
                PreparedImage(
                    view_name=view.view_name,
                    variant="overlay",
                    image_base64=_encode_rgb_to_jpeg_base64(overlay),
                )
            )
            images.append(
                PreparedImage(
                    view_name=view.view_name,
                    variant="crop",
                    image_base64=_encode_rgb_to_jpeg_base64(crop),
                )
            )
        return [ImageBatch(images=images, mode="pair")]

    if config.image_mode == "per_view":
        batches: List[ImageBatch] = []
        for view in views:
            image_rgb = _read_image_rgb(view.image_path)
            overlay = _draw_overlay(image_rgb, view.mask, view.bbox_xyxy, config.overlay_style)
            crop = _make_crop(image_rgb, view.mask, view.bbox_xyxy, config.crop_padding_ratio)
            batch_images = [
                PreparedImage(
                    view_name=view.view_name,
                    variant="overlay",
                    image_base64=_encode_rgb_to_jpeg_base64(overlay),
                ),
                PreparedImage(
                    view_name=view.view_name,
                    variant="crop",
                    image_base64=_encode_rgb_to_jpeg_base64(crop),
                ),
            ]
            batches.append(ImageBatch(images=batch_images, mode="per_view"))
        return batches

    raise ValueError(f"Unsupported image mode: {config.image_mode}")
