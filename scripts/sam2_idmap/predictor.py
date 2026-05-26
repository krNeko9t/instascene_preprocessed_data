from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


@dataclass
class Sam2IdMapConfig:
    checkpoint: str | None = None
    model_cfg: str | None = None
    model_id: str | None = "facebook/sam2.1-hiera-large"
    device: str = "cuda"
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.85
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 100


def build_mask_generator(config: Sam2IdMapConfig) -> SAM2AutomaticMaskGenerator:
    if config.checkpoint and config.model_cfg:
        sam_model = build_sam2(
            config.model_cfg,
            config.checkpoint,
            device=config.device,
        )
    elif config.model_id:
        return SAM2AutomaticMaskGenerator.from_pretrained(
            config.model_id,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=config.pred_iou_thresh,
            box_nms_thresh=config.box_nms_thresh,
            stability_score_thresh=config.stability_score_thresh,
            crop_n_layers=config.crop_n_layers,
            crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
            min_mask_region_area=config.min_mask_region_area,
            output_mode="binary_mask",
        )
    else:
        raise ValueError("Provide either --model_id or both --checkpoint and --model_cfg")

    return SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=config.points_per_side,
        points_per_batch=config.points_per_batch,
        pred_iou_thresh=config.pred_iou_thresh,
        box_nms_thresh=config.box_nms_thresh,
        stability_score_thresh=config.stability_score_thresh,
        crop_n_layers=config.crop_n_layers,
        crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
        min_mask_region_area=config.min_mask_region_area,
        output_mode="binary_mask",
    )


def load_rgb_image(image_path: str, resolution: int = -1) -> np.ndarray:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    if resolution == -1:
        if height > 1080:
            scale = height / 1080
        else:
            scale = 1.0
    else:
        scale = width / resolution

    if scale != 1.0:
        new_size = (int(width / scale), int(height / scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return image


@torch.inference_mode()
def generate_masks(
    mask_generator: SAM2AutomaticMaskGenerator,
    image: np.ndarray,
) -> list[dict]:
    device_type = mask_generator.predictor.device.type
    if device_type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    with autocast_ctx:
        return mask_generator.generate(image)
