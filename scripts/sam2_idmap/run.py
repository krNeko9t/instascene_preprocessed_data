from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from sam2_idmap.id_map import atomic_npy_save, masks_to_id_map
from sam2_idmap.predictor import Sam2IdMapConfig, build_mask_generator, generate_masks, load_rgb_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 automatic segmentation and save instance id maps.")
    parser.add_argument("--image", type=str, help="Path to a single input image.")
    parser.add_argument("--image_dir", type=str, help="Directory of input images.")
    parser.add_argument("--output", type=str, help="Output .npy path for single-image mode.")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch mode.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model id. Ignored when --checkpoint is set.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config relative to the sam2 package. Required with --checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=-1, help="Resize width before inference; -1 keeps original unless >1080p.")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.7)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--crop_n_layers", type=int, default=1)
    parser.add_argument("--min_mask_region_area", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    return parser.parse_args()


def resolve_output_path(
    image_path: Path,
    output: str | None,
    output_dir: str | None,
    single_image: bool,
) -> Path:
    if single_image:
        if output is None:
            raise ValueError("--output is required when using --image")
        return Path(output)

    if output_dir is None:
        raise ValueError("--output_dir is required when using --image_dir")
    return Path(output_dir) / f"{image_path.stem}_id.npy"


def collect_images(image: str | None, image_dir: str | None) -> list[Path]:
    if bool(image) == bool(image_dir):
        raise ValueError("Specify exactly one of --image or --image_dir")

    if image is not None:
        return [Path(image)]

    root = Path(image_dir)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")

    images = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in: {root}")
    return images


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.image, args.image_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    single_image = args.image is not None
    config = Sam2IdMapConfig(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg if args.checkpoint else None,
        model_id=None if args.checkpoint else args.model_id,
        device=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )

    mask_generator = build_mask_generator(config)

    skipped = 0
    processed = 0
    for image_path in tqdm(image_paths, desc="SAM2 id maps"):
        output_path = resolve_output_path(image_path, args.output, args.output_dir, single_image)
        if not args.overwrite and output_path.exists():
            skipped += 1
            continue

        image = load_rgb_image(str(image_path), resolution=args.resolution)
        masks = generate_masks(mask_generator, image)
        id_map = masks_to_id_map(masks, image.shape[0], image.shape[1])
        atomic_npy_save(str(output_path), id_map)
        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, total={len(image_paths)}")


if __name__ == "__main__":
    main()
