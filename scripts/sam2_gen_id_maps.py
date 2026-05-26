#!/usr/bin/env python3
"""Run SAM2 on scene images and write id_maps/{stem}.npy for vlm_gen_scene_desc --id-map-source npy.

Requires the sam2 package and torch (GPU recommended). Install separately from scripts/requirements.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from instascene.sam2.scene_batch import iter_targets_from_layout, iter_targets_from_manifest, run_batch
from sam2_idmap.predictor import Sam2IdMapConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM2 automatic segmentation and write scene id_maps/*.npy",
    )
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="3dovs", help="Dataset name or 'all'")
    parser.add_argument("--scene", type=str, default="bench", help="Scene name or 'all'")
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="images",
        help="Image subdirectory under each scene (layout mode only)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Scene-path manifest JSON; when set, ignores --dataset/--scene",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first scene error")
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Max images per scene (-1 means all)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model id. Ignored when --checkpoint is set.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional local SAM2 checkpoint")
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config relative to the sam2 package. Required with --checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resolution",
        type=int,
        default=-1,
        help="Resize width before inference; -1 keeps original unless height > 1080p",
    )
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.7)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--crop_n_layers", type=int, default=1)
    parser.add_argument("--min_mask_region_area", type=int, default=100)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if (args.manifest or "").strip():
        targets = iter_targets_from_manifest(Path(args.manifest))
    else:
        targets = iter_targets_from_layout(
            Path(args.data_root),
            args.dataset,
            args.scene,
            image_subdir=args.image_subdir,
        )

    sam2_config = Sam2IdMapConfig(
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

    batch = run_batch(
        targets,
        sam2_config,
        overwrite=bool(args.overwrite),
        resolution=args.resolution,
        max_images=args.max_images,
        fail_fast=bool(args.fail_fast),
    )

    for scene in batch.scenes:
        if scene.error:
            print(
                f"[error] {scene.label}: {scene.error} "
                f"(processed={scene.processed}, skipped={scene.skipped}, total={scene.total_images})",
                file=sys.stderr,
            )
        else:
            print(
                f"[done] {scene.label}: processed={scene.processed}, "
                f"skipped={scene.skipped}, total={scene.total_images}"
            )

    print(
        f"summary: scenes={len(batch.scenes)} processed={batch.processed} "
        f"skipped={batch.skipped} errors={batch.errors}"
    )
    return 1 if batch.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
