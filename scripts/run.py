from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    from config import DEFAULT_PROMPT_TEMPLATE

    parser = argparse.ArgumentParser(description="VLM scene object processing pipeline")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="3dovs", help="Dataset name or 'all'")
    parser.add_argument("--scene", type=str, default="bench", help="Scene name or 'all'")
    parser.add_argument("--mask-subdir", type=str, default="mask")
    parser.add_argument(
        "--id-map-source",
        type=str,
        default="npy",
        choices=["npy", "png"],
        help="ID map source: npy from id_maps/ or png from sam/<mask_subdir>/",
    )
    parser.add_argument("--min-pixel-count", type=int, default=300)
    parser.add_argument("--min-pixel-ratio", type=float, default=0.15)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument(
        "--image-mode",
        type=str,
        default="pair",
        choices=["crop_only", "overlay_only", "pair", "per_view"],
    )
    parser.add_argument(
        "--overlay-style",
        type=str,
        default="all",
        choices=["contour", "bbox", "semitransparent", "all"],
    )
    parser.add_argument("--crop-padding-ratio", type=float, default=0.15)
    parser.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Run output root directory, default: outputs/run_<timestamp>",
    )
    parser.add_argument("--run-name", type=str, default="", help="Optional run name suffix")
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="Only process this object ID in the target scene",
    )
    parser.add_argument(
        "--single-object",
        action="store_true",
        help="Only process one object ID (the first pending ID)",
    )
    parser.add_argument(
        "--no-save-debug-inputs",
        action="store_true",
        help="Disable saving per-object debug inputs (prompt and sent images)",
    )
    return parser


async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from config import PipelineConfig
    from pipeline import list_datasets, list_scenes, process_scene

    data_root = Path(args.data_root).expanduser().resolve()
    if args.output_dir.strip():
        output_dir = args.output_dir
    else:
        run_suffix = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/run_{run_suffix}"
    cfg = PipelineConfig.from_values(
        data_root=data_root,
        dataset=args.dataset,
        scene=args.scene,
        mask_subdir=args.mask_subdir,
        id_map_source=args.id_map_source,
        min_pixel_count=args.min_pixel_count,
        min_pixel_ratio=args.min_pixel_ratio,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        max_views=args.max_views,
        image_mode=args.image_mode,
        overlay_style=args.overlay_style,
        crop_padding_ratio=args.crop_padding_ratio,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        output_dir=output_dir,
        target_object_id=args.object_id,
        single_object_only=args.single_object,
        save_debug_inputs=not args.no_save_debug_inputs,
    )

    if (cfg.target_object_id is not None or cfg.single_object_only) and (
        cfg.dataset == "all" or cfg.scene == "all"
    ):
        raise ValueError("--object-id / --single-object must be used with a specific --dataset and --scene")

    datasets = list_datasets(cfg.data_root) if cfg.dataset == "all" else [cfg.dataset]
    for dataset in datasets:
        scenes = list_scenes(cfg.data_root, dataset) if cfg.scene == "all" else [cfg.scene]
        for scene in scenes:
            output_path = await process_scene(cfg, dataset, scene)
            print(f"[done] {dataset}/{scene} -> {output_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
