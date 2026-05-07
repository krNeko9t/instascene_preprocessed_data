from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path


def _parse_overlay_style_arg(value: str) -> str:
    allowed = {"contour", "bbox", "semitransparent", "all"}
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("overlay-style cannot be empty")
    invalid = sorted({token for token in tokens if token not in allowed})
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid overlay style(s): {', '.join(invalid)}; allowed: contour,bbox,semitransparent,all"
        )
    return ",".join(tokens)


def build_arg_parser() -> argparse.ArgumentParser:
    from config import DEFAULT_PROMPT_TEMPLATE

    parser = argparse.ArgumentParser(description="VLM scene object processing pipeline")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="3dovs", help="Dataset name or 'all'")
    parser.add_argument("--scene", type=str, default="bench", help="Scene name or 'all'")
    parser.add_argument("--mask-subdir", type=str, default="mask")
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="images",
        help="Subdirectory of each scene for RGB images (e.g. rgb for RE10K, images for Infinigen-style).",
    )
    parser.add_argument(
        "--id-map-source",
        type=str,
        default="npy",
        choices=["npy", "png", "sam2_json"],
        help="ID map source: npy from id_maps/, png from sam/<mask_subdir>/, or sam2_json (auto_masks.json RLE).",
    )
    parser.add_argument(
        "--sam2-json",
        type=str,
        default="",
        help="Path to auto_masks.json (optional; default: <scene_root>/../sam2_results/<scene>/auto_masks.json).",
    )
    parser.add_argument("--min-pixel-count", type=int, default=300)
    parser.add_argument("--min-pixel-ratio", type=float, default=0.15)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument(
        "--view-compose-spec",
        type=str,
        default="",
        help="Inline JSON spec to compose panels into ONE image per view (overrides --view-compose-spec-file).",
    )
    parser.add_argument(
        "--view-compose-spec-file",
        type=str,
        default="",
        help="Path to JSON file for compose spec (used when --view-compose-spec is empty).",
    )
    parser.add_argument(
        "--overlay-style",
        type=_parse_overlay_style_arg,
        default="bbox",
        help="Overlay style(s), supports comma-separated values, e.g. bbox,contour",
    )
    parser.add_argument("--crop-padding-ratio", type=float, default=0.15)
    parser.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Path to a .prompt file to use as prompt template (overrides --prompt-template)",
    )
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

    from config import (
        ObjectFilterConfig,
        PipelineConfig,
        RunControlConfig,
        SceneInputConfig,
        ViewRenderConfig,
        VlmConfig,
        resolve_view_compose_fields,
        resolve_vlm_prompt_template,
    )
    from pipeline import list_datasets, list_scenes, process_scene

    data_root = Path(args.data_root).expanduser().resolve()
    if args.output_dir.strip():
        output_dir = args.output_dir
    else:
        run_suffix = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/run_{run_suffix}"

    compose_spec, compose_spec_file = resolve_view_compose_fields(
        inline=args.view_compose_spec,
        spec_file=args.view_compose_spec_file,
    )
    prompt_file, prompt_template = resolve_vlm_prompt_template(
        prompt_file=args.prompt_file,
        inline_template=args.prompt_template,
    )

    sam2_json_path = (
        Path(args.sam2_json).expanduser().resolve()
        if (args.sam2_json or "").strip()
        else None
    )

    cfg = PipelineConfig(
        scene_input=SceneInputConfig(
            data_root=data_root,
            dataset=args.dataset,
            scene=args.scene,
            mask_subdir=args.mask_subdir,
            id_map_source=args.id_map_source,
            image_subdir=args.image_subdir,
            sam2_json_path=sam2_json_path,
        ),
        object_filter=ObjectFilterConfig(
            min_pixel_count=args.min_pixel_count,
            min_pixel_ratio=args.min_pixel_ratio,
            min_bbox_area_ratio=args.min_bbox_area_ratio,
        ),
        view_render=ViewRenderConfig(
            max_views=args.max_views,
            overlay_style=args.overlay_style,
            crop_padding_ratio=args.crop_padding_ratio,
            view_compose_spec=compose_spec,
            view_compose_spec_file=compose_spec_file,
        ),
        vlm=VlmConfig(
            api_base_url=args.api_base_url,
            api_key=args.api_key or os.getenv("VLM_API_KEY", ""),
            model_name=args.model_name,
            prompt_file=prompt_file,
            prompt_template=prompt_template,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        ),
        run=RunControlConfig(
            output_dir=output_dir,
            target_object_id=args.object_id,
            single_object_only=args.single_object,
            save_debug_inputs=not args.no_save_debug_inputs,
        ),
    )

    if (cfg.run.target_object_id is not None or cfg.run.single_object_only) and (
        cfg.scene_input.dataset == "all" or cfg.scene_input.scene == "all"
    ):
        raise ValueError("--object-id / --single-object must be used with a specific --dataset and --scene")

    datasets = list_datasets(cfg.scene_input.data_root) if cfg.scene_input.dataset == "all" else [cfg.scene_input.dataset]
    for dataset in datasets:
        scenes = list_scenes(cfg.scene_input.data_root, dataset) if cfg.scene_input.scene == "all" else [cfg.scene_input.scene]
        for scene in scenes:
            output_path = await process_scene(cfg, dataset, scene)
            print(f"[done] {dataset}/{scene} -> {output_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
