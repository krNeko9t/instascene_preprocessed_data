from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

from .argv import build_arg_parser


async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from ..config import (
        ObjectFilterConfig,
        PipelineConfig,
        RunControlConfig,
        SceneInputConfig,
        ViewRenderConfig,
        VlmConfig,
        resolve_view_compose_fields,
        resolve_vlm_prompt_template,
    )
    from ..runner import list_datasets, list_scenes, process_scene

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
