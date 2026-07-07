from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from instascene.manifest.io import (
    load_scene_paths_manifest,
    manifest_dataset_id,
    manifest_id_map_source,
    manifest_pair_by,
    resolve_scene_paths,
)
from instascene.types import normalize_overlay_styles
from instascene.vlm.config import (
    DEFAULT_PROMPT_TEMPLATE,
    ObjectFilterConfig,
    PipelineConfig,
    RunControlConfig,
    SceneInputConfig,
    ViewRenderConfig,
    VlmConfig,
    resolve_view_compose_fields,
    resolve_vlm_prompt_template,
)
from instascene.vlm.pipeline import SceneRunInput, process_scene


def _parse_overlay_style_arg(value: str) -> str:
    try:
        normalize_overlay_styles(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ",".join(token.strip() for token in value.split(",") if token.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM scene object processing pipeline")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Scene-path manifest JSON (from sample_scenes.py --dataset ...)",
    )
    parser.add_argument(
        "--scene-key",
        type=str,
        default="",
        help="Process only this scene_key (e.g. bench or scene_001/foo). Default: all scenes in manifest.",
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


def _iter_scene_inputs(manifest_path: Path, scene_key_filter: str) -> list[SceneRunInput]:
    doc = load_scene_paths_manifest(manifest_path)
    dataset_id = manifest_dataset_id(doc)
    id_map_source = manifest_id_map_source(doc)
    pair_by = manifest_pair_by(doc)
    inputs: list[SceneRunInput] = []
    for entry in doc.scenes:
        resolved = resolve_scene_paths(entry, doc.dataset_root)
        if scene_key_filter and resolved.scene_key != scene_key_filter:
            continue
        inputs.append(
            SceneRunInput(
                dataset_id=dataset_id,
                scene_key=resolved.scene_key,
                resolved=resolved,
                id_map_source=id_map_source,
                pair_by=pair_by,
            )
        )
    if scene_key_filter and not inputs:
        raise ValueError(f"scene_key {scene_key_filter!r} not found in manifest: {manifest_path}")
    if not inputs:
        raise ValueError(f"No scenes found in manifest: {manifest_path}")
    return inputs


async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
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

    cfg = PipelineConfig(
        scene_input=SceneInputConfig(manifest_path=manifest_path),
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

    scene_key_filter = (args.scene_key or "").strip()
    scene_inputs = _iter_scene_inputs(manifest_path, scene_key_filter)

    if (cfg.run.target_object_id is not None or cfg.run.single_object_only) and len(scene_inputs) != 1:
        raise ValueError(
            "--object-id / --single-object requires exactly one scene; use --scene-key to select one scene"
        )

    for scene_input in scene_inputs:
        output_path = await process_scene(cfg, scene_input)
        print(f"[done] {scene_input.dataset_id}/{scene_input.scene_key} -> {output_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
