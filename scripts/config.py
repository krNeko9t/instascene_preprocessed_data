from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ImageMode = Literal["crop_only", "overlay_only", "pair", "per_view"]
OverlayStyle = Literal["contour", "bbox", "semitransparent", "all"]
IdMapSource = Literal["npy", "png"]

DEFAULT_PROMPT_TEMPLATE = (
    "You are given images of the same object from multiple views in a scene. "
    "Describe the target object only. "
    "Return a concise caption in one sentence.\n\n"
    "Dataset: {dataset}\n"
    "Scene: {scene}\n"
    "Object ID: {object_id}\n"
    "Num views: {n_views}\n"
    "View names: {view_names}\n"
    "Input mode: {image_mode}"
)


def _load_prompt_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


@dataclass(slots=True)
class PipelineConfig:
    data_root: Path
    dataset: str = "3dovs"
    scene: str = "bench"
    mask_subdir: str = "mask"
    id_map_source: IdMapSource = "npy"
    min_pixel_count: int = 300
    min_pixel_ratio: float = 0.15
    min_bbox_area_ratio: float = 0.002
    max_views: int = 8
    image_mode: ImageMode = "pair"
    overlay_style: OverlayStyle = "all"
    crop_padding_ratio: float = 0.15
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4.1-mini"
    prompt_file: str = ""
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_concurrent: int = 4
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    request_timeout_seconds: float = 120.0
    output_dir: str = "vlm_results"
    target_object_id: int | None = None
    single_object_only: bool = False
    save_debug_inputs: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        data_root: str | Path,
        dataset: str,
        scene: str,
        mask_subdir: str,
        id_map_source: IdMapSource,
        min_pixel_count: int,
        min_pixel_ratio: float,
        min_bbox_area_ratio: float,
        max_views: int,
        image_mode: ImageMode,
        overlay_style: OverlayStyle,
        crop_padding_ratio: float,
        api_base_url: str,
        api_key: str | None,
        model_name: str,
        prompt_file: str,
        prompt_template: str | None,
        max_concurrent: int,
        max_retries: int,
        retry_backoff_seconds: float,
        request_timeout_seconds: float,
        output_dir: str,
        target_object_id: int | None,
        single_object_only: bool,
        save_debug_inputs: bool,
    ) -> "PipelineConfig":
        return cls(
            data_root=Path(data_root).expanduser().resolve(),
            dataset=dataset,
            scene=scene,
            mask_subdir=mask_subdir,
            id_map_source=id_map_source,
            min_pixel_count=min_pixel_count,
            min_pixel_ratio=min_pixel_ratio,
            min_bbox_area_ratio=min_bbox_area_ratio,
            max_views=max_views,
            image_mode=image_mode,
            overlay_style=overlay_style,
            crop_padding_ratio=crop_padding_ratio,
            api_base_url=api_base_url,
            api_key=api_key or os.getenv("VLM_API_KEY", ""),
            model_name=model_name,
            prompt_file=prompt_file,
            prompt_template=_load_prompt_file(prompt_file) if prompt_file else (prompt_template or DEFAULT_PROMPT_TEMPLATE),
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            request_timeout_seconds=request_timeout_seconds,
            output_dir=output_dir,
            target_object_id=target_object_id,
            single_object_only=single_object_only,
            save_debug_inputs=save_debug_inputs,
        )

    def scene_root(self, dataset: str | None = None, scene: str | None = None) -> Path:
        ds = dataset or self.dataset
        sc = scene or self.scene
        return self.data_root / ds / sc

    def scene_output_dir(self, dataset: str | None = None, scene: str | None = None) -> Path:
        root = Path(self.output_dir)
        if not root.is_absolute():
            root = self.data_root / root
        ds = dataset or self.dataset
        sc = scene or self.scene
        return root / ds / sc
