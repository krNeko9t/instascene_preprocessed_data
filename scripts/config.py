from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

OverlayStyle = str
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

DEFAULT_VIEW_COMPOSE_SPEC = json.dumps(
    {
        "layout": {"type": "horizontal"},
        "titles": {"enabled": True},
        "panels": [
            {"variant": "origin", "title": "origin"},
            {"variant": "mask_bw", "title": "mask"},
            {"variant": "highlight_outside_dark", "title": "highlight"},
        ],
        "align": {"mode": "pad", "fill_value": 255},
        "mask_bw": {"foreground": 255, "background": 0, "invert": False},
        "highlight_outside_dark": {"outside_factor": 0.35},
    },
    ensure_ascii=False,
)


def _load_prompt_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def _load_view_compose_spec_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"View compose spec file not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"View compose spec file is empty: {p}")
    return raw


def resolve_view_compose_fields(*, inline: str, spec_file: str) -> tuple[str, str]:
    """Resolve final JSON spec and the spec file path string (for bookkeeping)."""
    inline_s = (inline or "").strip()
    file_s = (spec_file or "").strip()
    if inline_s:
        return inline_s, file_s
    if file_s:
        return _load_view_compose_spec_file(file_s), file_s
    return DEFAULT_VIEW_COMPOSE_SPEC, ""


def resolve_vlm_prompt_template(*, prompt_file: str, inline_template: str) -> tuple[str, str]:
    """Return (prompt_file, template text). File wins when non-empty."""
    pf = (prompt_file or "").strip()
    if pf:
        return pf, _load_prompt_file(pf)
    return "", (inline_template or DEFAULT_PROMPT_TEMPLATE)


@dataclass(slots=True)
class SceneInputConfig:
    """Paths and loaders for scene imagery and ID maps."""

    data_root: Path
    dataset: str = "3dovs"
    scene: str = "bench"
    mask_subdir: str = "mask"
    id_map_source: IdMapSource = "npy"


@dataclass(slots=True)
class ObjectFilterConfig:
    """Thresholds for accepting a view of an object."""

    min_pixel_count: int = 300
    min_pixel_ratio: float = 0.15
    min_bbox_area_ratio: float = 0.002


@dataclass(slots=True)
class ViewRenderConfig:
    """View selection limits and per-view image composition."""

    max_views: int = 8
    overlay_style: OverlayStyle = "all"
    crop_padding_ratio: float = 0.15
    view_compose_spec: str = DEFAULT_VIEW_COMPOSE_SPEC
    view_compose_spec_file: str = ""


@dataclass(slots=True)
class VlmConfig:
    """API client, model, prompt text, and request concurrency / retries."""

    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4.1-mini"
    prompt_file: str = ""
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_concurrent: int = 4
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    request_timeout_seconds: float = 120.0


@dataclass(slots=True)
class RunControlConfig:
    """Where to write results and how much work to do per invocation."""

    output_dir: str = "vlm_results"
    target_object_id: int | None = None
    single_object_only: bool = False
    save_debug_inputs: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """Top-level pipeline settings, grouped by concern."""

    scene_input: SceneInputConfig
    object_filter: ObjectFilterConfig
    view_render: ViewRenderConfig
    vlm: VlmConfig
    run: RunControlConfig

    def scene_root(self, dataset: str | None = None, scene: str | None = None) -> Path:
        ds = dataset or self.scene_input.dataset
        sc = scene or self.scene_input.scene
        return self.scene_input.data_root / ds / sc

    def scene_output_dir(self, dataset: str | None = None, scene: str | None = None) -> Path:
        root = Path(self.run.output_dir)
        if not root.is_absolute():
            root = self.scene_input.data_root / root
        ds = dataset or self.scene_input.dataset
        sc = scene or self.scene_input.scene
        return root / ds / sc
