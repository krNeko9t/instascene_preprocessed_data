"""Typed model and parser for the view-compose spec (panels, layout, titles, align)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "ComposeSpec",
    "HighlightOutsideDarkSpec",
    "KNOWN_PANEL_VARIANTS",
    "LayoutSpec",
    "MaskBwSpec",
    "PanelSpec",
    "TitleSpec",
    "parse_compose_spec",
]

KNOWN_PANEL_VARIANTS = {"origin", "mask_bw", "highlight_outside_dark", "overlay", "crop"}

_DEFAULT_PANEL_TITLES = {
    "origin": "origin",
    "mask_bw": "mask",
    "highlight_outside_dark": "highlight",
    "overlay": "overlay",
    "crop": "crop",
}


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass(slots=True, frozen=True)
class PanelSpec:
    variant: str
    title: str


@dataclass(slots=True, frozen=True)
class LayoutSpec:
    """Layout of panels in the composed image; rows/cols are resolved for grid only."""

    type: str
    rows: int | None = None
    cols: int | None = None


@dataclass(slots=True, frozen=True)
class TitleSpec:
    enabled: bool = False
    height: int = 28
    bg_value: int = 245
    fg_value: int = 20


@dataclass(slots=True, frozen=True)
class MaskBwSpec:
    foreground: int = 255
    background: int = 0
    invert: bool = False


@dataclass(slots=True, frozen=True)
class HighlightOutsideDarkSpec:
    outside_factor: float = 0.35


@dataclass(slots=True, frozen=True)
class ComposeSpec:
    panels: tuple[PanelSpec, ...]
    layout: LayoutSpec
    titles: TitleSpec
    mask_bw: MaskBwSpec
    highlight_outside_dark: HighlightOutsideDarkSpec
    fill_value: int = 255

    def as_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_panels(raw: object) -> tuple[PanelSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("--view-compose-spec.panels must be a non-empty list")
    panels: list[PanelSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("--view-compose-spec.panels items must be objects")
        variant = item.get("variant")
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("Each panel must have a non-empty string field: variant")
        variant = variant.strip()
        if variant not in KNOWN_PANEL_VARIANTS:
            raise ValueError(f"Unsupported panel variant: {variant}")
        title = item.get("title")
        if title is not None and not isinstance(title, str):
            raise ValueError("panel.title must be a string when provided")
        if title is None:
            title = _DEFAULT_PANEL_TITLES[variant]
        panels.append(PanelSpec(variant=variant, title=title))
    return tuple(panels)


def _parse_layout(raw: object, *, n_panels: int) -> LayoutSpec:
    if not isinstance(raw, dict):
        raise ValueError("--view-compose-spec.layout must be an object")
    layout_type = str(raw.get("type", "horizontal")).strip().lower()
    if layout_type not in {"horizontal", "vertical", "grid"}:
        raise ValueError("layout.type must be one of: horizontal, vertical, grid")
    if layout_type != "grid":
        return LayoutSpec(type=layout_type)

    cols = raw.get("cols")
    rows = raw.get("rows")
    if cols is None and rows is None:
        cols = 2
    if cols is not None:
        cols = int(cols)
        if cols <= 0:
            raise ValueError("layout.cols must be positive")
    if rows is not None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("layout.rows must be positive")
    if cols is None:
        cols = math.ceil(n_panels / rows)
    if rows is None:
        rows = math.ceil(n_panels / cols)
    if rows * cols < n_panels:
        raise ValueError("layout.rows * layout.cols is smaller than number of panels")
    return LayoutSpec(type="grid", rows=rows, cols=cols)


def _parse_titles(raw: object) -> TitleSpec:
    if not isinstance(raw, dict):
        raise ValueError("--view-compose-spec.titles must be an object")
    return TitleSpec(
        enabled=bool(raw.get("enabled", False)),
        height=_clamp(int(raw.get("height", 28)), 16, 96),
        bg_value=_clamp(int(raw.get("bg_value", 245)), 0, 255),
        fg_value=_clamp(int(raw.get("fg_value", 20)), 0, 255),
    )


def _parse_align_fill_value(raw: object) -> int:
    if not isinstance(raw, dict):
        raise ValueError("--view-compose-spec.align must be an object")
    mode = str(raw.get("mode", "pad")).strip().lower()
    if mode != "pad":
        raise ValueError("Only align.mode='pad' is supported currently")
    return _clamp(int(raw.get("fill_value", 255)), 0, 255)


def _parse_mask_bw(raw: object) -> MaskBwSpec:
    opts = raw if isinstance(raw, dict) else {}
    return MaskBwSpec(
        foreground=_clamp(int(opts.get("foreground", 255)), 0, 255),
        background=_clamp(int(opts.get("background", 0)), 0, 255),
        invert=bool(opts.get("invert", False)),
    )


def _parse_highlight(raw: object) -> HighlightOutsideDarkSpec:
    opts = raw if isinstance(raw, dict) else {}
    factor = float(opts.get("outside_factor", 0.35))
    return HighlightOutsideDarkSpec(outside_factor=max(0.0, min(1.0, factor)))


def parse_compose_spec(raw_spec: str) -> ComposeSpec:
    try:
        data = json.loads(raw_spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --view-compose-spec JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--view-compose-spec must be a JSON object")

    panels = _parse_panels(data.get("panels"))
    return ComposeSpec(
        panels=panels,
        layout=_parse_layout(data.get("layout", {"type": "horizontal"}), n_panels=len(panels)),
        titles=_parse_titles(data.get("titles", {"enabled": False})),
        mask_bw=_parse_mask_bw(data.get("mask_bw")),
        highlight_outside_dark=_parse_highlight(data.get("highlight_outside_dark")),
        fill_value=_parse_align_fill_value(data.get("align", {"mode": "pad", "fill_value": 255})),
    )
