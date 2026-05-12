from __future__ import annotations

import json
from typing import Any


def default_title(variant: str) -> str:
    aliases = {
        "origin": "origin",
        "mask_bw": "mask",
        "highlight_outside_dark": "highlight",
        "overlay": "overlay",
        "crop": "crop",
    }
    return aliases.get(variant, variant)


def parse_compose_spec(raw_spec: str) -> dict[str, Any]:
    try:
        spec = json.loads(raw_spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --view-compose-spec JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("--view-compose-spec must be a JSON object")
    panels = spec.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("--view-compose-spec.panels must be a non-empty list")
    layout = spec.get("layout", {"type": "horizontal"})
    if not isinstance(layout, dict):
        raise ValueError("--view-compose-spec.layout must be an object")
    titles = spec.get("titles", {"enabled": False})
    if not isinstance(titles, dict):
        raise ValueError("--view-compose-spec.titles must be an object")
    align = spec.get("align", {"mode": "pad", "fill_value": 255})
    if not isinstance(align, dict):
        raise ValueError("--view-compose-spec.align must be an object")
    mode = str(align.get("mode", "pad")).strip().lower()
    if mode != "pad":
        raise ValueError("Only align.mode='pad' is supported currently")
    fill_value = int(align.get("fill_value", 255))
    fill_value = max(0, min(255, fill_value))

    spec["layout"] = layout
    spec["titles"] = titles
    spec["align"] = {"mode": mode, "fill_value": fill_value}
    return spec
