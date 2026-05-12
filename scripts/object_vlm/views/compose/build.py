from __future__ import annotations

from typing import Any, List

from ...config import PipelineConfig
from ..media.image_io import encode_rgb_to_jpeg_base64, read_image_rgb
from ..selection import ViewInfo
from ..types import ImageBatch, PreparedImage
from . import geometry, layout, spec, variants


def prepare_image_batches(views: List[ViewInfo], config: PipelineConfig) -> List[ImageBatch]:
    if not views:
        return []

    parsed = spec.parse_compose_spec(config.view_render.view_compose_spec)
    panels = parsed.get("panels")
    assert isinstance(panels, list)
    title_spec = parsed["titles"]
    layout_spec = parsed["layout"]
    fill_value = int(parsed["align"]["fill_value"])
    titles_enabled = bool(title_spec.get("enabled", False))

    images: List[PreparedImage] = []
    for view in views:
        image_rgb = read_image_rgb(view.image_path)
        rendered_panels: list[dict[str, Any]] = []
        for p in panels:
            if not isinstance(p, dict):
                raise ValueError("--view-compose-spec.panels items must be objects")
            variant = p.get("variant")
            if not isinstance(variant, str) or not variant.strip():
                raise ValueError("Each panel must have a non-empty string field: variant")
            title = p.get("title")
            if title is not None and not isinstance(title, str):
                raise ValueError("panel.title must be a string when provided")

            panel_rgb = variants.render_variant(
                image_rgb,
                view.mask,
                view.bbox_xyxy,
                variant=variant.strip(),
                config=config,
                spec=parsed,
            )
            panel_title = title if isinstance(title, str) else spec.default_title(variant.strip())
            rendered_panels.append({"variant": variant.strip(), "title": panel_title, "image_rgb": panel_rgb})

        if titles_enabled:
            max_h = max(rp["image_rgb"].shape[0] for rp in rendered_panels)
            max_w = max(rp["image_rgb"].shape[1] for rp in rendered_panels)
            for rp in rendered_panels:
                padded = geometry.pad_to_size_center(rp["image_rgb"], max_h, max_w, fill_value)
                rp["image_rgb"] = geometry.add_panel_title(padded, rp["title"], title_spec)

        composed_rgb = layout.compose_panels([rp["image_rgb"] for rp in rendered_panels], layout_spec, fill_value)
        panel_meta = [
            {
                "variant": rp["variant"],
                "title": (rp["title"] if isinstance(rp["title"], str) else spec.default_title(rp["variant"])),
            }
            for rp in rendered_panels
        ]
        images.append(
            PreparedImage(
                view_name=view.view_name,
                variant="compose",
                image_base64=encode_rgb_to_jpeg_base64(composed_rgb),
                meta={
                    "compose": {
                        "panels": panel_meta,
                        "layout": layout_spec,
                        "titles": title_spec,
                    }
                },
            )
        )

    return [ImageBatch(images=images, mode="compose", meta={"compose_spec": parsed})]
