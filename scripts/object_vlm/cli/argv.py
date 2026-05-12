from __future__ import annotations

import argparse

from ..config import DEFAULT_PROMPT_TEMPLATE


def parse_overlay_style_arg(value: str) -> str:
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
    parser = argparse.ArgumentParser(description="Multi-view instance VLM object pipeline")
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
        type=parse_overlay_style_arg,
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
