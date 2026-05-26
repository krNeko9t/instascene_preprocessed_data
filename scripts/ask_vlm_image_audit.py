#!/usr/bin/env python3
"""Standalone probe: ask a VLM how many images it sees and what each contains."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from instascene.vlm.client import DEFAULT_AUDIT_PROMPT, audit_images, encode_image_path_to_jpeg_b64


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a VLM how many images it sees and what each shows.")
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        required=True,
        help="One or more image file paths (sent in this order)",
    )
    parser.add_argument("--api-base-url", type=str, default=os.getenv("VLM_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.getenv("VLM_API_KEY", ""))
    parser.add_argument("--model-name", type=str, default=os.getenv("VLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Override the default audit prompt (plain text)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality 1-100 for data URLs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load/encode images and print sizes; do not call the API",
    )
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        print("Error: missing API key. Set --api-key or VLM_API_KEY.", file=sys.stderr)
        return 2

    paths = [p.expanduser().resolve() for p in args.images]
    for p in paths:
        if not p.is_file():
            print(f"Error: not a file: {p}", file=sys.stderr)
            return 2

    quality = max(1, min(100, args.jpeg_quality))
    b64_list = [encode_image_path_to_jpeg_b64(p, quality=quality) for p in paths]

    print(f"Prepared {len(b64_list)} image(s):")
    for i, p in enumerate(paths):
        n = len(b64_list[i])
        print(f"  [{i + 1}] {p}  (base64 jpeg payload ~{n} chars)")

    if args.dry_run:
        return 0

    _, text = audit_images(
        paths,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        prompt=args.prompt.strip() or DEFAULT_AUDIT_PROMPT,
        timeout=args.timeout,
        jpeg_quality=quality,
    )
    print("\n--- model reply ---\n")
    print(text)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
