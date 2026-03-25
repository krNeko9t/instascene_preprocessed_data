#!/usr/bin/env python3
"""
Standalone probe: send multiple images to an OpenAI-compatible VLM and ask it
how many images it sees and what each contains.

Run from anywhere (with deps installed), e.g.:

  cd /path/to/instascene_preprocessed_data/scripts
  python ask_vlm_image_audit.py \\
    --api-base-url https://api.openai.com/v1 \\
    --model-name gpt-4.1-mini \\
    --images img1.jpg img2.jpg

API key: --api-key or env VLM_API_KEY (same as the main pipeline).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment, misc]

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None  # type: ignore[assignment, misc]


DEFAULT_PROMPT = """You are given multiple images in a single user message, in order: image 1 first, then image 2, and so on.

Answer these questions:
1) How many distinct images can you actually see and use? (integer only for this line)
2) For each image index i starting from 1, write one short line describing something *specific and visible* in that image (a detail that could differ from other images). If you truly cannot access image i, write "UNSEEN" for that line.

Use exactly this format (no markdown fences):

COUNT: <integer>
IMAGE_1: <one short sentence or UNSEEN>
IMAGE_2: ...
(continue until IMAGE_<COUNT>)
"""


def _read_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if Image is None:
        raise RuntimeError("Need opencv-python or Pillow to read images: pip install opencv-python")
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    q = max(1, min(100, quality))
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")
    if Image is None:
        raise RuntimeError("Need opencv-python or Pillow to encode JPEG")
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    bio = io.BytesIO()
    im.save(bio, format="JPEG", quality=q)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    return str(content)


def build_messages(prompt: str, jpeg_b64_list: list[str]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in jpeg_b64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return [{"role": "user", "content": content}]


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

    b64_list: list[str] = []
    for p in paths:
        rgb = _read_rgb(p)
        b64_list.append(_encode_jpeg_b64(rgb, quality=max(1, min(100, args.jpeg_quality))))

    print(f"Prepared {len(b64_list)} image(s):")
    for i, p in enumerate(paths):
        n = len(b64_list[i])
        print(f"  [{i + 1}] {p}  (base64 jpeg payload ~{n} chars)")

    if args.dry_run:
        return 0

    prompt = args.prompt.strip() or DEFAULT_PROMPT
    messages = build_messages(prompt, b64_list)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url.rstrip("/"),
        timeout=args.timeout,
    )
    resp = client.chat.completions.create(model=args.model_name, messages=messages)
    text = _extract_text(resp.choices[0].message.content)
    print("\n--- model reply ---\n")
    print(text.strip())
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
