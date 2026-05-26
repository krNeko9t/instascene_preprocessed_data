from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from openai import AsyncOpenAI, OpenAI
from PIL import Image

from instascene.vlm.config import PipelineConfig
from instascene.vlm.preparer import PreparedImage

DEFAULT_AUDIT_PROMPT = """You are given multiple images in a single user message, in order: image 1 first, then image 2, and so on.

Answer these questions:
1) How many distinct images can you actually see and use? (integer only for this line)
2) For each image index i starting from 1, write one short line describing something *specific and visible* in that image (a detail that could differ from other images). If you truly cannot access image i, write "UNSEEN" for that line.

Use exactly this format (no markdown fences):

COUNT: <integer>
IMAGE_1: <one short sentence or UNSEEN>
IMAGE_2: ...
(continue until IMAGE_<COUNT>)
"""

__all__ = [
    "DEFAULT_AUDIT_PROMPT",
    "VLMClient",
    "audit_images",
    "build_vlm_messages",
    "encode_image_path_to_jpeg_b64",
    "extract_text",
]


def extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def build_vlm_messages(prompt: str, jpeg_b64_list: Sequence[str]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in jpeg_b64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return [{"role": "user", "content": content}]


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_rgb_to_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    q = max(1, min(100, quality))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=q)
        return base64.b64encode(bio.getvalue()).decode("ascii")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_image_path_to_jpeg_b64(path: Path, *, quality: int = 90) -> str:
    rgb = _read_rgb(path)
    return encode_rgb_to_jpeg_b64(rgb, quality=quality)


class VLMClient:
    def __init__(self, config: PipelineConfig) -> None:
        if not config.vlm.api_key:
            raise ValueError("Missing API key. Set --api-key or environment variable VLM_API_KEY.")
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.vlm.api_key,
            base_url=config.vlm.api_base_url,
            timeout=config.vlm.request_timeout_seconds,
        )
        self.semaphore = asyncio.Semaphore(max(1, config.vlm.max_concurrent))

    def _build_messages(self, prompt: str, images: Sequence[PreparedImage]) -> list[dict]:
        return build_vlm_messages(prompt, [img.image_base64 for img in images])

    async def infer(self, prompt: str, images: Sequence[PreparedImage]) -> str:
        messages = self._build_messages(prompt, images)
        last_error: Exception | None = None

        for attempt in range(self.config.vlm.max_retries + 1):
            try:
                async with self.semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.config.vlm.model_name,
                        messages=messages,
                    )
                choice = resp.choices[0].message
                return extract_text(choice.content).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.config.vlm.max_retries:
                    break
                wait_s = self.config.vlm.retry_backoff_seconds * (2**attempt)
                await asyncio.sleep(wait_s)

        raise RuntimeError(f"VLM request failed after retries: {last_error}") from last_error


def audit_images(
    image_paths: Sequence[Path],
    *,
    api_base_url: str,
    api_key: str,
    model_name: str,
    prompt: str = "",
    timeout: float = 120.0,
    jpeg_quality: int = 90,
) -> tuple[list[str], str]:
    """Encode images and call a sync OpenAI-compatible VLM. Returns (b64_list, model_reply)."""
    b64_list = [
        encode_image_path_to_jpeg_b64(p, quality=jpeg_quality)
        for p in image_paths
    ]
    audit_prompt = prompt.strip() or DEFAULT_AUDIT_PROMPT
    messages = build_vlm_messages(audit_prompt, b64_list)
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url.rstrip("/"),
        timeout=timeout,
    )
    resp = client.chat.completions.create(model=model_name, messages=messages)
    return b64_list, extract_text(resp.choices[0].message.content).strip()
