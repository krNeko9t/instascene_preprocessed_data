from __future__ import annotations

import asyncio
from typing import Sequence

from openai import AsyncOpenAI

from config import PipelineConfig
from image_preparer import PreparedImage


def _extract_text(content: object) -> str:
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
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image.image_base64}"},
                }
            )
        return [{"role": "user", "content": content}]

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
                return _extract_text(choice.content).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.config.vlm.max_retries:
                    break
                wait_s = self.config.vlm.retry_backoff_seconds * (2**attempt)
                await asyncio.sleep(wait_s)

        raise RuntimeError(f"VLM request failed after retries: {last_error}") from last_error
