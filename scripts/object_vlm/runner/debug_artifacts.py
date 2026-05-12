from __future__ import annotations

import base64
import json
from pathlib import Path

from ..views.types import ImageBatch


def safe_filename_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def dump_debug_batch(
    debug_dir: Path,
    *,
    object_id: int,
    batch_index: int,
    batch: ImageBatch,
    prompt: str,
) -> None:
    object_dir = debug_dir / f"object_{object_id:03d}"
    object_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = object_dir / f"batch_{batch_index:02d}_prompt.txt"
    with prompt_path.open("w", encoding="utf-8") as f:
        f.write(prompt)

    meta = {
        "object_id": object_id,
        "batch_index": batch_index,
        "mode": batch.mode,
        "batch_meta": batch.meta or {},
        "num_images": len(batch.images),
        "images": [
            {
                "index": i,
                "view_name": img.view_name,
                "variant": img.variant,
                "meta": img.meta or {},
            }
            for i, img in enumerate(batch.images)
        ],
    }
    with (object_dir / f"batch_{batch_index:02d}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for i, img in enumerate(batch.images):
        image_bytes = base64.b64decode(img.image_base64)
        view_token = safe_filename_token(img.view_name)
        var_token = safe_filename_token(img.variant)
        image_path = object_dir / f"batch_{batch_index:02d}_{i:02d}_{view_token}_{var_token}.jpg"
        with image_path.open("wb") as f:
            f.write(image_bytes)
