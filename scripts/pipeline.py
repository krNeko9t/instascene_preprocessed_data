from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from config import PipelineConfig
from data_loader import load_scene
from image_preparer import ImageBatch, prepare_image_batches
from view_selector import select_views_for_object
from vlm_client import VLMClient


@dataclass(slots=True)
class ObjectResult:
    id: int
    response: Any
    n_views: int
    mode: str


def _model_to_filename(model_name: str) -> str:
    safe = model_name.replace("/", "__")
    return f"{safe}.json"


def _build_prompt(
    config: PipelineConfig,
    *,
    dataset: str,
    scene: str,
    object_id: int,
    batch: ImageBatch,
) -> str:
    view_names = [img.view_name for img in batch.images]
    return config.prompt_template.format(
        dataset=dataset,
        scene=scene,
        object_id=object_id,
        n_views=len(set(view_names)),
        view_names=", ".join(sorted(set(view_names))),
        image_mode=batch.mode,
    )


def _load_existing_results(output_json: Path) -> Dict[int, Dict[str, Any]]:
    if not output_json.exists():
        return {}
    with output_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    existing: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                continue
            existing[int(item["id"])] = item
    return existing


def _append_error_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _safe_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _dump_debug_batch(
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
        "num_images": len(batch.images),
        "images": [
            {"index": i, "view_name": img.view_name, "variant": img.variant}
            for i, img in enumerate(batch.images)
        ],
    }
    with (object_dir / f"batch_{batch_index:02d}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for i, img in enumerate(batch.images):
        image_bytes = base64.b64decode(img.image_base64)
        view_token = _safe_token(img.view_name)
        var_token = _safe_token(img.variant)
        image_path = object_dir / f"batch_{batch_index:02d}_{i:02d}_{view_token}_{var_token}.jpg"
        with image_path.open("wb") as f:
            f.write(image_bytes)


async def process_scene(config: PipelineConfig, dataset: str, scene: str) -> Path:
    scene_data = load_scene(config.data_root, dataset, scene, config.mask_subdir)
    out_dir = config.scene_output_dir(dataset, scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / _model_to_filename(config.model_name)
    error_log = out_dir / "errors.log"
    debug_dir = out_dir / "debug_inputs"

    existing = _load_existing_results(output_json)
    if config.target_object_id is not None:
        if config.target_object_id not in scene_data.object_ids:
            raise ValueError(
                f"Object ID {config.target_object_id} not found in {dataset}/{scene}. "
                f"Available IDs: {scene_data.object_ids[:20]}{'...' if len(scene_data.object_ids) > 20 else ''}"
            )
        pending_ids = [config.target_object_id]
    else:
        pending_ids = [obj_id for obj_id in scene_data.object_ids if obj_id not in existing]
        if config.single_object_only and pending_ids:
            pending_ids = pending_ids[:1]
    if not pending_ids:
        return output_json

    client = VLMClient(config)
    new_results: List[ObjectResult] = []
    lock = asyncio.Lock()

    async def _process_one(obj_id: int) -> None:
        views = select_views_for_object(obj_id, scene_data, config)
        if not views:
            return
        batches = prepare_image_batches(views, config)
        if not batches:
            return

        if config.image_mode == "per_view":
            per_view_outputs = []
            for batch_index, batch in enumerate(batches):
                prompt = _build_prompt(
                    config,
                    dataset=dataset,
                    scene=scene,
                    object_id=obj_id,
                    batch=batch,
                )
                if config.save_debug_inputs:
                    _dump_debug_batch(
                        debug_dir,
                        object_id=obj_id,
                        batch_index=batch_index,
                        batch=batch,
                        prompt=prompt,
                    )
                text = await client.infer(prompt, batch.images)
                per_view_outputs.append(
                    {
                        "view_names": [img.view_name for img in batch.images],
                        "response": text,
                    }
                )
            response_payload: Any = per_view_outputs
        else:
            batch = batches[0]
            prompt = _build_prompt(
                config,
                dataset=dataset,
                scene=scene,
                object_id=obj_id,
                batch=batch,
            )
            if config.save_debug_inputs:
                _dump_debug_batch(
                    debug_dir,
                    object_id=obj_id,
                    batch_index=0,
                    batch=batch,
                    prompt=prompt,
                )
            response_payload = await client.infer(prompt, batch.images)

        result = ObjectResult(
            id=obj_id,
            response=response_payload,
            n_views=len(views),
            mode=config.image_mode,
        )
        async with lock:
            new_results.append(result)

    tasks = [asyncio.create_task(_process_one(obj_id)) for obj_id in pending_ids]
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"{dataset}/{scene}",
    ):
        try:
            await future
        except Exception as exc:  # noqa: BLE001
            _append_error_log(error_log, f"{dataset}/{scene}: {exc}")

    merged: Dict[int, Dict[str, Any]] = {**existing}
    for result in new_results:
        merged[result.id] = asdict(result)
    sorted_results = [merged[obj_id] for obj_id in sorted(merged)]
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)
    return output_json


def list_scenes(data_root: Path, dataset: str) -> List[str]:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_dir}")
    scenes = [p.name for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    return scenes


def list_datasets(data_root: Path) -> List[str]:
    datasets: List[str] = []
    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        has_scene = False
        for scene_dir in ds_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "images").exists() and (scene_dir / "sam").exists():
                has_scene = True
                break
        if has_scene:
            datasets.append(ds_dir.name)
    return datasets
