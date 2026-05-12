from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from scripts.scene_io.loader import load_scene

from ..api.client import VLMClient
from ..config import PipelineConfig
from ..views.compose.build import prepare_image_batches
from ..views.selection import select_views_for_object
from . import debug_artifacts, ids, prompt_build, result_json, store


@dataclass(slots=True)
class ObjectResult:
    id: int
    response: Any
    n_views: int
    mode: str
    parsed_json: Optional[Dict[str, Any]] = field(default=None)


async def process_scene(config: PipelineConfig, dataset: str, scene: str) -> Path:
    scene_data = load_scene(
        config.scene_input.data_root,
        dataset,
        scene,
        mask_subdir=config.scene_input.mask_subdir,
        id_map_source=config.scene_input.id_map_source,
        image_subdir=config.scene_input.image_subdir,
        sam2_json_path=config.scene_input.sam2_json_path,
    )
    out_dir = config.scene_output_dir(dataset, scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / ids.model_output_filename(config.vlm.model_name)
    error_log = out_dir / "errors.log"
    debug_dir = out_dir / "debug_inputs"

    existing = store.load_existing_object_results(output_json)
    if config.run.target_object_id is not None:
        if config.run.target_object_id not in scene_data.object_ids:
            raise ValueError(
                f"Object ID {config.run.target_object_id} not found in {dataset}/{scene}. "
                f"Available IDs: {scene_data.object_ids[:20]}{'...' if len(scene_data.object_ids) > 20 else ''}"
            )
        pending_ids = [config.run.target_object_id]
    else:
        pending_ids = [obj_id for obj_id in scene_data.object_ids if obj_id not in existing]
        if config.run.single_object_only and pending_ids:
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

        batch_outputs: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches):
            pr = prompt_build.build_object_prompt(
                config,
                dataset=dataset,
                scene=scene,
                object_id=obj_id,
                batch=batch,
            )
            if config.run.save_debug_inputs:
                debug_artifacts.dump_debug_batch(
                    debug_dir,
                    object_id=obj_id,
                    batch_index=0,
                    batch=batch,
                    prompt=pr,
                )
            text = await client.infer(pr, batch.images)
            batch_outputs.append(
                {
                    "batch_index": batch_index,
                    "mode": batch.mode,
                    "view_names": [img.view_name for img in batch.images],
                    "response": text,
                }
            )
        if len(batch_outputs) == 1:
            response_payload: Any = batch_outputs[0]["response"]
        else:
            response_payload = batch_outputs

        parsed = None
        if isinstance(response_payload, str):
            parsed = result_json.try_parse_json_object(response_payload)

        result = ObjectResult(
            id=obj_id,
            response=response_payload,
            n_views=len(views),
            mode=batches[0].mode,
            parsed_json=parsed,
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
            store.append_error_line(error_log, f"{dataset}/{scene}: {exc}")

    merged: Dict[int, Dict[str, Any]] = {**existing}
    for result in new_results:
        entry: Dict[str, Any] = {
            "id": result.id,
            "instance_id": ids.make_instance_id(dataset, scene, result.id),
            "n_views": result.n_views,
            "mode": result.mode,
        }
        if result.parsed_json is not None:
            entry["response"] = result.parsed_json
        else:
            entry["response"] = result.response
            entry["parse_failed"] = True
        merged[result.id] = entry
    sorted_results = [merged[obj_id] for obj_id in sorted(merged)]
    store.write_results_list(output_json, sorted_results)
    return output_json
