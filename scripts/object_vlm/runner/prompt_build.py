from ..config import PipelineConfig
from ..views.types import ImageBatch


def build_object_prompt(
    config: PipelineConfig,
    *,
    dataset: str,
    scene: str,
    object_id: int,
    batch: ImageBatch,
) -> str:
    view_names = [img.view_name for img in batch.images]
    return config.vlm.prompt_template.format(
        dataset=dataset,
        scene=scene,
        object_id=object_id,
        n_views=len(set(view_names)),
        view_names=", ".join(sorted(set(view_names))),
        image_mode=batch.mode,
    )
