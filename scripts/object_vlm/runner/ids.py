def make_instance_id(dataset: str, scene: str, object_id: int) -> str:
    return f"{dataset}_{scene}_inst{object_id:03d}"


def model_output_filename(model_name: str) -> str:
    safe = model_name.replace("/", "__")
    return f"{safe}.json"
