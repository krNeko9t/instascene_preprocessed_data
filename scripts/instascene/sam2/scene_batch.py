"""Batch SAM2 id-map generation for scene layout and manifest targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from instascene.manifest.io import load_scene_paths_manifest, resolve_scene_paths
from instascene.scene.pairing import build_image_index
from sam2_idmap.id_map import atomic_npy_save, masks_to_id_map
from sam2_idmap.predictor import (
    Sam2IdMapConfig,
    SAM2AutomaticMaskGenerator,
    build_mask_generator,
    generate_masks,
    load_rgb_image,
)

__all__ = [
    "BatchResult",
    "SceneResult",
    "SceneTarget",
    "iter_targets_from_layout",
    "iter_targets_from_manifest",
    "process_scene",
    "run_batch",
]


@dataclass(slots=True)
class SceneTarget:
    label: str
    scene_root: Path
    image_dir: Path
    id_map_dir: Path


@dataclass(slots=True)
class SceneResult:
    label: str
    processed: int = 0
    skipped: int = 0
    total_images: int = 0
    error: str | None = None


@dataclass(slots=True)
class BatchResult:
    scenes: list[SceneResult] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return sum(s.processed for s in self.scenes)

    @property
    def skipped(self) -> int:
        return sum(s.skipped for s in self.scenes)

    @property
    def errors(self) -> int:
        return sum(1 for s in self.scenes if s.error is not None)


def _list_dataset_names(data_root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        names.append(path.name)
    return names


def _list_scene_names(dataset_dir: Path, image_subdir: str) -> list[str]:
    scenes: list[str] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (path / image_subdir).is_dir():
            scenes.append(path.name)
    return scenes


def iter_targets_from_layout(
    data_root: Path,
    dataset: str,
    scene: str,
    image_subdir: str = "images",
) -> list[SceneTarget]:
    root = data_root.expanduser().resolve()
    datasets = _list_dataset_names(root) if dataset == "all" else [dataset]
    targets: list[SceneTarget] = []

    for ds in datasets:
        dataset_dir = root / ds
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_dir}")
        scene_names = (
            _list_scene_names(dataset_dir, image_subdir)
            if scene == "all"
            else [scene]
        )
        for sc in scene_names:
            scene_root = dataset_dir / sc
            image_dir = scene_root / image_subdir
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
            targets.append(
                SceneTarget(
                    label=f"{ds}/{sc}",
                    scene_root=scene_root.resolve(),
                    image_dir=image_dir.resolve(),
                    id_map_dir=(scene_root / "id_maps").resolve(),
                )
            )
    return targets


def iter_targets_from_manifest(manifest_path: Path) -> list[SceneTarget]:
    doc = load_scene_paths_manifest(manifest_path.expanduser().resolve())
    targets: list[SceneTarget] = []
    for entry in doc.scenes:
        resolved = resolve_scene_paths(entry, doc.dataset_root)
        targets.append(
            SceneTarget(
                label=resolved.scene_key,
                scene_root=resolved.scene_root,
                image_dir=resolved.image_dir,
                id_map_dir=(resolved.scene_root / "id_maps").resolve(),
            )
        )
    return targets


def _sorted_image_paths(image_dir: Path) -> list[Path]:
    index = build_image_index(image_dir)
    return [index[stem] for stem in sorted(index)]


def process_scene(
    target: SceneTarget,
    mask_generator: SAM2AutomaticMaskGenerator,
    *,
    overwrite: bool = False,
    resolution: int = -1,
    max_images: int = -1,
    show_progress: bool = True,
) -> SceneResult:
    result = SceneResult(label=target.label)
    try:
        if not target.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir is not a directory: {target.image_dir}")

        image_paths = _sorted_image_paths(target.image_dir)
        if not image_paths:
            raise RuntimeError(f"No images found in: {target.image_dir}")
        if max_images > 0:
            image_paths = image_paths[:max_images]

        result.total_images = len(image_paths)
        target.id_map_dir.mkdir(parents=True, exist_ok=True)

        iterator: tqdm | list[Path]
        if show_progress:
            iterator = tqdm(image_paths, desc=target.label, leave=False)
        else:
            iterator = image_paths

        for image_path in iterator:
            output_path = target.id_map_dir / f"{image_path.stem}.npy"
            if not overwrite and output_path.exists():
                result.skipped += 1
                continue

            image = load_rgb_image(str(image_path), resolution=resolution)
            masks = generate_masks(mask_generator, image)
            id_map = masks_to_id_map(masks, image.shape[0], image.shape[1])
            atomic_npy_save(str(output_path), id_map)
            result.processed += 1
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
    return result


def run_batch(
    targets: list[SceneTarget],
    sam2_config: Sam2IdMapConfig,
    *,
    overwrite: bool = False,
    resolution: int = -1,
    max_images: int = -1,
    fail_fast: bool = False,
) -> BatchResult:
    if not targets:
        raise ValueError("No scene targets to process")

    mask_generator = build_mask_generator(sam2_config)
    batch = BatchResult()

    for target in tqdm(targets, desc="scenes"):
        scene_result = process_scene(
            target,
            mask_generator,
            overwrite=overwrite,
            resolution=resolution,
            max_images=max_images,
            show_progress=len(targets) == 1,
        )
        batch.scenes.append(scene_result)
        if scene_result.error and fail_fast:
            break

    return batch
