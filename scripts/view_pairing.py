from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

IdMapSource = Literal["npy", "png", "sam2_json"]

# processed_infinigen_extracted: Image_0_0_0001_0.png <-> ObjectSegmentation_0_0_0001_0.png
PairingStrategy = Literal["stem", "infinigen"]

__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "IdMapSource",
    "PairingStrategy",
    "build_image_index",
    "list_mask_paths",
    "pair_image_and_masks",
]


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Map view stem -> first matching image path under image_dir."""
    image_index: Dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        image_index[path.stem] = path
    return image_index


def list_mask_paths(mask_dir: Path, id_map_source: IdMapSource) -> List[Path]:
    """List mask files in mask_dir for the given source type (no existence check)."""
    if id_map_source == "png":
        return sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if id_map_source == "npy":
        return sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy")
    raise ValueError(f"Unsupported id_map_source: {id_map_source}")


def _infinigen_image_pair_key(stem: str) -> str:
    prefix = "Image_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _infinigen_mask_pair_key(stem: str) -> str:
    prefix = "ObjectSegmentation_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def pair_image_and_masks(
    image_dir: Path,
    mask_paths: Sequence[Path],
    *,
    strategy: PairingStrategy = "stem",
) -> List[Tuple[str, Path, Path]]:
    """Pair each mask file to an image; view_name is the image stem.

    stem: mask stem must equal image stem (classic data_root/dataset/scene/images).
    infinigen: strip Image_ / ObjectSegmentation_ prefixes then match (processed_infinigen_extracted frames).
    """
    if strategy == "stem":
        image_index = build_image_index(image_dir)
        pairs: List[Tuple[str, Path, Path]] = []
        for map_path in mask_paths:
            view_name = map_path.stem
            image_path = image_index.get(view_name)
            if image_path is None:
                continue
            pairs.append((view_name, image_path, map_path))
        return pairs

    if strategy == "infinigen":
        key_to_image: Dict[str, Path] = {}
        for path in sorted(image_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            key = _infinigen_image_pair_key(path.stem)
            key_to_image[key] = path
        pairs = []
        for map_path in mask_paths:
            key = _infinigen_mask_pair_key(map_path.stem)
            image_path = key_to_image.get(key)
            if image_path is None:
                continue
            pairs.append((image_path.stem, image_path, map_path))
        return pairs

    raise ValueError(f"Unsupported pairing strategy: {strategy!r}")
