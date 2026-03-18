import argparse
import os
from pathlib import Path

from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image


def build_color_lut(ids: np.ndarray) -> Dict[int, Tuple[int, int, int]]:
    """
    为每个实例 id 构建一个稳定的颜色查找表。
    id=0 视为背景，设为黑色。
    其它 id 使用一个简单的哈希生成 RGB 颜色，保证同一个 id 在所有视角一致。
    """
    unique_ids: List[int] = sorted(int(i) for i in np.unique(ids))
    lut: Dict[int, Tuple[int, int, int]] = {}

    for k in unique_ids:
        if k == 0:
            lut[k] = (0, 0, 0)
        else:
            # 简单确定性哈希，避免依赖额外库
            r = (k * 37) % 256
            g = (k * 73) % 256
            b = (k * 109) % 256
            # 避免太暗
            if r + g + b < 60:
                r = (r + 80) % 256
                g = (g + 80) % 256
                b = (b + 80) % 256
            lut[k] = (r, g, b)
    return lut


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    将单通道 id mask 转为三通道彩色图 (H, W, 3)，uint8。
    """
    if mask.ndim != 2:
        raise ValueError(f"expect 2D mask, got shape {mask.shape}")

    ids = mask.astype(np.int64)
    lut = build_color_lut(ids)

    h, w = ids.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for k, (r, g, b) in lut.items():
        color[ids == k] = (r, g, b)

    return color


def load_mask(mask_path: Path) -> np.ndarray:
    """
    加载 mask，假设是单通道图像，像素值为实例 id。
    """
    img = Image.open(mask_path)
    # 转为单通道
    img = img.convert("I") if img.mode not in ("I", "L") else img
    arr = np.array(img)
    return arr


def load_image_if_exists(
    images_dir: Path, stem: str
) -> Optional[Image.Image]:
    """
    尝试在 images 目录下找到与 mask 同名（不含扩展名）的 RGB 图像。
    """
    if not images_dir.exists():
        return None

    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            img = Image.open(p).convert("RGB")
            return img
    return None


def make_overlay(
    image: Image.Image,
    color_mask: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    将彩色 mask 叠加到原图上，返回 overlay 图像。
    """
    img = image
    h_m, w_m, _ = color_mask.shape
    if img.size != (w_m, h_m):
        img = img.resize((w_m, h_m), Image.BILINEAR)

    img_arr = np.array(img).astype(np.float32)
    mask_arr = color_mask.astype(np.float32)

    overlay = (1.0 - alpha) * img_arr + alpha * mask_arr
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def process_scene(
    root: Path,
    dataset_name: str,
    scene_name: str,
    overwrite: bool = True,
) -> None:
    """
    处理单个 scene：
    - 读取 sam/mask 下的所有 mask 图像
    - 输出彩色 mask 和（如有原图）overlay 到 debug/dataset_name/scene_name
    """
    scene_root = root / dataset_name / scene_name
    masks_dir = scene_root / "sam" / "mask"
    images_dir = scene_root / "images"

    if not masks_dir.exists():
        print(f"[Skip] masks dir not found: {masks_dir}")
        return

    out_dir = root / "debug" / dataset_name / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        [p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
    )
    if not mask_files:
        print(f"[Skip] no masks in {masks_dir}")
        return

    print(f"[Scene] {dataset_name}/{scene_name} | masks: {len(mask_files)} | output: {out_dir}")

    for mask_path in mask_files:
        stem = mask_path.stem

        color_out = out_dir / f"{stem}_mask_color.png"
        overlay_out = out_dir / f"{stem}_overlay.png"

        if not overwrite and color_out.exists() and overlay_out.exists():
            continue

        try:
            mask = load_mask(mask_path)
            color_mask = colorize_mask(mask)

            # 存彩色 mask
            Image.fromarray(color_mask).save(color_out)

            # 如有对应原图，再存 overlay
            img = load_image_if_exists(images_dir, stem)
            if img is not None:
                overlay_img = make_overlay(img, color_mask)
                overlay_img.save(overlay_out)
        except Exception as e:
            print(f"[Error] {mask_path}: {e}")


def auto_discover_scenes(root: Path):
    """
    自动遍历 root 下的所有 dataset/scene 目录，
    假设结构为:
        root/
          dataset_name/
            scene_name/
              images/
              sam/mask/
    """
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        # 排除 debug、outputs 等非数据集目录（按需扩展）
        if dataset_name in {"debug", "outputs", ".git", "__pycache__"}:
            continue

        for scene_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            scene_name = scene_dir.name
            process_scene(root, dataset_name, scene_name)


def main():
    parser = argparse.ArgumentParser(
        description="可视化 sam/mask 实例 id，输出到 debug/dataset_name/scene_name 下。"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="数据根目录（包含 dataset_name/scene_name 结构），默认当前目录。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="指定单个数据集名称（可选）。不指定则自动遍历所有数据集。",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="指定单个 scene 名称（可选）。需要同时指定 --dataset。",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="如输出文件已存在则跳过，不覆盖。",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()
    overwrite = not args.no_overwrite

    if args.dataset is not None and args.scene is not None:
        process_scene(root, args.dataset, args.scene, overwrite=overwrite)
    elif args.dataset is not None:
        # 只处理一个 dataset 下的所有 scene
        dataset_dir = root / args.dataset
        if not dataset_dir.exists():
            print(f"[Error] dataset not found: {dataset_dir}")
            return
        for scene_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            process_scene(root, args.dataset, scene_dir.name, overwrite=overwrite)
    else:
        # 自动遍历所有 dataset/scene
        auto_discover_scenes(root)


if __name__ == "__main__":
    main()

