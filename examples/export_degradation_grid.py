"""Export ResShift degradation and MeanFlow endpoint visualizations.

这个脚本只用于检查数据流水线，不会加载模型，也不会改训练状态。
输出内容：
- GT: 高清图，训练中的 x0 / t=0
- LR_native: ResShift 生成的原始低分辨率图
- LR_up: 上采样到 HR 尺寸的条件图 c
- endpoint_noise: MeanFlow 的 t=1 端点，即 LR_up + Gaussian noise
- path_t_*.png: 从 GT 到 endpoint 的线性路径 (1-t) * GT + t * endpoint
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixel_resshift_4.config import get_config
from pixel_resshift_4.data import ResShiftBatchPreparer, build_dataloader


def seed_everything(seed: int) -> None:
    """固定随机种子，让这次导出的样本可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resize_like_training(x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    """按训练里 prepare_condition 的方式把 LR 上采样到 HR 尺寸。"""

    if x.shape[-2:] == size:
        return x
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """把 [-1, 1] 的 CHW tensor 转成 RGB 图像；超出范围的噪声端点仅用于可视化时裁剪。"""

    image = image.detach().float().cpu().clamp(-1.0, 1.0)
    array = ((image + 1.0) * 127.5).round().to(torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def make_labeled_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    """把一个样本的多张图横向拼成一张带标签的图。"""

    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length.")

    width, height = images[0].size
    label_height = 28
    gap = 6
    canvas = Image.new("RGB", (len(images) * width + (len(images) - 1) * gap, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (image, label) in enumerate(zip(images, labels)):
        x_offset = index * (width + gap)
        draw.text((x_offset + 4, 6), label, fill=(0, 0, 0))
        canvas.paste(image.resize((width, height), Image.BICUBIC), (x_offset, label_height))

    return canvas


def save_labeled_grid(path: Path, images: list[Image.Image], labels: list[str]) -> Image.Image:
    """保存横向拼图，并把生成的 PIL Image 返回给后续 overview 复用。"""

    canvas = make_labeled_grid(images, labels)
    canvas.save(path)
    return canvas


def save_stacked_rows(path: Path, rows: list[tuple[str, Image.Image]]) -> Image.Image:
    """把一个样本的两张拼图上下叠起来，方便快速检查一整组。"""

    max_width = max(image.size[0] for _, image in rows)
    row_label_height = 24
    gap = 10
    total_height = sum(image.size[1] + row_label_height for _, image in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    y_offset = 0
    for title, image in rows:
        draw.text((4, y_offset + 5), title, fill=(0, 0, 0))
        canvas.paste(image, (0, y_offset + row_label_height))
        y_offset += image.size[1] + row_label_height + gap
    canvas.save(path)
    return canvas


def export_samples(args: argparse.Namespace) -> Path:
    config = get_config(args.config)
    seed_everything(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 这个脚本只导出样本，不需要训练用的大 batch 和 DDP。
    config.training.micro_batch_size = min(args.num_samples, int(config.training.micro_batch_size))
    config.data.num_workers = args.num_workers
    config.data.pin_memory = False
    config.data.drop_last = False

    loader = build_dataloader(config, "train")
    preparer = ResShiftBatchPreparer(config, phase="train")
    device = torch.device(args.device)
    # CPU 上 DiffJPEG / interpolate 之后偶尔会产生非连续 tensor，
    # 而 ResShift 的 filter2D 内部使用 view；这里只在导出脚本里做 contiguous 包装。
    preparer._lazy_import_resshift(device)
    resshift_filter2d = preparer.filter2D
    preparer.filter2D = lambda image, kernel: resshift_filter2d(image.contiguous(), kernel)

    saved = 0
    overview_images: list[Image.Image] = []
    overview_labels: list[str] = []
    # 均匀取 t=0 到 t=1 的路径点，便于直接观察 HR -> noisy LR prior 的过渡。
    # 注意 GT / LR / endpoint 会单独保存；这里的 path 图仍包含 t=0 和 t=1，方便做动图或逐帧查看。
    if int(args.num_steps) <= 1:
        t_values = [1.0]
    else:
        t_values = np.linspace(0.0, 1.0, int(args.num_steps)).tolist()
    split_t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mode = config.meanflow.condition_upsample_mode
    noise_std = float(config.meanflow.endpoint_noise_std)

    for raw_batch in loader:
        batch = preparer.prepare(raw_batch, device)
        gt = batch["gt"].float()
        lq = batch["lq"].float()
        lr_up = resize_like_training(lq, gt.shape[-2:], mode=mode)
        endpoint = lr_up + torch.randn_like(lr_up) * noise_std

        for item_idx in range(gt.shape[0]):
            if saved >= args.num_samples:
                break

            sample_dir = output_dir / f"sample_{saved:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            gt_i = gt[item_idx]
            lq_i = lq[item_idx]
            lr_up_i = lr_up[item_idx]
            endpoint_i = endpoint[item_idx]

            tensor_to_pil(gt_i).save(sample_dir / "gt_t0.png")
            tensor_to_pil(lq_i).save(sample_dir / "lr_native.png")
            tensor_to_pil(lr_up_i).save(sample_dir / "lr_up_condition.png")
            tensor_to_pil(endpoint_i).save(sample_dir / "endpoint_t1_noise.png")

            if args.split_layout:
                # 用户检查用的固定两张图：
                # 1) GT / LR 条件 / t=1；其中 t=1 是加噪 endpoint，即 LR_up + noise。
                first_grid = save_labeled_grid(
                    sample_dir / "01_gt_lr_t1.png",
                    [tensor_to_pil(gt_i), tensor_to_pil(lr_up_i), tensor_to_pil(endpoint_i)],
                    ["GT", "LR", "t=1"],
                )

                transition_images = []
                transition_labels = []
                for t_value in split_t_values:
                    z_t = (1.0 - t_value) * gt_i + t_value * endpoint_i
                    transition = tensor_to_pil(z_t)
                    transition.save(sample_dir / f"transition_t{t_value:.1f}.png")
                    transition_images.append(transition)
                    transition_labels.append(f"t={t_value:.1f}")

                second_grid = save_labeled_grid(
                    sample_dir / "02_transition_t0_t1.png",
                    transition_images,
                    transition_labels,
                )
                group_image = save_stacked_rows(
                    sample_dir / "group.png",
                    [
                        ("01: GT / LR / t=1", first_grid),
                        ("02: t=0 -> t=1", second_grid),
                    ],
                )

                overview_images.append(group_image)
                overview_labels.append(f"sample_{saved:02d}")
                saved += 1
                continue

            grid_images = [
                tensor_to_pil(gt_i),
                tensor_to_pil(lr_up_i),
                tensor_to_pil(endpoint_i),
            ]
            grid_labels = ["GT t=0", "LR up c", "endpoint t=1"]

            for step_idx, t_value in enumerate(t_values):
                z_t = (1.0 - t_value) * gt_i + t_value * endpoint_i
                transition = tensor_to_pil(z_t)
                transition.save(sample_dir / f"path_{step_idx:02d}_t{t_value:.3f}.png")
                grid_images.append(transition)
                grid_labels.append(f"t={t_value:.2f}")

            save_labeled_grid(sample_dir / "grid.png", grid_images, grid_labels)

            # 总览图只放每个样本的一行：GT / LR_up / endpoint / t=0.1...1.0。
            overview_images.append(Image.open(sample_dir / "grid.png").convert("RGB"))
            overview_labels.append(f"sample_{saved:02d}")
            saved += 1

        if saved >= args.num_samples:
            break

    if saved == 0:
        raise RuntimeError("No samples were exported. Please check dataset path/config.")

    # 把多个 grid 纵向拼成一张 overview，方便快速扫一眼。
    max_width = max(image.size[0] for image in overview_images)
    total_height = sum(image.size[1] + 32 for image in overview_images)
    overview = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(overview)
    y_offset = 0
    for label, image in zip(overview_labels, overview_images):
        draw.text((4, y_offset + 6), label, fill=(0, 0, 0))
        overview.paste(image, (0, y_offset + 32))
        y_offset += image.size[1] + 32
    overview.save(output_dir / "overview.png")

    print(f"exported {saved} samples to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config yaml.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported PNGs.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10, help="Number of transition frames from t=0 to t=1.")
    parser.add_argument("--split-layout", action="store_true", help="Save each sample as two requested grids.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    export_samples(args)


if __name__ == "__main__":
    main()
