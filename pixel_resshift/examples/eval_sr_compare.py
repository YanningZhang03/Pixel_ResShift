import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

try:
    import lpips
except Exception:
    lpips = None

from pixel_resshift.config import get_config
from pixel_resshift.dit import build_model
from pixel_resshift.mean_flow import PixelResShiftMeanFlow


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def list_images(root, max_samples, seed):
    paths = [p for p in Path(root).rglob("*") if p.suffix in IMAGE_EXTS]
    paths = sorted(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_samples > 0:
        paths = paths[:max_samples]
    return paths


def pil_to_tensor(image, size):
    image = image.convert("RGB")
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def to_01(x):
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)


def psnr_rgb(pred, gt):
    mse = (to_01(pred) - to_01(gt)).pow(2).flatten(1).mean(dim=1)
    return (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).detach().cpu().tolist()


def ssim_rgb(pred, gt):
    pred_np = to_01(pred).detach().cpu().permute(0, 2, 3, 1).numpy()
    gt_np = to_01(gt).detach().cpu().permute(0, 2, 3, 1).numpy()
    scores = []
    for pred_item, gt_item in zip(pred_np, gt_np):
        scores.append(
            float(
                structural_similarity(
                    gt_item,
                    pred_item,
                    channel_axis=2,
                    data_range=1.0,
                )
            )
        )
    return scores


def tensor_to_pil(x):
    arr = (to_01(x).detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    return Image.fromarray(arr.astype(np.uint8))


def save_grid(path, rows, labels):
    if not rows:
        return
    width, height = rows[0][0].size
    gap = 8
    label_height = 24
    canvas = Image.new(
        "RGB",
        (
            len(labels) * width + (len(labels) - 1) * gap,
            label_height + len(rows) * height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(labels):
        draw.text((col * (width + gap) + 4, 4), label, fill=(0, 0, 0))
    for row_idx, images in enumerate(rows):
        y = label_height + row_idx * height
        for col, image in enumerate(images):
            canvas.paste(image, (col * (width + gap), y))
    canvas.save(path)


def average(values):
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-n", type=int, default=24)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    config = get_config(args.config)
    # 推理不需要训练时的感知损失模块，避免额外加载 ConvNeXt/LPIPS 到 flow 里。
    config.meanflow.lpips = False
    config.meanflow.convnext = False
    device = torch.device(args.device)

    model = build_model(config).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    flow = PixelResShiftMeanFlow(model, config).to(device).eval()

    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    paths = list_images(args.source_root, args.max_samples, args.seed)
    if not paths:
        raise RuntimeError(f"No images found under {args.source_root}")

    all_rows = []
    metric_rows = []
    metrics = {
        "bicubic_psnr": [],
        "pred_psnr": [],
        "bicubic_ssim": [],
        "pred_ssim": [],
        "bicubic_lpips": [],
        "pred_lpips": [],
    }
    size = int(config.model.image_size)
    scale = int(getattr(config.degradation, "sf", 4))

    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        gt = torch.stack([pil_to_tensor(Image.open(p), size) for p in batch_paths]).to(device)
        lq = F.interpolate(
            gt,
            size=(size // scale, size // scale),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(-1, 1)
        bicubic = F.interpolate(lq, size=(size, size), mode="bicubic", align_corners=False).clamp(-1, 1)
        pred = flow.restore(lq, y=None, add_noise=args.noise).clamp(-1, 1)

        bicubic_psnr = psnr_rgb(bicubic, gt)
        pred_psnr = psnr_rgb(pred, gt)
        bicubic_ssim = ssim_rgb(bicubic, gt)
        pred_ssim = ssim_rgb(pred, gt)

        if lpips_model is not None:
            bicubic_lpips = lpips_model(bicubic, gt).reshape(-1).detach().cpu().tolist()
            pred_lpips = lpips_model(pred, gt).reshape(-1).detach().cpu().tolist()
        else:
            bicubic_lpips = [float("nan")] * len(batch_paths)
            pred_lpips = [float("nan")] * len(batch_paths)

        for idx, path in enumerate(batch_paths):
            record = {
                "path": str(path),
                "bicubic_psnr": bicubic_psnr[idx],
                "pred_psnr": pred_psnr[idx],
                "bicubic_ssim": bicubic_ssim[idx],
                "pred_ssim": pred_ssim[idx],
                "bicubic_lpips": bicubic_lpips[idx],
                "pred_lpips": pred_lpips[idx],
            }
            metric_rows.append(record)
            for key, value in record.items():
                if key in metrics and not math.isnan(float(value)):
                    metrics[key].append(float(value))

            global_idx = start + idx
            if global_idx < args.save_n:
                row = [
                    tensor_to_pil(gt[idx]),
                    tensor_to_pil(bicubic[idx]),
                    tensor_to_pil(pred[idx]),
                ]
                all_rows.append(row)
                row_path = out_dir / "samples" / f"{global_idx:04d}.png"
                save_grid(row_path, [row], ["GT", "LR-up/Bicubic", "PixelResShift-120k"])

    summary = {
        "checkpoint": args.ckpt,
        "source_root": args.source_root,
        "num_samples": len(metric_rows),
        "noise_endpoint": bool(args.noise),
        "bicubic": {
            "psnr_rgb": average(metrics["bicubic_psnr"]),
            "ssim_rgb": average(metrics["bicubic_ssim"]),
            "lpips_vgg": average(metrics["bicubic_lpips"]),
        },
        "pixel_resshift_120k": {
            "psnr_rgb": average(metrics["pred_psnr"]),
            "ssim_rgb": average(metrics["pred_ssim"]),
            "lpips_vgg": average(metrics["pred_lpips"]),
        },
    }

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_grid(out_dir / "images" / "comparison_grid.png", all_rows, ["GT", "LR-up/Bicubic", "PixelResShift-120k"])

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
