import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pixel_resshift.config import get_config
from pixel_resshift.dit import build_model
from pixel_resshift.mean_flow import PixelResShiftMeanFlow


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def list_images(root, max_samples, seed):
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix in IMAGE_EXTS)
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:max_samples]


def pil_to_tensor(path, size):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def to_01(x):
    return ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)


def tensor_to_pil(x):
    arr = (to_01(x).detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    return Image.fromarray(arr.astype(np.uint8))


def save_grid(path, rows, labels):
    width, height = rows[0][0].size
    gap = 8
    label_height = 24
    canvas = Image.new(
        "RGB",
        (len(labels) * width + (len(labels) - 1) * gap, label_height + len(rows) * height),
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


@torch.no_grad()
def restore_from_explicit_endpoint(flow, lq, add_noise):
    target_size = (int(flow.config.model.image_size), int(flow.config.model.image_size))
    c, x1 = flow.build_endpoint(lq, target_size=target_size, add_noise=add_noise)
    batch_size = lq.shape[0]
    device = lq.device
    y = torch.full((batch_size,), flow.num_classes, dtype=torch.long, device=device)
    t = torch.ones(batch_size, device=device)
    h = torch.ones(batch_size, device=device)
    omega = torch.ones(batch_size, device=device)
    t_min = torch.zeros(batch_size, device=device)
    t_max = torch.ones(batch_size, device=device)
    u, _ = flow.model(x1, t, h, omega, t_min, t_max, y, c)
    pred = (x1 - u).clamp(-1, 1)
    return c, x1, pred


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(args.config)
    config.meanflow.lpips = False
    config.meanflow.convnext = False
    device = torch.device(args.device)

    model = build_model(config).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    flow = PixelResShiftMeanFlow(model, config).to(device).eval()

    size = int(config.model.image_size)
    scale = int(getattr(config.degradation, "sf", 4))
    paths = list_images(args.source_root, args.num_samples, args.seed)
    rows = []
    stats = []

    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        gt = torch.stack([pil_to_tensor(path, size) for path in batch_paths]).to(device)
        lq = F.interpolate(
            gt,
            size=(size // scale, size // scale),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(-1, 1)
        lq_up = F.interpolate(lq, size=(size, size), mode="bicubic", align_corners=False).clamp(-1, 1)

        c_noise, x1_noise, pred_noise = restore_from_explicit_endpoint(flow, lq, add_noise=True)
        _, _, pred_clean = restore_from_explicit_endpoint(flow, lq, add_noise=False)

        for idx, path in enumerate(batch_paths):
            rows.append(
                [
                    tensor_to_pil(gt[idx]),
                    tensor_to_pil(lq_up[idx]),
                    tensor_to_pil(x1_noise[idx]),
                    tensor_to_pil(pred_noise[idx]),
                    tensor_to_pil(pred_clean[idx]),
                ]
            )
            stats.append(
                {
                    "path": str(path),
                    "mae_pred_noise_vs_lrup": float((pred_noise[idx] - lq_up[idx]).abs().mean().detach().cpu()),
                    "mae_pred_clean_vs_lrup": float((pred_clean[idx] - lq_up[idx]).abs().mean().detach().cpu()),
                    "mae_endpoint_noise_vs_lrup": float((x1_noise[idx] - lq_up[idx]).abs().mean().detach().cpu()),
                    "mae_pred_noise_vs_gt": float((pred_noise[idx] - gt[idx]).abs().mean().detach().cpu()),
                    "mae_pred_clean_vs_gt": float((pred_clean[idx] - gt[idx]).abs().mean().detach().cpu()),
                }
            )

    save_grid(
        out_dir / "noise_restore_grid.png",
        rows,
        ["GT", "LR-up", "noisy x1", "Pred add_noise=True", "Pred add_noise=False"],
    )
    with open(out_dir / "noise_restore_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(rows), "stats_mean": {
        key: float(np.mean([item[key] for item in stats])) for key in stats[0] if key != "path"
    }}, indent=2), flush=True)


if __name__ == "__main__":
    main()
