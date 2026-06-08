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


def list_images(root, num_samples, seed):
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix in IMAGE_EXTS)
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:num_samples]


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


def tensor_to_pil(x, thumb_size):
    arr = (to_01(x).detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    image = Image.fromarray(arr.astype(np.uint8))
    if thumb_size is not None and image.size != (thumb_size, thumb_size):
        image = image.resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
    return image


def residual_to_pil(x, ref, thumb_size, gain):
    """把 |x - ref| 转成可视化图片。

    x/ref 的值域是 [-1, 1]，最大绝对差是 2。先除以 2 归一化到 [0, 1]，
    再乘 gain 放大。gain=4 表示 0.25 左右的像素差会显示到接近满亮。
    """

    diff = (x.detach().float() - ref.detach().float()).abs() * 0.5 * gain
    arr = (diff.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round()
    image = Image.fromarray(arr.astype(np.uint8))
    if thumb_size is not None and image.size != (thumb_size, thumb_size):
        image = image.resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
    return image


def mae_pair(x, ref):
    return float((x - ref).abs().mean().detach().cpu())


def psnr_pair(x, ref):
    x01 = to_01(x)
    ref01 = to_01(ref)
    mse = (x01 - ref01).pow(2).mean().clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).detach().cpu())


def draw_center_text(draw, box, text, fill=(0, 0, 0)):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, fill=fill)


def save_residual_grid(
    path,
    reference_images,
    residual_ref_image,
    residual_images,
    t_values,
    r_values,
    source_name,
    gain,
):
    """保存以 LR-up 为基准的残差网格。

    左边三列是 GT、LR-up、|GT - LR-up|；右边三角区域是
    |x_pred(z_t, r, t) - LR-up|。
    """

    cell = reference_images["GT"].size[0]
    gap = 8
    label_w = 72
    header_h = 48
    divider_gap = 20
    left_cols = ["GT", "LR-up", "|GT-LR|"]
    cols = len(left_cols) + len(r_values)
    rows = len(t_values)
    width = label_w + cols * cell + (cols - 1) * gap + divider_gap
    height = header_h + rows * cell + (rows - 1) * gap + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 6), f"{source_name[:70]}   residual gain x{gain:g}", fill=(0, 0, 0))
    left_x0 = label_w
    pred_x0 = label_w + len(left_cols) * (cell + gap) + divider_gap

    for col, name in enumerate(left_cols):
        x = left_x0 + col * (cell + gap)
        draw_center_text(draw, (x, 20, x + cell, header_h), name)
    for col, r_value in enumerate(r_values):
        x = pred_x0 + col * (cell + gap)
        draw_center_text(draw, (x, 20, x + cell, header_h), f"|pred-LR| r={r_value:.1f}")

    divider_x = left_x0 + len(left_cols) * cell + (len(left_cols) - 1) * gap + divider_gap // 2
    draw.line((divider_x, header_h - 8, divider_x, height - 12), fill=(120, 120, 120), width=1)

    for row, t_value in enumerate(t_values):
        y = header_h + row * (cell + gap)
        draw_center_text(draw, (0, y, label_w - 8, y + cell), f"t = {t_value:.1f}")
        canvas.paste(reference_images["GT"], (left_x0, y))
        canvas.paste(reference_images["LR-up"], (left_x0 + cell + gap, y))
        canvas.paste(residual_ref_image, (left_x0 + 2 * (cell + gap), y))
        for col, r_value in enumerate(r_values):
            image = residual_images.get((t_value, r_value))
            if image is not None:
                canvas.paste(image, (pred_x0 + col * (cell + gap), y))

    canvas.save(path)


@torch.no_grad()
def predict_x0(flow, z_t, c, t_value, r_value):
    batch = z_t.shape[0]
    device = z_t.device
    t = torch.full((batch,), float(t_value), device=device)
    r = torch.full((batch,), float(r_value), device=device)
    h = t - r
    omega = torch.ones(batch, device=device)
    t_min = torch.zeros(batch, device=device)
    t_max = torch.ones(batch, device=device)
    y = torch.full((batch,), flow.num_classes, dtype=torch.long, device=device)
    u, _ = flow.model(z_t, t, h, omega, t_min, t_max, y, c)
    return (z_t - t.reshape(-1, 1, 1, 1) * u).clamp(-1, 1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--thumb-size", type=int, default=128)
    parser.add_argument("--residual-gain", type=float, default=4.0)
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
    if not paths:
        raise RuntimeError(f"No images found under {args.source_root}")

    t_values = [1.0, 0.8, 0.5, 0.1]
    r_values = [1.0, 0.8, 0.5, 0.0]
    samples = []
    all_pred_lr_mae = []
    all_gt_lr_mae = []

    for idx, path in enumerate(paths):
        gt = pil_to_tensor(path, size).unsqueeze(0).to(device)
        lq = F.interpolate(
            gt,
            size=(size // scale, size // scale),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(-1, 1)
        c, x1 = flow.build_endpoint(lq, target_size=gt.shape[-2:], add_noise=True)

        reference_images = {
            "GT": tensor_to_pil(gt[0], args.thumb_size),
            "LR-up": tensor_to_pil(c[0], args.thumb_size),
        }
        residual_ref_image = residual_to_pil(gt[0], c[0], args.thumb_size, args.residual_gain)

        residual_images = {}
        residual_metrics = {}
        for t_value in t_values:
            z_t = (1.0 - t_value) * gt + t_value * x1
            for r_value in r_values:
                if r_value > t_value + 1e-8:
                    continue
                pred = predict_x0(flow, z_t, c, t_value, r_value)
                residual_images[(t_value, r_value)] = residual_to_pil(
                    pred[0], c[0], args.thumb_size, args.residual_gain
                )
                key = f"t{t_value:.1f}_r{r_value:.1f}"
                residual_metrics[key] = {
                    "pred_mae_vs_lr_up": mae_pair(pred, c),
                    "pred_psnr_vs_lr_up": psnr_pair(pred, c),
                }
                all_pred_lr_mae.append(residual_metrics[key]["pred_mae_vs_lr_up"])

        gt_lr_mae = mae_pair(gt, c)
        all_gt_lr_mae.append(gt_lr_mae)
        out_path = out_dir / f"sample_{idx:02d}_residual_grid.png"
        save_residual_grid(
            out_path,
            reference_images,
            residual_ref_image,
            residual_images,
            t_values,
            r_values,
            path.name,
            args.residual_gain,
        )
        samples.append(
            {
                "index": idx,
                "path": str(path),
                "grid": str(out_path),
                "gt_mae_vs_lr_up": gt_lr_mae,
                "gt_psnr_vs_lr_up": psnr_pair(gt, c),
                "pred_residuals": residual_metrics,
            }
        )
        print(f"saved {out_path}", flush=True)

    summary = {
        "num_samples": len(samples),
        "t_values": t_values,
        "r_values": r_values,
        "residual_gain": args.residual_gain,
        "residual_definition": "absolute RGB residual: abs(image - LR-up), visualized as abs(diff) / 2 * residual_gain",
        "gt_vs_lr_up_mae_mean": float(np.mean(all_gt_lr_mae)),
        "all_pred_vs_lr_up_mae_mean": float(np.mean(all_pred_lr_mae)),
        "samples": samples,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(samples)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
