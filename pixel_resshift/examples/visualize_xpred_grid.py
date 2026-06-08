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


def psnr_pair(x, y):
    # 这里用于衡量 pred(t=1,r=1) 和 LR-up 的距离；输入值域是 [-1, 1]。
    x01 = to_01(x)
    y01 = to_01(y)
    mse = (x01 - y01).pow(2).mean().clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).detach().cpu())


def draw_center_text(draw, box, text, fill=(0, 0, 0)):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, fill=fill)


def save_xpred_grid(path, reference_images, zt_images, xpred_images, t_values, r_values, source_name):
    """保存 x-pred 可视化。

    左边先给出 GT / LR-up 参考图，再给出构造出的 z_t；
    右边三角区域是直接预测的干净图：
        x_pred(z_t, r, t) = z_t - t * u(z_t, r, t)
    注意这里不是 z_t -> z_r 的中间状态，所以 r=t 的对角线也会是去噪预测图。
    """

    cell = zt_images[t_values[0]].size[0]
    gap = 8
    label_w = 72
    header_h = 48
    divider_gap = 20
    ref_cols = ["GT", "LR-up"]
    cols = len(ref_cols) + 1 + len(r_values)
    rows = len(t_values)
    width = label_w + cols * cell + (cols - 1) * gap + divider_gap
    height = header_h + rows * cell + (rows - 1) * gap + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 6), source_name[:80], fill=(0, 0, 0))
    ref_x0 = label_w
    zt_x = label_w + len(ref_cols) * (cell + gap)
    pred_x0 = zt_x + cell + gap + divider_gap

    for col, name in enumerate(ref_cols):
        x = ref_x0 + col * (cell + gap)
        draw_center_text(draw, (x, 20, x + cell, header_h), name)
    draw_center_text(draw, (zt_x, 20, zt_x + cell, header_h), "z_t")
    for col, r_value in enumerate(r_values):
        x = pred_x0 + col * (cell + gap)
        draw_center_text(draw, (x, 20, x + cell, header_h), f"r = {r_value:.1f}")

    divider_x = zt_x + cell + gap + divider_gap // 2
    draw.line((divider_x, header_h - 8, divider_x, height - 12), fill=(120, 120, 120), width=1)

    for row, t_value in enumerate(t_values):
        y = header_h + row * (cell + gap)
        draw_center_text(draw, (0, y, label_w - 8, y + cell), f"t = {t_value:.1f}")
        for col, name in enumerate(ref_cols):
            x = ref_x0 + col * (cell + gap)
            canvas.paste(reference_images[name], (x, y))
        canvas.paste(zt_images[t_value], (zt_x, y))
        for col, r_value in enumerate(r_values):
            x = pred_x0 + col * (cell + gap)
            image = xpred_images.get((t_value, r_value))
            if image is not None:
                canvas.paste(image, (x, y))

    canvas.save(path)


@torch.no_grad()
def predict_x0(flow, z_t, c, t_value, r_value):
    """直接输出 pMF 的 x-pred。

    训练里对应:
        pred_x = z_t - t * u

    这里的 r 只作为模型输入，控制网络预测哪一个平均速度场；
    输出仍然是干净端 x0 的预测，而不是中间状态 z_r。
    """

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
    summary = []

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

        zt_tensors = {}
        zt_images = {}
        xpred_images = {}
        xpred_tensors = {}
        for t_value in t_values:
            # z_t 必须和训练/推理保持同一条原始路径：
            #   z_t = (1 - t) * x0 + t * x1
            # 这里不要 clamp。当前配置 clip_endpoint=False，noisy x1 会超出 [-1, 1]；
            # 如果提前裁剪，t=1 的 z_t 就不再等于推理时真正送进模型的 noisy x1。
            # 后面的 tensor_to_pil 会在“显示图片”时单独裁剪，不影响模型输入张量。
            z_t = (1.0 - t_value) * gt + t_value * x1
            zt_tensors[t_value] = z_t
            zt_images[t_value] = tensor_to_pil(z_t[0], args.thumb_size)

        for t_value in t_values:
            for r_value in r_values:
                if r_value > t_value + 1e-8:
                    continue
                pred = predict_x0(flow, zt_tensors[t_value], c, t_value, r_value)
                xpred_tensors[(t_value, r_value)] = pred
                xpred_images[(t_value, r_value)] = tensor_to_pil(pred[0], args.thumb_size)

        out_path = out_dir / f"sample_{idx:02d}_xpred_grid.png"
        save_xpred_grid(out_path, reference_images, zt_images, xpred_images, t_values, r_values, path.name)
        pred_11 = xpred_tensors[(1.0, 1.0)]
        pred_11_mae_vs_lr = float((pred_11 - c).abs().mean().detach().cpu())
        pred_11_psnr_vs_lr = psnr_pair(pred_11, c)
        summary.append(
            {
                "index": idx,
                "path": str(path),
                "grid": str(out_path),
                "pred_t1_r1_mae_vs_lr_up": pred_11_mae_vs_lr,
                "pred_t1_r1_psnr_vs_lr_up": pred_11_psnr_vs_lr,
            }
        )
        print(f"saved {out_path}", flush=True)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_samples": len(summary),
                "t_values": t_values,
                "r_values": r_values,
                "prediction_formula": "x_pred = z_t - t * u(z_t, r, t)",
                "pred_t1_r1_vs_lr_up": {
                    "mae_mean": float(np.mean([item["pred_t1_r1_mae_vs_lr_up"] for item in summary])),
                    "psnr_mean": float(np.mean([item["pred_t1_r1_psnr_vs_lr_up"] for item in summary])),
                },
                "samples": summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
