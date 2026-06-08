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


def draw_center_text(draw, box, text, fill=(0, 0, 0)):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, fill=fill)


def save_tr_grid(path, zt_images, pred_images, t_values, r_values, source_name):
    """保存一张和示意图类似的 t-r 三角可视化。

    左边一列是真实构造出来的 z_t；右边三角区域是模型从 z_t 预测到 z_r 的结果。
    空白格表示 r > t，这些组合不符合从 t 往更干净时刻 r 推进的方向。
    """

    cell = zt_images[t_values[0]].size[0]
    gap = 8
    label_w = 72
    header_h = 48
    divider_gap = 20
    cols = 1 + len(r_values)
    rows = len(t_values)
    width = label_w + cols * cell + (cols - 1) * gap + divider_gap
    height = header_h + rows * cell + (rows - 1) * gap + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 6), source_name[:80], fill=(0, 0, 0))
    zt_x = label_w
    pred_x0 = label_w + cell + gap + divider_gap

    draw_center_text(draw, (zt_x, 20, zt_x + cell, header_h), r"$z_t$")
    for col, r_value in enumerate(r_values):
        x = pred_x0 + col * (cell + gap)
        draw_center_text(draw, (x, 20, x + cell, header_h), f"r = {r_value:.1f}")

    divider_x = label_w + cell + gap + divider_gap // 2
    draw.line((divider_x, header_h - 8, divider_x, height - 12), fill=(120, 120, 120), width=1)

    for row, t_value in enumerate(t_values):
        y = header_h + row * (cell + gap)
        draw_center_text(draw, (0, y, label_w - 8, y + cell), f"t = {t_value:.1f}")
        canvas.paste(zt_images[t_value], (zt_x, y))
        for col, r_value in enumerate(r_values):
            x = pred_x0 + col * (cell + gap)
            image = pred_images.get((t_value, r_value))
            if image is not None:
                canvas.paste(image, (x, y))

    canvas.save(path)


@torch.no_grad()
def predict_to_r(flow, z_t, c, t_value, r_value):
    """用平均速度从时刻 t 的状态预测到更干净的时刻 r。

    current restore 的特例是 t=1, r=0:
        x0_pred = z_1 - (1 - 0) * u(z_1, 0, 1)
    这里把它推广到任意 r <= t:
        z_r_pred = z_t - (t - r) * u(z_t, r, t)
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
    return (z_t - h.reshape(-1, 1, 1, 1) * u).clamp(-1, 1)


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

        zt_tensors = {}
        zt_images = {}
        pred_images = {}
        for t_value in t_values:
            # 与训练/推理保持一致：模型输入使用未裁剪的线性路径。
            # tensor_to_pil 会在保存可视化时裁剪，不能把显示裁剪提前施加到 z_t 张量上。
            z_t = (1.0 - t_value) * gt + t_value * x1
            zt_tensors[t_value] = z_t
            zt_images[t_value] = tensor_to_pil(z_t[0], args.thumb_size)

        for t_value in t_values:
            for r_value in r_values:
                if r_value > t_value + 1e-8:
                    continue
                if abs(r_value - t_value) < 1e-8:
                    pred = zt_tensors[t_value]
                else:
                    pred = predict_to_r(flow, zt_tensors[t_value], c, t_value, r_value)
                pred_images[(t_value, r_value)] = tensor_to_pil(pred[0], args.thumb_size)

        out_path = out_dir / f"sample_{idx:02d}_tr_grid.png"
        save_tr_grid(out_path, zt_images, pred_images, t_values, r_values, path.name)
        summary.append({"index": idx, "path": str(path), "grid": str(out_path)})
        print(f"saved {out_path}", flush=True)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_samples": len(summary),
                "t_values": t_values,
                "r_values": r_values,
                "prediction_formula": "z_r_pred = z_t - (t-r) * u(z_t, r, t)",
                "samples": summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
