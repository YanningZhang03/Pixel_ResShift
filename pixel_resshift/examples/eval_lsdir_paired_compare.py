import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

try:
    import lpips
except Exception:
    lpips = None

from pixel_resshift.config import get_config
from pixel_resshift.dit import build_model
from pixel_resshift.mean_flow import PixelResShiftMeanFlow


def pil_to_tensor(path, size):
    image = Image.open(path).convert("RGB")
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
    gap = 6
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


def average(values):
    return float(np.mean(values)) if values else float("nan")


def id_from_gt(path):
    return path.name.replace("_gt.png", "")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="/data/share/LSDIR")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-n", type=int, default=16)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    root = Path(args.root)
    dirs = {
        "gt": root / "LSDIR_GT",
        "lq": root / "LSDIR_LQ",
        "SwinIR": root / "LASDIR_SwinIR",
        "ESRGAN": root / "LSDIR_ESRGAN",
        "OSEDiff": root / "LSDIR_OSEDiff",
        "OMGSR": root / "LSDIR_OMGSR",
    }
    gt_paths = sorted(dirs["gt"].glob("*_gt.png"))
    rng = random.Random(args.seed)
    rng.shuffle(gt_paths)

    records = []
    for gt_path in gt_paths:
        item_id = id_from_gt(gt_path)
        paired = {
            "gt": gt_path,
            "lq": dirs["lq"] / f"{item_id}_gt.png",
            "SwinIR": dirs["SwinIR"] / f"{item_id}_lq_SwinIR.png",
            "ESRGAN": dirs["ESRGAN"] / f"{item_id}_lq_out.png",
            "OSEDiff": dirs["OSEDiff"] / f"{item_id}_lq.png",
            "OMGSR": dirs["OMGSR"] / f"{item_id}_lq.png",
        }
        if all(path.exists() for path in paired.values()):
            records.append((item_id, paired))
        if args.max_samples > 0 and len(records) >= args.max_samples:
            break
    if not records:
        raise RuntimeError(f"No paired LSDIR records found under {root}")

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    config = get_config(args.config)
    config.meanflow.lpips = False
    config.meanflow.convnext = False
    device = torch.device(args.device)
    size = int(config.model.image_size)

    model = build_model(config).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    flow = PixelResShiftMeanFlow(model, config).to(device).eval()

    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    method_names = ["LQ", "SwinIR", "ESRGAN", "OSEDiff", "OMGSR", "PixelResShift-120k"]
    metric_store = {name: {"psnr": [], "ssim": [], "lpips": []} for name in method_names}
    csv_rows = []
    grid_rows = []

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        ids = [item_id for item_id, _ in batch_records]
        gt = torch.stack([pil_to_tensor(paths["gt"], size) for _, paths in batch_records]).to(device)
        lq = torch.stack([pil_to_tensor(paths["lq"], size) for _, paths in batch_records]).to(device)
        external = {
            name: torch.stack([pil_to_tensor(paths[name], size) for _, paths in batch_records]).to(device)
            for name in ["SwinIR", "ESRGAN", "OSEDiff", "OMGSR"]
        }
        pred = flow.restore(lq, y=None, add_noise=args.noise).clamp(-1, 1)
        outputs = {"LQ": lq, **external, "PixelResShift-120k": pred}

        method_metrics = {}
        for name, tensor in outputs.items():
            method_metrics[name] = {
                "psnr": psnr_rgb(tensor, gt),
                "ssim": ssim_rgb(tensor, gt),
                "lpips": (
                    lpips_model(tensor, gt).reshape(-1).detach().cpu().tolist()
                    if lpips_model is not None
                    else [float("nan")] * len(ids)
                ),
            }
            for key in ("psnr", "ssim", "lpips"):
                metric_store[name][key].extend(method_metrics[name][key])

        for row_idx, item_id in enumerate(ids):
            row = {"id": item_id}
            for name in method_names:
                for key in ("psnr", "ssim", "lpips"):
                    row[f"{name}_{key}"] = method_metrics[name][key][row_idx]
            csv_rows.append(row)

            global_idx = start + row_idx
            if global_idx < args.save_n:
                images = [
                    tensor_to_pil(gt[row_idx]),
                    tensor_to_pil(lq[row_idx]),
                    tensor_to_pil(external["SwinIR"][row_idx]),
                    tensor_to_pil(external["ESRGAN"][row_idx]),
                    tensor_to_pil(external["OSEDiff"][row_idx]),
                    tensor_to_pil(external["OMGSR"][row_idx]),
                    tensor_to_pil(pred[row_idx]),
                ]
                grid_rows.append(images)
                save_grid(
                    out_dir / "samples" / f"{global_idx:04d}_{item_id}.png",
                    [images],
                    ["GT", "LQ", "SwinIR", "ESRGAN", "OSEDiff", "OMGSR", "PRS-120k"],
                )

    summary = {
        "checkpoint": args.ckpt,
        "root": str(root),
        "num_samples": len(records),
        "image_size": size,
        "noise_endpoint": bool(args.noise),
        "metrics_rgb": {
            name: {
                "psnr": average(metric_store[name]["psnr"]),
                "ssim": average(metric_store[name]["ssim"]),
                "lpips_vgg": average(metric_store[name]["lpips"]),
            }
            for name in method_names
        },
    }

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_grid(out_dir / "images" / "comparison_grid.png", grid_rows, ["GT", "LQ", "SwinIR", "ESRGAN", "OSEDiff", "OMGSR", "PRS-120k"])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
