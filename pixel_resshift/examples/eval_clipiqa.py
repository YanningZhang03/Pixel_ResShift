import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import pyiqa

from pixel_resshift.config import get_config
from pixel_resshift.dit import build_model
from pixel_resshift.mean_flow import PixelResShiftMeanFlow


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def list_images(root, max_samples, seed):
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix in IMAGE_EXTS)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_samples > 0:
        paths = paths[:max_samples]
    return paths


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


@torch.no_grad()
def restore_from_forced_noisy_endpoint(flow, lq):
    """和现有 ImageNet 评测保持一致：从加噪 LR 端点 x1 一步恢复。"""

    target_size = (int(flow.config.model.image_size), int(flow.config.model.image_size))
    c, x1 = flow.build_endpoint(lq, target_size=target_size, add_noise=True)
    batch_size = x1.shape[0]
    device = x1.device

    y = torch.full((batch_size,), flow.num_classes, dtype=torch.long, device=device)
    t = torch.ones(batch_size, device=device)
    h = torch.ones(batch_size, device=device)
    omega = torch.ones(batch_size, device=device)
    t_min = torch.zeros(batch_size, device=device)
    t_max = torch.ones(batch_size, device=device)

    u, _ = flow.model(x1, t, h, omega, t_min, t_max, y, c)
    return (x1 - u).clamp(-1, 1)


@torch.no_grad()
def metric_scores(metric, images, sub_batch_size):
    values = []
    for start in range(0, images.shape[0], sub_batch_size):
        batch = to_01(images[start : start + sub_batch_size])
        score = metric(batch)
        values.extend(score.reshape(-1).detach().cpu().tolist())
    return [float(v) for v in values]


def combine(root):
    root = Path(root)
    summaries = sorted(root.glob("shard*/summary_shard.json"))
    if not summaries:
        raise RuntimeError(f"No shard summaries found under {root}")

    total_n = 0
    sums = {"gt": 0.0, "bicubic": 0.0, "pred": 0.0}
    metric_name = None
    shards = []
    for path in summaries:
        with path.open() as f:
            data = json.load(f)
        shards.append(data)
        metric_name = metric_name or data["metric_name"]
        n = int(data["num_samples"])
        total_n += n
        for key in sums:
            sums[key] += float(data[key]["sum"])

    summary = {
        "metric_name": metric_name,
        "num_samples": total_n,
        "higher_is_better": True,
        "gt": sums["gt"] / total_n,
        "bicubic": sums["bicubic"] / total_n,
        "pred": sums["pred"] / total_n,
        "notes": {
            "sr_eval": "GT=center-crop-resize 256, LR=synthetic bicubic x4, pred=forced noisy endpoint restore",
            "metric_impl": "pyiqa",
        },
        "shards": shards,
    }
    out_path = root / f"summary_{metric_name}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--ckpt")
    parser.add_argument("--source-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--metric-name", default="clipiqa")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iqa-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--combine-root")
    args = parser.parse_args()

    if args.combine_root:
        combine(args.combine_root)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = get_config(args.config)
    config.meanflow.lpips = False
    config.meanflow.convnext = False

    model = build_model(config).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    flow = PixelResShiftMeanFlow(model, config).to(device).eval()

    metric = pyiqa.create_metric(args.metric_name, device=device)
    metric.eval()

    paths = list_images(args.source_root, args.max_samples, args.seed)
    paths = paths[args.shard_index :: args.num_shards]
    if not paths:
        raise RuntimeError(f"No images found under {args.source_root}")

    size = int(config.model.image_size)
    scale = int(getattr(config.degradation, "sf", 4))
    sums = {"gt": 0.0, "bicubic": 0.0, "pred": 0.0}
    count = 0

    print(f"metric={args.metric_name} shard={args.shard_index}/{args.num_shards} images={len(paths)}", flush=True)
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
        bicubic = F.interpolate(lq, size=(size, size), mode="bicubic", align_corners=False).clamp(-1, 1)
        pred = restore_from_forced_noisy_endpoint(flow, lq)

        gt_scores = metric_scores(metric, gt, args.iqa_batch_size)
        bicubic_scores = metric_scores(metric, bicubic, args.iqa_batch_size)
        pred_scores = metric_scores(metric, pred, args.iqa_batch_size)
        sums["gt"] += float(np.sum(gt_scores))
        sums["bicubic"] += float(np.sum(bicubic_scores))
        sums["pred"] += float(np.sum(pred_scores))
        count += len(batch_paths)

        if args.progress_every > 0 and (count % args.progress_every == 0 or count == len(paths)):
            print(f"shard={args.shard_index} processed={count}/{len(paths)}", flush=True)

    summary = {
        "metric_name": args.metric_name,
        "num_samples": count,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "higher_is_better": True,
        "gt": {"mean": sums["gt"] / count, "sum": sums["gt"]},
        "bicubic": {"mean": sums["bicubic"] / count, "sum": sums["bicubic"]},
        "pred": {"mean": sums["pred"] / count, "sum": sums["pred"]},
    }
    with (out_dir / "summary_shard.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
