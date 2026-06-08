import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

try:
    import pyiqa
except Exception:
    pyiqa = None

from pixel_resshift.config import get_config
from pixel_resshift.dit import build_model
from pixel_resshift.mean_flow import PixelResShiftMeanFlow


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
FEATURE_DIM = 2048


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


def build_inception(device):
    """构建和 pMF_torch 评测口径一致的 torchvision InceptionV3。"""

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def inception_pool3(model, images):
    """提取 pool3 特征，用于 FID。

    输入 images 为 [-1, 1] 的 NCHW Tensor。这里沿用 pMF_torch 的预处理：
    先转 uint8-like [0, 255]，再 resize 到 Inception 的 299 分辨率，并映射到 [-1, 1]。
    """

    x = to_01(images) * 255.0
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - 128.0) / 128.0

    x = model.Conv2d_1a_3x3(x)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = model.maxpool1(x)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = model.maxpool2(x)
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)
    x = model.avgpool(x)
    x = model.dropout(x)
    return torch.flatten(x, 1)


class FeatureAccumulator:
    """流式累计均值和协方差所需的一阶/二阶矩，避免保存 5 万张图片或全部特征。"""

    def __init__(self):
        self.n = 0
        self.sum = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.sumsq = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)

    def update(self, features):
        feats = features.detach().double().cpu().numpy()
        self.n += feats.shape[0]
        self.sum += feats.sum(axis=0)
        self.sumsq += feats.T @ feats

    def state_dict(self):
        return {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}


def stats_from_state(state):
    n = int(state["n"])
    mean = state["sum"] / n
    cov = (state["sumsq"] - np.outer(state["sum"], state["sum"]) / n) / max(n - 1, 1)
    return mean, cov


def fid_from_states(real_state, fake_state):
    mu1, sigma1 = stats_from_state(real_state)
    mu2, sigma2 = stats_from_state(fake_state)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm((sigma1 + np.eye(FEATURE_DIM) * eps) @ (sigma2 + np.eye(FEATURE_DIM) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def merge_states(states):
    out = {
        "n": 0,
        "sum": np.zeros(FEATURE_DIM, dtype=np.float64),
        "sumsq": np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64),
    }
    for state in states:
        out["n"] += int(state["n"])
        out["sum"] += state["sum"]
        out["sumsq"] += state["sumsq"]
    return out


@torch.no_grad()
def restore_from_forced_noisy_endpoint(flow, lq):
    """和现有评测保持一致：从加噪 LR 端点 x1 做一步恢复。"""

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
def musiq_scores(metric, images, sub_batch_size):
    if metric is None:
        return []
    scores = []
    for start in range(0, images.shape[0], sub_batch_size):
        batch = to_01(images[start : start + sub_batch_size])
        value = metric(batch)
        scores.extend(value.reshape(-1).detach().cpu().tolist())
    return [float(v) for v in scores]


def save_shard(out_dir, stats, musiq, args, paths_count):
    arrays = {
        "gt_n": np.array(stats["gt"].n, dtype=np.int64),
        "gt_sum": stats["gt"].sum,
        "gt_sumsq": stats["gt"].sumsq,
        "bicubic_n": np.array(stats["bicubic"].n, dtype=np.int64),
        "bicubic_sum": stats["bicubic"].sum,
        "bicubic_sumsq": stats["bicubic"].sumsq,
        "pred_n": np.array(stats["pred"].n, dtype=np.int64),
        "pred_sum": stats["pred"].sum,
        "pred_sumsq": stats["pred"].sumsq,
        "gt_musiq_sum": np.array(musiq["gt_sum"], dtype=np.float64),
        "bicubic_musiq_sum": np.array(musiq["bicubic_sum"], dtype=np.float64),
        "pred_musiq_sum": np.array(musiq["pred_sum"], dtype=np.float64),
        "musiq_n": np.array(musiq["n"], dtype=np.int64),
    }
    np.savez_compressed(out_dir / "stats.npz", **arrays)
    summary = {
        "processed": int(paths_count),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "musiq_available": bool(musiq["n"] > 0),
        "gt_musiq": musiq["gt_sum"] / musiq["n"] if musiq["n"] else None,
        "bicubic_musiq": musiq["bicubic_sum"] / musiq["n"] if musiq["n"] else None,
        "pred_musiq": musiq["pred_sum"] / musiq["n"] if musiq["n"] else None,
    }
    with (out_dir / "summary_shard.json").open("w") as f:
        json.dump(summary, f, indent=2)


def combine(root):
    paths = sorted(Path(root).glob("shard*/stats.npz"))
    if not paths:
        raise RuntimeError(f"No shard stats found under {root}")

    states = {name: [] for name in ("gt", "bicubic", "pred")}
    musiq = {"n": 0, "gt_sum": 0.0, "bicubic_sum": 0.0, "pred_sum": 0.0}
    for path in paths:
        data = np.load(path)
        for name in states:
            states[name].append(
                {
                    "n": int(data[f"{name}_n"]),
                    "sum": data[f"{name}_sum"],
                    "sumsq": data[f"{name}_sumsq"],
                }
            )
        musiq["n"] += int(data["musiq_n"])
        musiq["gt_sum"] += float(data["gt_musiq_sum"])
        musiq["bicubic_sum"] += float(data["bicubic_musiq_sum"])
        musiq["pred_sum"] += float(data["pred_musiq_sum"])

    merged = {name: merge_states(value) for name, value in states.items()}
    summary = {
        "num_samples": int(merged["gt"]["n"]),
        "fid": {
            "bicubic_vs_gt": fid_from_states(merged["gt"], merged["bicubic"]),
            "pred_vs_gt": fid_from_states(merged["gt"], merged["pred"]),
        },
        "musiq": {
            "gt": musiq["gt_sum"] / musiq["n"] if musiq["n"] else None,
            "bicubic": musiq["bicubic_sum"] / musiq["n"] if musiq["n"] else None,
            "pred": musiq["pred_sum"] / musiq["n"] if musiq["n"] else None,
            "num_samples": int(musiq["n"]),
            "higher_is_better": True,
        },
        "notes": {
            "fid_feature": "torchvision InceptionV3 pool3, pMF_torch preprocessing",
            "sr_eval": "GT=center-crop-resize 256, LR=synthetic bicubic x4, pred=forced noisy endpoint restore",
        },
    }
    with (Path(root) / "summary_fid_musiq.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--ckpt")
    parser.add_argument("--source-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--musiq-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--skip-musiq", action="store_true")
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

    inception = build_inception(device)
    musiq_metric = None
    if not args.skip_musiq:
        if pyiqa is None:
            raise RuntimeError("pyiqa is required for MUSIQ. Install pyiqa or pass --skip-musiq.")
        musiq_metric = pyiqa.create_metric("musiq", device=device)
        musiq_metric.eval()

    paths = list_images(args.source_root, args.max_samples, args.seed)
    paths = paths[args.shard_index :: args.num_shards]
    if not paths:
        raise RuntimeError(f"No images found under {args.source_root}")

    stats = {
        "gt": FeatureAccumulator(),
        "bicubic": FeatureAccumulator(),
        "pred": FeatureAccumulator(),
    }
    musiq = {"n": 0, "gt_sum": 0.0, "bicubic_sum": 0.0, "pred_sum": 0.0}

    size = int(config.model.image_size)
    scale = int(getattr(config.degradation, "sf", 4))
    print(f"shard={args.shard_index}/{args.num_shards} images={len(paths)}", flush=True)

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

        stats["gt"].update(inception_pool3(inception, gt))
        stats["bicubic"].update(inception_pool3(inception, bicubic))
        stats["pred"].update(inception_pool3(inception, pred))

        if musiq_metric is not None:
            gt_scores = musiq_scores(musiq_metric, gt, args.musiq_batch_size)
            bicubic_scores = musiq_scores(musiq_metric, bicubic, args.musiq_batch_size)
            pred_scores = musiq_scores(musiq_metric, pred, args.musiq_batch_size)
            musiq["n"] += len(gt_scores)
            musiq["gt_sum"] += float(np.sum(gt_scores))
            musiq["bicubic_sum"] += float(np.sum(bicubic_scores))
            musiq["pred_sum"] += float(np.sum(pred_scores))

        done = start + len(batch_paths)
        if args.progress_every > 0 and (done % args.progress_every == 0 or done == len(paths)):
            print(f"shard={args.shard_index} processed={done}/{len(paths)}", flush=True)

    save_shard(out_dir, stats, musiq, args, len(paths))
    print(f"saved {out_dir / 'stats.npz'}", flush=True)


if __name__ == "__main__":
    main()
