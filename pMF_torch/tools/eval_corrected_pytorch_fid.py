"""Evaluate converted pMF checkpoints with the standard pytorch-fid Inception.

This script intentionally leaves the repository evaluator untouched.  The
current `utils/fid_util.py` uses torchvision's classification InceptionV3,
while the public ImageNet FID reference stats are computed in the FID Inception
feature space.  Mixing those two spaces can make good samples look terrible.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import _load_config  # noqa: E402
from pmf import generate, pixelMeanFlow  # noqa: E402
from utils.ckpt_util import restore_checkpoint  # noqa: E402
from utils.fid_util import compute_fid  # noqa: E402


def _get_ema_state(ema_params: dict, ema_key: int):
    """EMA key may be saved as int, float, or string depending on converter."""
    for key in (ema_key, float(ema_key), str(ema_key)):
        if key in ema_params:
            return ema_params[key]
    raise KeyError(f"EMA {ema_key} not found. Available keys: {list(ema_params.keys())}")


@torch.no_grad()
def _fid_features(model: InceptionV3, images_uint8: np.ndarray, device: torch.device) -> np.ndarray:
    """Extract pool3 FID features from uint8 NHWC samples."""
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    x = x.to(device, non_blocking=True)
    pred = model(x)[0]
    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(-1)
    return pred.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ema", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = _load_config(args.config)
    if args.num_samples is not None:
        config.fid.num_samples = args.num_samples
    if args.batch_size is not None:
        config.fid.device_batch_size = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"num_samples={config.fid.num_samples}, batch_size={config.fid.device_batch_size}", flush=True)

    pmf_model = pixelMeanFlow(**config.model.to_dict(), eval_mode=True).to(device)
    step, ema_params = restore_checkpoint(config.load_from, pmf_model, optimizer=None)
    pmf_model.load_state_dict(_get_ema_state(ema_params, args.ema))
    pmf_model.eval()
    print(f"loaded checkpoint step={step}, ema={args.ema}", flush=True)

    fid_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=True)
    fid_model = fid_model.to(device).eval()

    ref = np.load(config.fid.cache_ref)
    ref_mu = ref["ref_mu"] if "ref_mu" in ref.files else ref["mu"]
    ref_sigma = ref["ref_sigma"] if "ref_sigma" in ref.files else ref["sigma"]

    features = []
    total = int(config.fid.num_samples)
    batch_size = int(config.fid.device_batch_size)
    steps = math.ceil(total / batch_size)
    omega = float(config.sampling.omegas[0])
    t_min, t_max = config.sampling.interval[0]
    start = time.time()

    for step_idx in range(steps):
        cur_bsz = min(batch_size, total - step_idx * batch_size)
        images = generate(
            pmf_model,
            n_sample=cur_bsz,
            config=config,
            num_steps=config.sampling.num_steps,
            omega=omega,
            t_min=float(t_min),
            t_max=float(t_max),
            sample_idx=step_idx,
            device=device,
        )
        images_uint8 = images.permute(0, 2, 3, 1)
        images_uint8 = (127.5 * images_uint8 + 128.0).clamp(0, 255).to(torch.uint8)
        features.append(_fid_features(fid_model, images_uint8.cpu().numpy(), device))

        done = min(total, (step_idx + 1) * batch_size)
        if step_idx == 0 or (step_idx + 1) % 25 == 0 or done == total:
            elapsed = time.time() - start
            print(f"generated {done}/{total} elapsed_sec={elapsed:.1f}", flush=True)

    feats = np.concatenate(features, axis=0)[:total]
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    fid = compute_fid(mu, ref_mu, sigma, ref_sigma)
    print(f"CORRECTED_PYTORCH_FID samples={total} fid={fid:.6f}", flush=True)


if __name__ == "__main__":
    main()
