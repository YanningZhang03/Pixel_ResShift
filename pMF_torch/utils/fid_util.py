"""FID and Inception Score utilities — pure PyTorch.

Uses torchvision's InceptionV3 for feature extraction (matching pytorch-fid).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from utils.logging_util import log_for_0

# Inception V3 feature extractor matching pytorch-fid's approach
# We use the 2048-dim pool3 features for FID, and 1008-dim logits for IS.

_inception_model: nn.Module | None = None


def _load_inception(device: torch.device) -> nn.Module:
    """Load InceptionV3 with pool3 features + logits."""
    global _inception_model
    if _inception_model is not None:
        return _inception_model.to(device)

    log_for_0("Loading InceptionV3 for FID/IS evaluation...")
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _inception_model = model
    log_for_0("InceptionV3 loaded.")
    return model.to(device)


@torch.no_grad()
def _get_inception_features(
    images_uint8: np.ndarray,
    device: torch.device,
    batch_size: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract InceptionV3 features from uint8 NHWC images.

    Returns:
        features: (N, 2048) float64
        logits: (N, 1000) float64
    """
    model = _load_inception(device)
    N = len(images_uint8)

    all_features = []
    all_logits = []

    for i in range(0, N, batch_size):
        batch = images_uint8[i : i + batch_size]
        # (B, H, W, C) uint8 -> (B, C, H, W) float32 [-1, 1]
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 128.0) / 128.0

        # Forward through inception - need features from the pool before FC
        # We'll use a hook-based approach
        features = _forward_inception(model, x)
        logits = model.fc(features)

        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    return (
        np.concatenate(all_features).astype(np.float64),
        np.concatenate(all_logits).astype(np.float64),
    )


def _forward_inception(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract pool3 features (2048-d) from InceptionV3."""
    # Manually forward through inception layers to get pre-FC features
    # Following torchvision InceptionV3 structure
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
    x = torch.flatten(x, 1)  # (B, 2048)
    return x


def compute_fid(mu1, mu2, sigma1, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_1d(sigma1).astype(np.float64)
    sigma2 = np.atleast_1d(sigma2).astype(np.float64)

    diff = mu1 - mu2
    tr_covmean = np.sum(
        np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real
    )
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid


def compute_inception_score(logits: np.ndarray, splits: int = 10):
    rng = np.random.RandomState(2020)
    logits = logits[rng.permutation(logits.shape[0]), :]

    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    probs = probs.astype(np.float64)

    N = probs.shape[0]
    split_size = N // splits

    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1).mean()
        scores.append(np.exp(kl))

    scores = np.array(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


def get_reference(cache_path: str) -> dict:
    assert os.path.exists(cache_path), f"Cache file must exist: {cache_path}"
    log_for_0(f"Loading FID reference stats from {cache_path}")
    with np.load(cache_path) as data:
        if "ref_mu" in data:
            return {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
        raise NotImplementedError("Unsupported reference format")


def compute_stats(
    samples: np.ndarray,
    device: torch.device,
    batch_size: int = 200,
    fid_samples: int = 50000,
) -> dict:
    """Compute FID statistics from uint8 NHWC images."""

    feats, logits = _get_inception_features(samples, device, batch_size)

    # All-gather across DDP if needed
    if dist.is_initialized() and dist.get_world_size() > 1:
        feats_t = torch.from_numpy(feats).to(device)
        logits_t = torch.from_numpy(logits).to(device)

        gathered_feats = [torch.zeros_like(feats_t) for _ in range(dist.get_world_size())]
        gathered_logits = [torch.zeros_like(logits_t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_feats, feats_t)
        dist.all_gather(gathered_logits, logits_t)

        feats = torch.cat(gathered_feats).cpu().numpy()
        logits = torch.cat(gathered_logits).cpu().numpy()

    feats = feats[:fid_samples]
    logits = logits[:fid_samples]

    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)

    return {"mu": mu, "sigma": sigma, "logits": logits}
