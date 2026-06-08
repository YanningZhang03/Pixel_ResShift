"""Sampling and FID evaluation utilities — pure PyTorch."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.distributed as dist

from pmf import pixelMeanFlow, generate
from utils import fid_util
from utils.logging_util import log_for_0


@torch.no_grad()
def run_sample(
    model: pixelMeanFlow,
    n_sample: int,
    config,
    device: torch.device,
    sample_idx: int | None = None,
    omega: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    ema_state_dict=None,
) -> np.ndarray:
    """Generate samples and return uint8 NHWC numpy array."""
    orig_state = None
    if ema_state_dict is not None:
        orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state_dict)

    images = generate(
        model,
        n_sample=n_sample,
        config=config,
        num_steps=config.sampling.num_steps,
        omega=omega,
        t_min=t_min,
        t_max=t_max,
        sample_idx=sample_idx,
        device=device,
    )

    if orig_state is not None:
        model.load_state_dict(orig_state)

    # (B, C, H, W) float -> (B, H, W, C) uint8
    images = images.permute(0, 2, 3, 1)
    images = (127.5 * images + 128.0).clamp(0, 255).to(torch.uint8)

    assert not torch.any(torch.isnan(images.float())), "NaN in samples!"
    return images.cpu().numpy()


def generate_fid_samples(
    model: pixelMeanFlow,
    config,
    device: torch.device,
    ema_state_dict=None,
    omega: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> np.ndarray:
    """Generate samples for FID evaluation."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    bsz = config.fid.device_batch_size

    num_steps = math.ceil(config.fid.num_samples / bsz / world_size)

    samples_all = []
    log_for_0("Generating FID samples (first batch may be slower)...")

    for step in range(num_steps):
        sample_idx = rank + world_size * step
        log_for_0(f"  Sampling step {step}/{num_steps}...")
        samples = run_sample(
            model, bsz, config, device,
            sample_idx=sample_idx,
            omega=omega, t_min=t_min, t_max=t_max,
            ema_state_dict=ema_state_dict,
        )
        samples_all.append(samples)

    return np.concatenate(samples_all, axis=0)


def get_fid_evaluator(config, writer, model, device):
    """Create FID evaluator closure."""
    stats_ref = fid_util.get_reference(config.fid.cache_ref)

    def _evaluate_one_mode(ema_state_dict=None, ema_key=None, **kwargs):
        omega = kwargs.get("omega", 1.0)
        t_min_val = kwargs.get("t_min", 0.0)
        t_max_val = kwargs.get("t_max", 1.0)

        samples = generate_fid_samples(
            model, config, device,
            ema_state_dict=ema_state_dict,
            omega=omega, t_min=t_min_val, t_max=t_max_val,
        )

        stats = fid_util.compute_stats(samples, device)

        mode_str = f"ema_{ema_key}" if ema_key is not None else "online"
        descriptor = f"omega_{omega:.2f}_tmin_{t_min_val:.2f}_tmax_{t_max_val:.2f}_{mode_str}"

        log_for_0(f"Computing FID/IS: {descriptor}")
        fid = fid_util.compute_fid(
            stats_ref["mu"], stats["mu"], stats_ref["sigma"], stats["sigma"]
        )
        is_score, _ = fid_util.compute_inception_score(stats["logits"])

        metric = {
            f"FID_{descriptor}": fid,
            f"IS_{descriptor}": is_score,
        }
        log_for_0(f"FID ({descriptor}): {fid:.4f}, IS: {is_score:.4f}")
        return metric, fid, is_score

    def evaluator(step, ema_params, ema_only=False, **kwargs):
        metric_dict = {}
        ema_key = kwargs.pop("ema", None)
        ema_list = [ema_key] if ema_key is not None else list(ema_params.keys())

        fid, is_score = None, None
        for ek in ema_list:
            metric, fid, is_score = _evaluate_one_mode(
                ema_state_dict=ema_params.get(ek), ema_key=ek, **kwargs
            )
            metric_dict.update(metric)

        if not ema_only:
            metric, fid, is_score = _evaluate_one_mode(**kwargs)
            metric_dict.update(metric)

        writer.write_scalars(step + 1, metric_dict)
        return fid, is_score

    return evaluator
