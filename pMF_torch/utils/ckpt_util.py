"""Checkpoint utilities — pure PyTorch."""

from __future__ import annotations

import os
import glob
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils.logging_util import log_for_0


def save_checkpoint(
    workdir: str,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    ema_params: dict[float, OrderedDict],
    keep: int = 3,
):
    """Save a training checkpoint."""
    os.makedirs(workdir, exist_ok=True)
    ckpt_path = os.path.join(workdir, f"checkpoint_{step}.pt")
    log_for_0("Saving checkpoint step %d.", step)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "ema_params": ema_params,
        },
        ckpt_path,
    )
    log_for_0("Checkpoint step %d saved to %s.", step, ckpt_path)

    # Clean up old checkpoints, keep only `keep` most recent
    existing = sorted(glob.glob(os.path.join(workdir, "checkpoint_*.pt")))
    while len(existing) > keep:
        old = existing.pop(0)
        os.remove(old)
        log_for_0("Removed old checkpoint: %s", old)


def restore_checkpoint(
    workdir: str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[float, OrderedDict]]:
    """Restore the latest checkpoint from workdir.

    Returns:
        (step, ema_params)
    """
    ckpt_files = sorted(glob.glob(os.path.join(workdir, "checkpoint_*.pt")))
    if not ckpt_files:
        log_for_0("No checkpoint found in %s", workdir)
        return 0, {}

    ckpt_path = ckpt_files[-1]
    log_for_0("Restoring checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = ckpt.get("step", 0)
    ema_params = ckpt.get("ema_params", {})
    log_for_0("Restored from checkpoint at step %d", step)
    return step, ema_params
