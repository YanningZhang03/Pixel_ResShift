"""Logging utilities — pure PyTorch with DDP."""

from __future__ import annotations

import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None


def _is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log_for_0(*args):
    if _is_main_process():
        logging.info(*args, stacklevel=2)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f"{self.elapse_with_reset():.2f} s"


class MetricsTracker:
    def __init__(self):
        self._sum: dict[str, float] | None = None
        self._n: int = 0

    def update(self, metrics_dict: dict[str, torch.Tensor | float]):
        local_mean = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                local_mean[k] = v.detach().float().item()
            else:
                local_mean[k] = float(v)

        if self._sum is None:
            self._sum = local_mean
        else:
            for k in local_mean:
                self._sum[k] = self._sum.get(k, 0.0) + local_mean[k]
        self._n += 1

    def finalize(self) -> dict[str, float]:
        if self._n == 0:
            return {}
        out = {k: v / self._n for k, v in self._sum.items()}
        self._sum, self._n = None, 0
        return out


class Writer:
    def __init__(self, config, workdir: str):
        self.workdir = workdir
        self.use_wandb = False
        if not _is_main_process():
            return

        self.use_wandb = getattr(config.logging, "use_wandb", False) and wandb is not None

        if self.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity or None,
                notes=config.logging.wandb_notes or None,
                tags=config.logging.wandb_tags or None,
                dir="/tmp",
                settings=wandb.Settings(_service_wait=60),
            )
            wandb.config.update(config.to_dict(), allow_val_change=True)
        else:
            log_for_0("Wandb logging is disabled. Images will be saved to disk.")

    def write_scalars(self, step: int, scalar_dict: dict):
        if not _is_main_process():
            return
        log_str = f"[{step}]"
        for k, v in scalar_dict.items():
            log_str += f" {k}={v:.5g}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        if self.use_wandb:
            wandb.log(scalar_dict, step=step)

    def write_images(self, step: int, image_dict: dict):
        if not _is_main_process():
            return

        def to_pil(v):
            if isinstance(v, Image.Image):
                return v
            if isinstance(v, np.ndarray):
                if v.dtype != np.uint8:
                    v = v.astype(np.uint8)
                if v.ndim == 3 and v.shape[0] == 3:
                    v = v.transpose(1, 2, 0)
                return Image.fromarray(v)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
                return to_pil(v)
            raise TypeError(f"Unsupported image type: {type(v)}")

        if self.use_wandb:
            wandb.log(
                {k: wandb.Image(to_pil(v)) for k, v in image_dict.items()},
                step=step,
            )
        else:
            img_dir = os.path.join(self.workdir, "images")
            os.makedirs(img_dir, exist_ok=True)
            for k, v in image_dict.items():
                to_pil(v).save(os.path.join(img_dir, f"{step}_{k}.png"))

    def __del__(self):
        if not _is_main_process():
            return
        if self.use_wandb and wandb is not None:
            wandb.finish()
            shutil.rmtree("/tmp/wandb", ignore_errors=True)
