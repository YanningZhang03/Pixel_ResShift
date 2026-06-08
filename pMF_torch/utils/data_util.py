"""Data utilities — pure PyTorch."""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

from utils.input_pipeline import center_crop_arr, worker_init_fn
from utils.logging_util import log_for_0
from torchvision.datasets.folder import pil_loader

NUM_CLASSES = 1000


def _get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def create_imagenet_dataloader(
    imagenet_root: str,
    split: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 4,
    for_fid: bool = False,
):
    """Create ImageNet DataLoader for FID / general use."""

    if for_fid:
        def fid_transform(pil_image):
            cropped = center_crop_arr(pil_image, image_size)
            return np.array(cropped)  # (H, W, C) uint8
        transform = fid_transform
    else:
        from torchvision import transforms
        def train_transform(pil_image):
            cropped = center_crop_arr(pil_image, image_size)
            arr = np.array(cropped)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return (t - 0.5) / 0.5
        transform = train_transform

    dataset = datasets.ImageFolder(
        os.path.join(imagenet_root, split),
        transform=transform,
        loader=pil_loader,
    )

    rank = _get_rank()
    world_size = _get_world_size()
    log_for_0(f"Dataset {split} (FID={for_fid}): {dataset}")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    return dataloader, len(sampler), len(dataset)
