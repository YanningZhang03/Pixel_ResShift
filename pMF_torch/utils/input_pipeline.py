"""ImageNet input pipeline — pure PyTorch with DDP."""

from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.datasets.folder import pil_loader

from utils.logging_util import log_for_0


def _get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center cropping implementation from ADM."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def process_images(
    images: torch.Tensor,
    use_flip: bool = True,
) -> torch.Tensor:
    """Process a batch of uint8 images -> float32 [-1, 1] (NCHW).

    Args:
        images: (B, C, H, W) uint8 tensor
        use_flip: apply random horizontal flip

    Returns:
        (B, C, H, W) float32 normalized to [-1, 1]
    """
    x = images.float() / 255.0

    if use_flip:
        flip_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < 0.5
        x = torch.where(flip_mask, x.flip(-1), x)

    x = (x - 0.5) / 0.5
    return x


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_split(dataset_cfg, batch_size: int, split: str):
    """Create a DataLoader for an ImageNet split.

    Returns:
        (dataloader, steps_per_epoch)
    """
    rank = _get_rank()
    world_size = _get_world_size()
    image_size = dataset_cfg.image_size

    def loader_with_crop(path: str):
        img = pil_loader(path)
        img_cropped = center_crop_arr(img, image_size)
        arr = np.array(img_cropped)  # (H, W, C) uint8
        return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W) uint8

    root = os.path.join(dataset_cfg.root, split)

    ds = datasets.ImageFolder(
        root,
        transform=None,
        loader=loader_with_crop,
    )
    log_for_0(f"Dataset: {ds}")

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    it = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=dataset_cfg.num_workers,
        prefetch_factor=(
            dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
        ),
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
    return it, steps_per_epoch
