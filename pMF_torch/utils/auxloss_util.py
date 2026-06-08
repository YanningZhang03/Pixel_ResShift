"""Auxiliary perceptual losses — pure PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils.logging_util import log_for_0


def paired_random_resized_crop(
    x1: torch.Tensor,
    x2: torch.Tensor,
    out_size: int = 224,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same random resized crop to paired image batches.

    Args:
        x1, x2: (B, C, H, W) tensors.
    Returns:
        Cropped (B, C, out_size, out_size) tensors.
    """
    B, C, H, W = x1.shape
    y1_list, y2_list = [], []

    for i in range(B):
        import torchvision.transforms as T
        params = T.RandomResizedCrop.get_params(
            x1[i], scale=scale, ratio=ratio
        )
        top, left, crop_h, crop_w = params
        y1_list.append(TF.resized_crop(x1[i], top, left, crop_h, crop_w, [out_size, out_size]))
        y2_list.append(TF.resized_crop(x2[i], top, left, crop_h, crop_w, [out_size, out_size]))

    return torch.stack(y1_list), torch.stack(y2_list)


def init_auxloss(config, device: torch.device | str = "cuda"):
    """Initialize auxiliary loss function.

    Returns:
        auxloss_fn(model_images, gt_images) -> (lpips_dist, convnext_dist)
            both of shape (B,).
    """
    lpips_model = None
    convnext_model = None

    if config.model.lpips:
        log_for_0("Loading LPIPS model...")
        import lpips as lpips_lib
        lpips_model = lpips_lib.LPIPS(net="vgg").to(device)
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad = False
        param_count = sum(p.numel() for p in lpips_model.parameters())
        log_for_0(f"LPIPS model loaded with {param_count:,} parameters.")

    if config.model.convnext:
        log_for_0("Loading ConvNext feature extractor...")
        from models.convnext import load_convnext_model
        convnext_model = load_convnext_model(device)

    def auxloss_fn(
        model_images: torch.Tensor,
        gt_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = model_images.shape[0]
        dev = model_images.device

        crop_model, crop_gt = paired_random_resized_crop(
            model_images, gt_images, out_size=224
        )

        if lpips_model is not None:
            lpips_dist = lpips_model(crop_model, crop_gt).reshape(-1)
        else:
            lpips_dist = torch.zeros(bsz, device=dev)

        if convnext_model is not None:
            feat_model = convnext_model(crop_model)
            feat_gt = convnext_model(crop_gt)
            class_dist = (feat_model - feat_gt).pow(2).sum(dim=-1)
        else:
            class_dist = torch.zeros(bsz, device=dev)

        return lpips_dist, class_dist

    log_for_0("Auxiliary loss function initialized.")
    return auxloss_fn
