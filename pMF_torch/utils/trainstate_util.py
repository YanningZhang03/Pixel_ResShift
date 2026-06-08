"""Training state creation — pure PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.logging_util import log_for_0
from utils.state_util import print_params
from utils.ema_util import create_ema_state_dicts


def create_model_and_optimizer(config, model: nn.Module, lr_lambda, base_lr, device):
    """Create optimizer and EMA state for the model.

    Args:
        config: ConfigDict
        model: pixelMeanFlow module
        lr_lambda: callable for LambdaLR
        base_lr: base learning rate
        device: torch device

    Returns:
        (model, optimizer, scheduler, ema_params)
    """
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_for_0(f"Total trainable parameters: {param_count:,}")
    print_params(model)

    optimizer_name = getattr(config.training, "optimizer", "muon").lower()

    if optimizer_name == "muon":
        from utils.muon import MuonAdamW, partition_params
        muon_params, adam_params = partition_params(model)
        log_for_0(f"Muon optimizer: {len(muon_params)} Muon param groups, "
                  f"{len(adam_params)} AdamW param groups")
        optimizer = MuonAdamW(
            muon_params=muon_params,
            adam_params=adam_params,
            lr=base_lr,
            adam_lr=base_lr,
            momentum=0.95,
            adam_betas=(0.9, config.training.adam_b2),
        )
    elif optimizer_name == "adamw":
        log_for_0("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, config.training.adam_b2),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    ema_vals = config.training.ema_val
    if isinstance(ema_vals, (float, int)):
        ema_vals = [ema_vals]
    ema_params = create_ema_state_dicts(model, ema_vals)

    return model, optimizer, scheduler, ema_params
