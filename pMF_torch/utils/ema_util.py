"""Exponential Moving Average utilities — pure PyTorch."""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn as nn


def const_schedule(step: int, ema_value: float) -> float:
    return ema_value


def edm_schedule(step: int, ema_halflife_kimg: float) -> float:
    ema_halflife_nimg = ema_halflife_kimg * 1000
    ema_halflife_nimg = min(ema_halflife_nimg, step * 1024 * 0.05)
    ema_beta = 0.5 ** (1024 / max(ema_halflife_nimg, 1e-8))
    return ema_beta


def ema_schedules(config):
    ema_type = config.training.get("ema_type", "const")
    if ema_type == "const":
        return const_schedule
    elif ema_type == "edm":
        return edm_schedule
    else:
        raise ValueError(f"Unknown EMA type: {ema_type}")


@torch.no_grad()
def update_ema(
    ema_state_dict: OrderedDict,
    model_state_dict: OrderedDict,
    alpha: float,
) -> OrderedDict:
    """EMA update: ema = alpha * ema + (1 - alpha) * model."""
    new_ema = OrderedDict()
    for key in ema_state_dict:
        new_ema[key] = alpha * ema_state_dict[key] + (1 - alpha) * model_state_dict[key]
    return new_ema


def create_ema_state_dicts(
    model: nn.Module, ema_vals: list[float]
) -> dict[float, OrderedDict]:
    """Create initial EMA state dicts (deep copies of model)."""
    ema_params = {}
    for val in ema_vals:
        ema_params[val] = copy.deepcopy(model.state_dict())
    return ema_params
