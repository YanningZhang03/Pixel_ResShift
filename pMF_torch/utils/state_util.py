"""Utilities for printing model parameters — pure PyTorch."""

import torch.nn as nn

from utils.logging_util import log_for_0


def print_params(model: nn.Module):
    """Print all parameters in the model."""
    params = {name: p for name, p in model.named_parameters()}

    if not params:
        log_for_0("No parameters found.")
        return

    total_params = 0
    max_name = max(len(n) for n in params)
    max_shape = max(len(str(p.shape)) for p in params.values())
    max_digits = max(len(f"{p.numel():,}") for p in params.values())
    log_for_0("-" * (max_name + max_digits + max_shape + 8))

    for name, param in params.items():
        n = param.numel()
        str_shape = str(param.shape).rjust(max_shape)
        str_n = f"{n:,}".rjust(max_digits)
        log_for_0(f" {name.ljust(max_name)} | {str_shape} | {str_n} ")
        total_params += n

    log_for_0("-" * (max_name + max_digits + max_shape + 8))
    log_for_0(f"Total parameters: {total_params:,}")
