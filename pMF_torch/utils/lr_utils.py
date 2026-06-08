"""Learning rate utilities for training — pure PyTorch."""

import math
from torch.optim.lr_scheduler import LambdaLR


def make_warmup_const_schedule(base_lr: float, warmup_epochs: int, steps_per_epoch: int):
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return lr_lambda, base_lr


def make_warmup_cosine_schedule(
    base_lr: float,
    warmup_epochs: int,
    steps_per_epoch: int,
    total_epochs: int,
    lr_min_factor: float = 0.0,
):
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = int(total_epochs * steps_per_epoch)
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_factor + (1.0 - lr_min_factor) * cosine

    return lr_lambda, base_lr


def lr_schedules(config, steps_per_epoch):
    """Build LR schedule from config.

    Returns:
        (lr_lambda, base_lr) — use with LambdaLR:
            scheduler = LambdaLR(optimizer, lr_lambda)
    """
    base_lr = float(config.training.learning_rate)
    warmup_epochs = int(config.training.get("warmup_epochs", 0))
    schedule_kind = config.training.get("lr_schedule", "warmup_const")

    if schedule_kind == "warmup_const":
        return make_warmup_const_schedule(base_lr, warmup_epochs, steps_per_epoch)

    elif schedule_kind == "warmup_cosine":
        total_epochs = int(config.training.num_epochs)
        lr_min_factor = float(config.training.get("lr_min_factor", 0.0))
        return make_warmup_cosine_schedule(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            lr_min_factor=lr_min_factor,
        )

    else:
        raise ValueError(
            f"Unknown lr_schedule '{schedule_kind}'. "
            "Supported: 'warmup_const', 'warmup_cosine'."
        )
