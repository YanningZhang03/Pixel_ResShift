import torch
from torch.optim.lr_scheduler import LambdaLR

from .compat import ensure_pmf_torch_path


ensure_pmf_torch_path()

from utils.lr_utils import lr_schedules  # noqa: E402
from utils.muon import MuonAdamW, partition_params  # noqa: E402


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def create_optimizer_and_scheduler(config, model, steps_per_epoch):
    """Create optimizer and LR scheduler using pMF_torch utilities.

    这里尽量不重写 pMF_torch 的优化器逻辑：
    - 学习率日程直接复用 ``utils.lr_utils.lr_schedules``
    - Muon / AdamW 参数划分直接复用 ``utils.muon.partition_params``
    - MuonAdamW 也直接调用 pMF_torch 原实现
    """

    optimizer_name = str(config.training.optimizer).lower()
    lr_lambda, base_lr = lr_schedules(config, steps_per_epoch)

    if optimizer_name == "muon":
        muon_params, adam_params = partition_params(model)
        optimizer = MuonAdamW(
            muon_params=muon_params,
            adam_params=adam_params,
            lr=float(base_lr),
            adam_lr=float(base_lr),
            momentum=0.95,
            adam_betas=(0.9, float(config.training.adam_b2)),
            weight_decay=float(config.training.weight_decay),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(base_lr),
            betas=(0.9, float(config.training.adam_b2)),
            weight_decay=float(config.training.weight_decay),
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
