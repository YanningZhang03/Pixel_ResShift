"""
Muon optimizer — MomentUm Orthogonalized by Newton-schulz.

Reference: https://github.com/KellerJordan/Muon
           https://kellerjordan.github.io/posts/muon/

Muon runs standard SGD-momentum and then replaces each 2D parameter's update
with the nearest orthogonal matrix via Newton-Schulz iteration (5 steps in
bfloat16). For parameters that are not 2D hidden weights (embeddings, biases,
layernorms, output heads), it falls back to AdamW.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients maximize the slope at zero.
    Operates in bfloat16 for efficiency on tensor cores.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonAdamW(Optimizer):
    """Combined Muon + AdamW optimizer.

    Automatically partitions model parameters:
    - 2D hidden weight matrices (Linear layers excluding embeddings and output head)
      are optimized with Muon (SGD-momentum + Newton-Schulz orthogonalization).
    - All other parameters (biases, layernorms, embeddings, 1D params)
      are optimized with AdamW.

    Args:
        muon_params: Parameters to optimize with Muon (should be 2D weight matrices).
        adam_params: Parameters to optimize with AdamW.
        lr: Learning rate for Muon parameters.
        adam_lr: Learning rate for AdamW parameters. If None, uses ``lr``.
        momentum: Momentum coefficient for Muon (default: 0.95).
        adam_betas: Betas for AdamW (default: (0.9, 0.95)).
        adam_eps: Epsilon for AdamW (default: 1e-8).
        weight_decay: Weight decay for both Muon and AdamW.
        ns_steps: Number of Newton-Schulz iteration steps (default: 5).
        nesterov: Whether to use Nesterov momentum for Muon (default: True).
    """

    def __init__(
        self,
        muon_params,
        adam_params,
        lr: float = 0.02,
        adam_lr: float | None = None,
        momentum: float = 0.95,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        adam_lr = adam_lr if adam_lr is not None else lr
        muon_params = list(muon_params)
        adam_params = list(adam_params)

        param_groups = [
            dict(
                params=muon_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                nesterov=nesterov,
                use_muon=True,
            ),
            dict(
                params=adam_params,
                lr=adam_lr,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=weight_decay,
                use_muon=False,
            ),
        ]

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adam(group)

        return loss

    def _step_muon(self, group):
        lr = group["lr"]
        beta = group["momentum"]
        wd = group["weight_decay"]
        ns_steps = group["ns_steps"]
        nesterov = group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - beta)

            update = grad.lerp_(buf, beta) if nesterov else buf

            if update.ndim == 4:
                update = update.view(len(update), -1)

            update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
            scale = max(1, update.size(-2) / update.size(-1)) ** 0.5

            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr * scale)

    def _step_adam(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m = state["exp_avg"]
            v = state["exp_avg_sq"]

            m.lerp_(grad, 1 - beta1)
            v.lerp_(grad.square(), 1 - beta2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            if wd > 0:
                p.mul_(1 - lr * wd)
            p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


def partition_params(model: nn.Module):
    """Partition model parameters into Muon-eligible and Adam-eligible groups.

    Muon is applied to 2D weight matrices of hidden layers (Linear, but not
    embeddings or output heads). Everything else goes to AdamW.

    Returns:
        (muon_params, adam_params): Two lists of parameters.
    """
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embeddings and output heads should use Adam
        is_embedding = "embed" in name.lower()
        is_head = "head" in name.lower() or "final" in name.lower()

        if param.ndim >= 2 and not is_embedding and not is_head:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params
