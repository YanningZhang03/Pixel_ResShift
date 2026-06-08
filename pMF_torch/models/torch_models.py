from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchLinear(nn.Linear):
    """Linear layer with custom initialization matching the JAX original."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
        bias_init: str = "zeros",
    ):
        super().__init__(in_features, out_features, bias=bias)

        if weight_init == "scaled_variance":
            std = init_constant / sqrt(in_features)
            nn.init.normal_(self.weight, std=std)
        elif weight_init == "zeros":
            nn.init.zeros_(self.weight)
        else:
            raise ValueError(f"Invalid weight_init: {weight_init}")

        if bias and bias_init == "zeros":
            nn.init.zeros_(self.bias)
        elif bias and bias_init != "zeros":
            raise ValueError(f"Invalid bias_init: {bias_init}")


class TorchEmbedding(nn.Module):
    """Embedding layer with custom initialization matching the JAX original."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if weight_init is None:
            std = 0.02
        elif weight_init == "scaled_variance":
            std = init_constant / sqrt(embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {weight_init}")

        nn.init.normal_(self.embedding.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(mean_sq + self.eps)
        return (normed.to(x.dtype)) * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
    ):
        super().__init__()
        init_kwargs = dict(
            bias=False,
            weight_init=weight_init,
            init_constant=weight_init_constant,
        )
        self.w1 = TorchLinear(in_features, hidden_features, **init_kwargs)
        self.w3 = TorchLinear(in_features, hidden_features, **init_kwargs)
        self.w2 = TorchLinear(hidden_features, in_features, **init_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
