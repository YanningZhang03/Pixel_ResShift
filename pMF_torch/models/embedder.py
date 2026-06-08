import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torch_models import TorchLinear, TorchEmbedding


class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        init_kwargs = dict(
            out_features=hidden_size,
            bias=True,
            weight_init=weight_init,
            init_constant=init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.embedding_table = TorchEmbedding(
            num_classes + 1,
            hidden_size,
            weight_init=weight_init,
            init_constant=init_constant,
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class BottleneckPatchEmbedder(nn.Module):
    """Image to Patch Embedding with a bottleneck Conv layer."""

    def __init__(
        self,
        input_size: int,
        initial_patch_size: int,
        pca_channels: int,
        in_channels: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (initial_patch_size, initial_patch_size)
        self.img_size = (input_size, input_size)
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj1 = nn.Conv2d(
            in_channels,
            pca_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.proj1.weight)
        if bias:
            nn.init.zeros_(self.proj1.bias)

        self.proj2 = nn.Conv2d(
            pca_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.proj2.weight)
        if bias:
            nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — PyTorch convention (NCHW)
        x = self.proj2(self.proj1(x))  # (B, hidden_size, grid_h, grid_w)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        return x
