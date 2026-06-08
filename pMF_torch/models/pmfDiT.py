import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedder import TimestepEmbedder, LabelEmbedder, BottleneckPatchEmbedder
from models.torch_models import TorchLinear, RMSNorm, SwiGLUMlp


#################################################################################
#                   Modern Transformer Components with Vec Gates               #
#################################################################################


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        init_kwargs = dict(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            weight_init=weight_init,
            init_constant=weight_init_constant,
        )

        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        # (B, S, H, D) -> (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Force math SDPA on CUDA so forward-mode AD (used by torch.func.jvp)
        # does not route into flash-attention kernels that lack forward AD.
        if q.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                attn = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 8 / 3,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(
            hidden_size,
            num_heads=num_heads,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
        )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUMlp(
            hidden_size,
            mlp_hidden_dim,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
        )

        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning             #
#################################################################################


class pmfDiT(nn.Module):
    """
    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        aux_head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        eval_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.eval_mode = eval_mode

        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens

        self.x_embedder = BottleneckPatchEmbedder(
            input_size,
            patch_size,
            128 if hidden_size <= 1024 else 256,
            in_channels,
            hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=hidden_size,
            weight_init="scaled_variance",
            init_constant=embedding_init_constant,
        )

        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(num_classes, **embed_kwargs)

        token_std = token_init_constant / math.sqrt(hidden_size)
        self.time_tokens = nn.Parameter(torch.randn(1, num_time_tokens, hidden_size) * token_std)
        self.class_tokens = nn.Parameter(torch.randn(1, num_class_tokens, hidden_size) * token_std)
        self.omega_tokens = nn.Parameter(torch.randn(1, num_cfg_tokens, hidden_size) * token_std)
        self.t_min_tokens = nn.Parameter(torch.randn(1, num_interval_tokens, hidden_size) * token_std)
        self.t_max_tokens = nn.Parameter(torch.randn(1, num_interval_tokens, hidden_size) * token_std)

        total_tokens = (
            self.x_embedder.num_patches
            + num_class_tokens
            + num_cfg_tokens
            + 2 * num_interval_tokens
            + num_time_tokens
        )
        self.prefix_tokens = (
            num_class_tokens
            + num_cfg_tokens
            + 2 * num_interval_tokens
            + num_time_tokens
        )
        head_dim = hidden_size // num_heads

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs_2d(head_dim, self.x_embedder.num_patches),
            persistent=False,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, hidden_size) * 0.02)

        head_depth = aux_head_depth
        shared_depth = depth - head_depth

        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=weight_init_constant,
        )

        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth)]
        )

        self.v_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth if not eval_mode else 0)]
        )

        self.u_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if not eval_mode:
            self.v_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        else:
            self.v_final_layer = None

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nhpwqc", x)
        images = x.reshape(x.shape[0], h * p, w * p, c)
        # NHWC -> NCHW
        images = images.permute(0, 3, 1, 2)
        return images

    def _build_sequence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / w)
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + h_embed.unsqueeze(1)
        omega_tokens = self.omega_tokens + omega_embed.unsqueeze(1)
        t_min_tokens = self.t_min_tokens + t_min_embed.unsqueeze(1)
        t_max_tokens = self.t_max_tokens + t_max_embed.unsqueeze(1)
        class_tokens = self.class_tokens + y_embed.unsqueeze(1)

        seq = torch.cat(
            [
                class_tokens,
                omega_tokens,
                t_min_tokens,
                t_max_tokens,
                time_tokens,
                x_embed,
            ],
            dim=1,
        )

        seq = seq + self.pos_embed

        return seq

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._build_sequence(x, t, h, w, t_min, t_max, y)

        for block in self.shared_blocks:
            seq = block(seq, self.rope_freqs)

        u_seq = v_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs)

        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs)

        u_tokens = u_seq[:, self.prefix_tokens:]
        v_tokens = v_seq[:, self.prefix_tokens:]

        u = self.unpatchify(self.u_final_layer(u_tokens))
        if self.v_final_layer is not None:
            v = self.unpatchify(self.v_final_layer(v_tokens))
        else:
            v = torch.zeros_like(u)

        t = t.reshape(-1, 1, 1, 1)

        u = (x - u) / t.clamp(min=0.05, max=1.0)
        v = (x - v) / t.clamp(min=0.05, max=1.0)

        return u, v


#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs_2d(
    dim: int, seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    dim = dim // 2  # for 2d rotary embeddings
    T = int(seq_len**0.5)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(T, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", positions, freqs)
    freqs_w = torch.einsum("i,j->ij", positions, freqs)
    freqs_grid = torch.cat(
        [
            freqs_h[:, None, :].expand(-1, T, -1),
            freqs_w[None, :, :].expand(T, -1, -1),
        ],
        dim=-1,
    )  # (T, T, dim)
    real = torch.cos(freqs_grid).reshape(seq_len, dim)
    imag = torch.sin(freqs_grid).reshape(seq_len, dim)
    return torch.view_as_complex(torch.stack([real, imag], dim=-1))  # (seq_len, dim)


def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (B, S, H, D) where D = head_dim
    *leading, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*leading, D // 2, 2))
    # freqs_cis: (S, D//2) -> (1, S, 1, D//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    T = freqs_cis.shape[1]
    # Only apply RoPE to the last T tokens (patch tokens); prefix tokens are untouched.
    x_complex_out = x_complex.clone()
    x_complex_out[:, -T:, :] = x_complex[:, -T:, :] * freqs_cis
    x_out = torch.view_as_real(x_complex_out).reshape(*leading, D)
    return x_out.to(x.dtype)


#################################################################################
#                                   pMF Configs                                 #
#################################################################################

pmfDiT_B_16 = partial(
    pmfDiT,
    input_size=256,
    depth=16,
    hidden_size=768,
    patch_size=16,
    num_heads=12,
    aux_head_depth=8,
)

pmfDiT_B_32 = partial(
    pmfDiT,
    input_size=512,
    depth=16,
    hidden_size=768,
    patch_size=32,
    num_heads=12,
    aux_head_depth=8,
)

pmfDiT_L_16 = partial(
    pmfDiT,
    input_size=256,
    depth=32,
    hidden_size=1024,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_L_32 = partial(
    pmfDiT,
    input_size=512,
    depth=32,
    hidden_size=1024,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H_16 = partial(
    pmfDiT,
    input_size=256,
    depth=48,
    hidden_size=1280,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H_32 = partial(
    pmfDiT,
    input_size=512,
    depth=48,
    hidden_size=1280,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)
