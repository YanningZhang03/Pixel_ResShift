from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compat import ensure_pmf_torch_path


ensure_pmf_torch_path()

from models.embedder import BottleneckPatchEmbedder  # noqa: E402
from models.pmfDiT import pmfDiT as _BasePMFDiT  # noqa: E402
from models.torch_models import RMSNorm, TorchLinear  # noqa: E402


class ImageCrossAttention(nn.Module):
    """只让图像 patch token 向 LR 条件 token 做 cross-attention。

    pMF_torch 原始 DiT 里，前缀 token 保存类别、时间、CFG 区间等全局信息；
    这些 token 仍按原版 self-attention 传播。这里额外加入的 cross-attention
    只更新图像 token，语义更接近“每个当前图像 patch 去查询对应 LR 结构信息”。
    """

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

        self.query_norm = RMSNorm(hidden_size)
        self.context_norm = RMSNorm(hidden_size)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # 零初始化门控：刚开始等价于 pMF_torch 原主干，训练中再逐渐学会使用 LR 条件。
        self.cross_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        seq: torch.Tensor,
        context: torch.Tensor,
        prefix_tokens: int,
    ) -> torch.Tensor:
        prefix = seq[:, :prefix_tokens]
        image_tokens = seq[:, prefix_tokens:]

        batch, query_len, _ = image_tokens.shape
        context_len = context.shape[1]

        q = self.q_proj(self.query_norm(image_tokens))
        k = self.k_proj(self.context_norm(context))
        v = self.v_proj(self.context_norm(context))

        q = q.reshape(batch, query_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, context_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, context_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 与 pMF_torch 的 self-attention 一样，禁用 flash/mem-efficient kernel，
        # 避免 torch.func.jvp 在 forward-mode AD 下踩到不支持的 CUDA kernel。
        if q.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                attn = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).reshape(batch, query_len, self.hidden_size)
        image_tokens = image_tokens + self.out_proj(attn) * self.cross_scale
        return torch.cat([prefix, image_tokens], dim=1)


class ConditionalPMFDiT(_BasePMFDiT):
    """pMF_torch DiT with LR cross-attention condition.

    这里尽量不动 pMF_torch 的主干结构：
    - 时间 / CFG / 区间 / 类别 token 的构造完全沿用原版；
    - shared backbone 与 u/v 双头也完全沿用原版；
    - ``x`` 的 patch embedding 保持 3 通道，不再拼接 LR；
    - 额外把 LR 条件图 ``c`` 编成 context token，并在每层后用 cross-attention 注入。

    这版吸收了 pMF-ResShift-SR 的思路：当前图像 token 作为 Query，
    LR 条件 token 作为 Key/Value。输出头仍保持 ``x`` 的通道数，预测 HR 像素空间速度场。
    """

    def __init__(self, *args, cond_channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_channels = self.in_channels
        self.cond_channels = int(cond_channels)

        if self.cond_channels > 0:
            pca_channels = 128 if self.hidden_size <= 1024 else 256
            # LR 条件单独 patchify 成 context token；x_embedder 保持 pMF_torch 原样。
            self.cond_embedder = BottleneckPatchEmbedder(
                self.x_embedder.img_size[0],
                self.patch_size,
                pca_channels,
                self.cond_channels,
                self.hidden_size,
                bias=True,
            )
            cross_kwargs = dict(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                weight_init="scaled_variance",
                weight_init_constant=0.32,
            )
            self.shared_cross_blocks = nn.ModuleList(
                [ImageCrossAttention(**cross_kwargs) for _ in self.shared_blocks]
            )
            self.u_cross_blocks = nn.ModuleList(
                [ImageCrossAttention(**cross_kwargs) for _ in self.u_heads]
            )
            self.v_cross_blocks = nn.ModuleList(
                [ImageCrossAttention(**cross_kwargs) for _ in self.v_heads]
            )
        else:
            self.cond_embedder = None
            self.shared_cross_blocks = nn.ModuleList()
            self.u_cross_blocks = nn.ModuleList()
            self.v_cross_blocks = nn.ModuleList()

    def _build_context(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor | None:
        """把 LR 条件图编码成 cross-attention 的 Key/Value 序列。

        ``c=None`` 时使用全零条件图，保留接口兼容性。context 加上与图像 token
        相同的空间位置编码，让 cross-attention 能区分 LR 特征来自哪个位置。
        """

        if self.cond_channels <= 0:
            return None

        if c is None:
            c = x.new_zeros(x.shape[0], self.cond_channels, x.shape[2], x.shape[3])

        if c.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"Condition image size mismatch: got {c.shape[-2:]}, expected {x.shape[-2:]}."
            )
        if c.shape[1] != self.cond_channels:
            raise ValueError(
                f"Condition channel mismatch: got {c.shape[1]}, expected {self.cond_channels}."
            )

        context = self.cond_embedder(c)
        context = context + self.pos_embed[:, self.prefix_tokens:]
        return context

    def _build_sequence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor | None = None,
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

    def _apply_cross(
        self,
        seq: torch.Tensor,
        context: torch.Tensor | None,
        cross_block: ImageCrossAttention,
    ) -> torch.Tensor:
        if context is None:
            return seq
        return cross_block(seq, context, self.prefix_tokens)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._build_sequence(x, t, h, w, t_min, t_max, y, c=c)
        context = self._build_context(x, c)

        for idx, block in enumerate(self.shared_blocks):
            seq = block(seq, self.rope_freqs)
            if context is not None:
                seq = self._apply_cross(seq, context, self.shared_cross_blocks[idx])

        u_seq = v_seq = seq
        for idx, block in enumerate(self.u_heads):
            u_seq = block(u_seq, self.rope_freqs)
            if context is not None:
                u_seq = self._apply_cross(u_seq, context, self.u_cross_blocks[idx])

        for idx, block in enumerate(self.v_heads):
            v_seq = block(v_seq, self.rope_freqs)
            if context is not None:
                v_seq = self._apply_cross(v_seq, context, self.v_cross_blocks[idx])

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


cond_pmfDiT_B_16 = partial(
    ConditionalPMFDiT,
    input_size=256,
    depth=16,
    hidden_size=768,
    patch_size=16,
    num_heads=12,
    aux_head_depth=8,
)

cond_pmfDiT_B_32 = partial(
    ConditionalPMFDiT,
    input_size=512,
    depth=16,
    hidden_size=768,
    patch_size=32,
    num_heads=12,
    aux_head_depth=8,
)

cond_pmfDiT_L_16 = partial(
    ConditionalPMFDiT,
    input_size=256,
    depth=32,
    hidden_size=1024,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

cond_pmfDiT_L_32 = partial(
    ConditionalPMFDiT,
    input_size=512,
    depth=32,
    hidden_size=1024,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)

cond_pmfDiT_H_16 = partial(
    ConditionalPMFDiT,
    input_size=256,
    depth=48,
    hidden_size=1280,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

cond_pmfDiT_H_32 = partial(
    ConditionalPMFDiT,
    input_size=512,
    depth=48,
    hidden_size=1280,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)


_MODEL_FACTORY = {
    # 这些模型名与 pMF_torch/models/pmfDiT.py 里的 preset 一一对应。
    "pmfDiT_B_16": cond_pmfDiT_B_16,
    "pmfDiT_B_32": cond_pmfDiT_B_32,
    "pmfDiT_L_16": cond_pmfDiT_L_16,
    "pmfDiT_L_32": cond_pmfDiT_L_32,
    "pmfDiT_H_16": cond_pmfDiT_H_16,
    "pmfDiT_H_32": cond_pmfDiT_H_32,
}


def build_model(config):
    """Build a conditional pMF_torch DiT using the original pMF_torch size presets."""

    model_cfg = config.model
    model_str = model_cfg.model_str
    if model_str not in _MODEL_FACTORY:
        supported = ", ".join(sorted(_MODEL_FACTORY))
        raise ValueError(f"Unknown model_str '{model_str}'. Supported: {supported}")

    model_fn = _MODEL_FACTORY[model_str]
    # image_size 主要作为配置自描述字段保存；真正的结构尺寸由 pMF_torch preset 决定。
    return model_fn(
        in_channels=int(model_cfg.in_channels),
        cond_channels=int(model_cfg.cond_channels),
        num_classes=int(model_cfg.num_classes),
        eval_mode=bool(model_cfg.eval_mode),
    )
