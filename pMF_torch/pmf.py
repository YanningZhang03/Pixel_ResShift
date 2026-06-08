"""pixel MeanFlow — pure PyTorch implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, functional_call

from models import pmfDiT


# ------------------------------------------------------------------
#  Sampling (inference)
# ------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: "pixelMeanFlow",
    n_sample: int,
    config,
    num_steps: int,
    omega: float,
    t_min: float,
    t_max: float,
    sample_idx: int | None = None,
    device: torch.device | str = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate samples from the model.

    Returns:
        images: (B, C, H, W) float32 tensor.
    """
    num_classes = config.dataset.num_classes
    img_size = config.dataset.image_size
    img_channels = config.dataset.image_channels

    x_shape = (n_sample, img_channels, img_size, img_size)
    z_t = torch.randn(x_shape, device=device, generator=generator) * model.noise_scale

    if sample_idx is not None:
        all_y = torch.arange(n_sample, dtype=torch.long, device=device)
        y = (all_y + sample_idx * n_sample) % num_classes
    else:
        y = torch.randint(0, num_classes, (n_sample,), device=device, generator=generator)

    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        z_t = model.sample_one_step(z_t, y, i, t_steps, omega, t_min, t_max)

    return z_t


# ------------------------------------------------------------------
#  Model
# ------------------------------------------------------------------


class pixelMeanFlow(nn.Module):
    """pixel MeanFlow — PyTorch version."""

    def __init__(
        self,
        model_str: str,
        num_classes: int = 1000,
        P_mean: float = -0.4,
        P_std: float = 1.0,
        cfg_max: float = 7.0,
        noise_scale: float = 1.0,
        data_proportion: float = 0.5,
        cfg_beta: float = 1.0,
        class_dropout_prob: float = 0.1,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        eval_mode: bool = False,
        lpips: bool = False,
        lpips_lambda: float = 1.0,
        convnext: bool = False,
        convnext_lambda: float = 0.0,
        perceptual_max_t: float = 1.0,
        tr_uniform: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.P_mean = P_mean
        self.P_std = P_std
        self.cfg_max = cfg_max
        self.noise_scale = noise_scale
        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.eval_mode = eval_mode
        self.use_lpips = lpips
        self.lpips_lambda = lpips_lambda
        self.use_convnext = convnext
        self.convnext_lambda = convnext_lambda
        self.perceptual_max_t = perceptual_max_t
        self.tr_uniform = tr_uniform

        net_fn = getattr(pmfDiT, model_str)
        self.net: pmfDiT.pmfDiT = net_fn(num_classes=num_classes, eval_mode=eval_mode)

    # ==============================================================
    #  Solver
    # ==============================================================

    def sample_one_step(
        self,
        z_t: torch.Tensor,
        labels: torch.Tensor,
        i: int,
        t_steps: torch.Tensor,
        omega: float,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]
        dev = z_t.device

        t_b = t.expand(bsz)
        r_b = r.expand(bsz)
        omega_b = torch.full((bsz,), omega, device=dev)
        t_min_b = torch.full((bsz,), t_min, device=dev)
        t_max_b = torch.full((bsz,), t_max, device=dev)

        u, _ = self.u_fn(z_t, t_b, t_b - r_b, omega_b, t_min_b, t_max_b, y=labels)

        dt = (t - r).reshape(1, 1, 1, 1)
        return z_t - dt * u

    # ==============================================================
    #  Schedule
    # ==============================================================

    def logit_normal_dist(self, bz: int, device: torch.device) -> torch.Tensor:
        rnd = torch.randn(bz, 1, 1, 1, device=device)
        return torch.sigmoid(rnd * self.P_std + self.P_mean)

    def sample_tr(self, bz: int, device: torch.device):
        t = self.logit_normal_dist(bz, device)
        r = self.logit_normal_dist(bz, device)

        if self.tr_uniform:
            unif_mask = torch.rand(bz, 1, 1, 1, device=device) < 0.1
            t = torch.where(unif_mask, torch.rand_like(t), t)
            r = torch.where(unif_mask, torch.rand_like(r), r)

        data_size = int(bz * self.data_proportion)
        fm_mask = torch.arange(bz, device=device) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = torch.where(fm_mask, t, r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz: int, device: torch.device, s_max: float = 7.0):
        u = torch.rand(bz, 1, 1, 1, device=device)

        if self.cfg_beta == 1.0:
            s = torch.exp(u * torch.log1p(torch.tensor(s_max, device=device)))
        else:
            smax = torch.tensor(s_max, device=device)
            b = torch.tensor(self.cfg_beta, device=device)
            log_base = (1.0 - b) * torch.log1p(smax)
            log_inner = torch.log1p(u * torch.expm1(log_base))
            s = torch.exp(log_inner / (1.0 - b))

        return s.float()

    def sample_cfg_interval(self, bz: int, device: torch.device, fm_mask: torch.Tensor):
        t_min = torch.rand(bz, 1, 1, 1, device=device) * 0.5
        t_max = torch.rand(bz, 1, 1, 1, device=device) * 0.5 + 0.5

        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)

        return t_min, t_max

    # ==============================================================
    #  Training Utils & Guidance
    # ==============================================================

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        h = torch.zeros_like(t)
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)
        _, v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)
        return v

    def v_fn(self, x, t, omega, y):
        bz = x.shape[0]
        x2 = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, dtype=torch.long, device=x.device)
        y2 = torch.cat([y, y_null], dim=0)
        t2 = torch.cat([t, t], dim=0)
        w2 = torch.cat([omega, torch.ones_like(omega)], dim=0)

        out = self.v_cond_fn(x2, t2, w2, y2)
        v_c, v_u = out.chunk(2, dim=0)
        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        bz = v_t.shape[0]
        dev = v_t.device

        rand_mask = torch.rand(bz, device=dev) < self.class_dropout_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=dev)[:, None, None, None] < num_drop

        labels = torch.where(
            drop_mask.reshape(bz),
            torch.full_like(labels, self.num_classes),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)
        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        w_interval = torch.where((t >= t_min) & (t <= t_max), w, torch.ones_like(w))
        v_c = self.v_cond_fn(z_t, t, w_interval, y=y)
        v_g = v_t + (1 - 1 / w_interval) * (v_c - v_u)

        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c

    # ==============================================================
    #  Forward Pass and Loss
    # ==============================================================

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        aux_fn=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss.

        Args:
            images: (B, C, H, W) float32
            labels: (B,) long

        Returns:
            loss: scalar
            dict_losses: dict of per-metric scalars
        """
        x = images
        bz = x.shape[0]
        dev = x.device

        t, r, fm_mask = self.sample_tr(bz, dev)

        e = torch.randn_like(x) * self.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / t.clamp(0.05, 1.0)

        t_min, t_max = self.sample_cfg_interval(bz, dev, fm_mask)
        omega = self.sample_cfg_scale(bz, dev, s_max=self.cfg_max)

        v_g, v_c = self.guidance_fn(v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max)

        labels, v_g = self.cond_drop(v_t, v_g, labels)

        # ---- JVP via torch.func ----
        # Build a *pure function* that maps (z_t, t, r) -> (u, v) using
        # the current network parameters so that torch.func.jvp works.
        net_params = dict(self.net.named_parameters())
        net_buffers = dict(self.net.named_buffers())

        def u_fn_pure(z_t_in, t_in, r_in):
            bz_inner = z_t_in.shape[0]
            h_in = t_in - r_in
            u_out, v_out = functional_call(
                self.net,
                {**net_params, **net_buffers},
                (
                    z_t_in,
                    t_in.reshape(bz_inner),
                    h_in.reshape(bz_inner),
                    omega.reshape(bz_inner),
                    t_min.reshape(bz_inner),
                    t_max.reshape(bz_inner),
                    labels,
                ),
            )
            return u_out, v_out

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)

        (u, v), (du_dt, _) = jvp(
            u_fn_pure, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=False
        )

        V = u + (t - r) * du_dt.detach()
        v_g = v_g.detach()

        def adp_wt_fn(loss_per_sample):
            adp_wt = (loss_per_sample + self.norm_eps) ** self.norm_p
            return loss_per_sample / adp_wt.detach()

        loss_u = (V - v_g).pow(2).sum(dim=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        loss_v = (v - v_g).pow(2).sum(dim=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        # Perceptual aux loss
        if self.use_convnext or self.use_lpips:
            assert aux_fn is not None
            pred_x = z_t - t * u
            aux_loss_lpips, aux_loss_convnext = aux_fn(pred_x, x)
            mask = t.flatten() < self.perceptual_max_t
            aux_loss_lpips = torch.where(mask, aux_loss_lpips, torch.zeros_like(aux_loss_lpips))
            aux_loss_convnext = torch.where(mask, aux_loss_convnext, torch.zeros_like(aux_loss_convnext))
            aux_loss = (
                adp_wt_fn(aux_loss_lpips) * self.lpips_lambda
                + adp_wt_fn(aux_loss_convnext) * self.convnext_lambda
            )
        else:
            aux_loss = torch.zeros_like(loss_u)
            aux_loss_lpips = torch.zeros(1, device=dev)
            aux_loss_convnext = torch.zeros(1, device=dev)

        loss = (loss_u + loss_v + aux_loss).mean()

        dict_losses = {
            "loss": loss.detach(),
            "loss_u": (V - v_g).pow(2).mean().detach(),
            "loss_v": (v - v_g).pow(2).mean().detach(),
            "aux_loss_lpips": aux_loss_lpips.mean().detach(),
            "aux_loss_convnext": aux_loss_convnext.mean().detach(),
        }

        return loss, dict_losses
