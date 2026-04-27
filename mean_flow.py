import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

from .compat import ensure_pmf_torch_path


ensure_pmf_torch_path()

from utils.auxloss_util import paired_random_resized_crop  # noqa: E402

try:
    from lpips import LPIPS
except Exception:
    LPIPS = None


class PixelResShiftMeanFlow(nn.Module):
    """SR-adapted pMF_torch meanflow module.

    这份实现的目标不是做无条件生成，而是做成对超分：

    - ``x0``: 高清 GT 图像，对应 ``t = 0``
    - ``c``: 由 LR 图像上采样得到的条件图
    - ``x1``: 退化端点，定义为 ``c + noise``，对应 ``t = 1``
    - ``z_t``: 线性路径 ``(1 - t) * x0 + t * x1``

    在损失上严格沿用 pMF_torch 的训练路径：
    1. 先用 ``v`` head 在 ``h = 0`` 处预测瞬时速度 ``v_c``；
    2. 再把 ``v_c`` 作为 JVP 的状态方向，求 ``u`` 对时间的导数；
    3. 用复合速度 ``V = u + (t-r) stopgrad(du/dt)`` 对齐教师速度。

    SR 任务里教师速度来自确定的线性路径 ``v_t = (z_t - x0) / t``；
    LR 条件 ``c`` 则对应 pMF 里的条件信息，uncond 分支使用零条件图。
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        mf_cfg = config.meanflow
        self.P_mean = float(mf_cfg.P_mean)
        self.P_std = float(mf_cfg.P_std)
        self.cfg_max = float(mf_cfg.cfg_max)
        self.noise_scale = float(mf_cfg.noise_scale)
        self.data_proportion = float(mf_cfg.data_proportion)
        self.cfg_beta = float(mf_cfg.cfg_beta)
        self.class_dropout_prob = float(mf_cfg.class_dropout_prob)
        self.norm_p = float(mf_cfg.norm_p)
        self.norm_eps = float(mf_cfg.norm_eps)
        self.tr_uniform = bool(mf_cfg.tr_uniform)
        self.use_lpips = bool(mf_cfg.lpips)
        self.lpips_lambda = float(mf_cfg.lpips_lambda)
        self.use_convnext = bool(getattr(mf_cfg, "convnext", False))
        self.convnext_lambda = float(getattr(mf_cfg, "convnext_lambda", 0.0))
        self.perceptual_max_t = float(mf_cfg.perceptual_max_t)
        self.num_classes = int(config.model.num_classes)

        if self.use_lpips:
            if LPIPS is None:
                raise RuntimeError("LPIPS is enabled in config but the lpips package is not installed.")
            self.lpips = LPIPS(net="vgg").eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            self.lpips = None

        if self.use_convnext:
            from models.convnext import load_convnext_model  # noqa: E402

            self.convnext = load_convnext_model(device="cpu").eval()
            for param in self.convnext.parameters():
                param.requires_grad = False
        else:
            self.convnext = None

    def _maybe_move_lpips(self, device):
        if self.lpips is None:
            return
        try:
            param = next(self.lpips.parameters())
        except StopIteration:
            return
        if param.device != device:
            self.lpips.to(device)

    def _maybe_move_convnext(self, device):
        if self.convnext is None:
            return
        try:
            param = next(self.convnext.parameters())
        except StopIteration:
            return
        if param.device != device:
            self.convnext.to(device)

    # ------------------------------------------------------------------
    # pMF_torch schedule helpers
    # ------------------------------------------------------------------

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

    def sample_cfg_scale(self, bz: int, device: torch.device, s_max: float):
        # 与 pMF_torch 完全一致：omega 从 [1, 1 + s_max] 的对数空间采样。
        u = torch.rand(bz, 1, 1, 1, device=device)

        if self.cfg_beta == 1.0:
            s = torch.exp(u * torch.log1p(torch.tensor(s_max, device=device)))
        else:
            smax = torch.tensor(s_max, device=device)
            beta = torch.tensor(self.cfg_beta, device=device)
            log_base = (1.0 - beta) * torch.log1p(smax)
            log_inner = torch.log1p(u * torch.expm1(log_base))
            s = torch.exp(log_inner / (1.0 - beta))

        return s.float()

    def sample_cfg_interval(self, bz: int, device: torch.device, fm_mask: torch.Tensor):
        # pMF_torch 的区间 CFG：非 FM 样本随机限制 guidance 生效区间；
        # r=t 的 FM 样本则退回完整 [0, 1] 区间。
        t_min = torch.rand(bz, 1, 1, 1, device=device) * 0.5
        t_max = torch.rand(bz, 1, 1, 1, device=device) * 0.5 + 0.5

        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)

        return t_min, t_max

    def adp_wt_fn(self, loss_per_sample):
        adp_wt = (loss_per_sample + self.norm_eps) ** self.norm_p
        return loss_per_sample / adp_wt.detach()

    # ------------------------------------------------------------------
    # SR path construction
    # ------------------------------------------------------------------

    def _resize(self, x, size, mode):
        if x.shape[-2:] == size:
            return x
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        return F.interpolate(x, size=size, mode=mode)

    def prepare_condition(self, lq, target_size):
        # DiT 运行在 HR 像素空间上，所以先把 LR 图像上采样为条件图 c。
        mode = self.config.meanflow.condition_upsample_mode
        return self._resize(lq, target_size, mode=mode)

    def build_endpoint(self, lq, target_size, add_noise=True):
        c = self.prepare_condition(lq, target_size)
        x1 = c
        noise_std = float(self.config.meanflow.endpoint_noise_std)
        if add_noise and noise_std > 0:
            # 这里对齐 ResShift 送入网络后的端点噪声强度：
            # raw prior std = kappa * sqrt_eta_T = 2.0 * 0.99 = 1.98；
            # ResShift 还会做 _scale_input，等效 std = 1.98 / sqrt(1 + 1.98^2) ~= 0.8927。
            # 条件 c 仍然是干净 LR 上采样图，只有状态端点 x1 加噪。
            # endpoint 本身仍然是 “LR 条件 + 噪声”，后续路径保持从 HR 到 endpoint 的线性残差偏移。
            x1 = x1 + torch.randn_like(x1) * noise_std
        if self.config.meanflow.clip_endpoint:
            x1 = torch.clamp(x1, -1.0, 1.0)
        return c, x1

    def _labels_from_batch(self, batch, batch_size, device):
        # 超分默认没有类别标签；保留接口只是为了兼容 pMF_torch 的 class tokens。
        y = batch.get("y", batch.get("label", None))
        if y is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        return y.to(device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # pMF_torch guidance helpers adapted to LR image conditioning
    # ------------------------------------------------------------------

    def u_fn(self, x, t, h, omega, t_min, t_max, y, c):
        bz = x.shape[0]
        return self.model(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
            c,
        )

    def v_cond_fn(self, x, t, omega, y, c):
        # h=0 时，v head 表示 t 时刻的瞬时速度；这是 JVP 的状态方向。
        h = torch.zeros_like(t)
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)
        _, v = self.u_fn(x, t, h, omega, t_min, t_max, y, c)
        return v

    def v_fn(self, x, t, omega, y, c):
        bz = x.shape[0]
        x2 = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, dtype=torch.long, device=x.device)
        y2 = torch.cat([y, y_null], dim=0)
        t2 = torch.cat([t, t], dim=0)
        omega2 = torch.cat([omega, torch.ones_like(omega)], dim=0)
        # SR 的主要条件是 LR 图像：cond 分支看真实 c，uncond 分支看零条件图。
        c2 = torch.cat([c, torch.zeros_like(c)], dim=0)

        out = self.v_cond_fn(x2, t2, omega2, y2, c2)
        v_c, v_u = out.chunk(2, dim=0)
        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels, c):
        bz = v_t.shape[0]
        dev = v_t.device

        # 这里照搬 pMF_torch 的“按数量丢弃前若干样本”的实现。
        # 由于 SR 的条件主要来自 LR 图像，drop 时同时把 c 置零。
        rand_mask = torch.rand(bz, device=dev) < self.class_dropout_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=dev)[:, None, None, None] < num_drop

        labels = torch.where(
            drop_mask.reshape(bz),
            torch.full_like(labels, self.num_classes),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)
        c = torch.where(drop_mask, torch.zeros_like(c), c)
        return labels, c, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, c, fm_mask, omega, t_min, t_max):
        v_c, v_u = self.v_fn(z_t, t, omega, y, c)
        v_g_fm = v_t + (1 - 1 / omega) * (v_c - v_u)

        w_interval = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        v_c = self.v_cond_fn(z_t, t, w_interval, y, c)
        v_g = v_t + (1 - 1 / w_interval) * (v_c - v_u)

        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c

    def aux_losses(self, pred_x, x0, t):
        bz = x0.shape[0]
        dev = x0.device

        if not (self.use_lpips or self.use_convnext):
            zeros = torch.zeros(bz, device=dev)
            return zeros, zeros

        pred_crop, gt_crop = paired_random_resized_crop(
            pred_x.clamp(-1, 1),
            x0.clamp(-1, 1),
            out_size=224,
        )

        if self.use_lpips:
            self._maybe_move_lpips(dev)
            loss_lpips = self.lpips(pred_crop, gt_crop).reshape(-1)
        else:
            loss_lpips = torch.zeros(bz, device=dev)

        if self.use_convnext:
            self._maybe_move_convnext(dev)
            # pMF_torch 的 ConvNeXtV2 forward 带 no_grad；这里保持同样的实现语义。
            feat_pred = self.convnext(pred_crop)
            feat_gt = self.convnext(gt_crop)
            loss_convnext = (feat_pred - feat_gt).pow(2).sum(dim=-1)
        else:
            loss_convnext = torch.zeros(bz, device=dev)

        active_mask = t.reshape(-1) < self.perceptual_max_t
        loss_lpips = torch.where(active_mask, loss_lpips, torch.zeros_like(loss_lpips))
        loss_convnext = torch.where(active_mask, loss_convnext, torch.zeros_like(loss_convnext))
        return loss_lpips, loss_convnext

    # ------------------------------------------------------------------
    # Forward / Loss
    # ------------------------------------------------------------------

    def forward(self, batch):
        # DDP 需要通过 module.forward() 进入计算图，才能为参数梯度注册同步 hook。
        return self.forward_loss(batch)

    def forward_loss(self, batch):
        x0 = batch["gt"].float()
        lq = batch["lq"].float()
        bz = x0.shape[0]
        dev = x0.device
        labels = self._labels_from_batch(batch, bz, dev)

        t, r, fm_mask = self.sample_tr(bz, dev)
        c, x1 = self.build_endpoint(lq, target_size=x0.shape[-2:], add_noise=True)
        z_t = (1 - t) * x0 + t * x1
        # 与 pMF_torch 对齐：教师瞬时速度写成 (z_t - x0) / t。
        # 在线性路径下它等价于 x1 - x0，但保留 clamp 可以避免极小 t 带来的数值尖峰。
        v_t = (z_t - x0) / t.clamp(0.05, 1.0)

        t_min, t_max = self.sample_cfg_interval(bz, dev, fm_mask)
        omega = self.sample_cfg_scale(bz, dev, s_max=self.cfg_max)

        v_g, v_c = self.guidance_fn(v_t, z_t, t, r, labels, c, fm_mask, omega, t_min, t_max)
        labels, c_for_model, v_g = self.cond_drop(v_t, v_g, labels, c)

        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        params_and_buffers = {**params, **buffers}

        def net_pure(z_in, t_in, r_in):
            bz_inner = z_in.shape[0]
            h_in = t_in - r_in
            return functional_call(
                self.model,
                params_and_buffers,
                (
                    z_in,
                    t_in.reshape(bz_inner),
                    h_in.reshape(bz_inner),
                    omega.reshape(bz_inner),
                    t_min.reshape(bz_inner),
                    t_max.reshape(bz_inner),
                    labels,
                    c_for_model,
                ),
            )

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)
        # pMF_torch 的关键点：JVP 的状态方向不是手写真实速度，
        # 而是模型 v head 在 h=0 预测出来的条件瞬时速度 v_c。
        (u, v), (du_dt, _) = jvp(
            net_pure,
            (z_t, t, r),
            (v_c, dtdt, dtdr),
            has_aux=False,
        )

        V = u + (t - r) * du_dt.detach()
        v_g = v_g.detach()

        loss_u_raw = (V - v_g).pow(2).sum(dim=(1, 2, 3))
        loss_u = self.adp_wt_fn(loss_u_raw)

        loss_v_raw = (v - v_g).pow(2).sum(dim=(1, 2, 3))
        loss_v = self.adp_wt_fn(loss_v_raw)

        pred_x = z_t - t * u

        loss_lpips_raw, loss_convnext_raw = self.aux_losses(pred_x, x0, t)
        loss_aux = (
            self.adp_wt_fn(loss_lpips_raw) * self.lpips_lambda
            + self.adp_wt_fn(loss_convnext_raw) * self.convnext_lambda
        )

        loss = (loss_u + loss_v + loss_aux).mean()

        metrics = {
            "loss": float(loss.detach()),
            "loss_u": float((V - v_g).pow(2).mean().detach()),
            "loss_v": float((v - v_g).pow(2).mean().detach()),
            "loss_lpips": float(loss_lpips_raw.mean().detach()),
            "loss_convnext": float(loss_convnext_raw.mean().detach()),
        }
        return loss, metrics

    @torch.no_grad()
    def restore(self, lq, y=None, add_noise=True):
        """One-step restoration from the ResShift-style noisy LR prior back to x0.

        ResShift 推理不是直接从干净 LR 开始，而是从 ``LR_up + noise`` 的 prior 开始；
        干净 LR_up 仍作为条件 c 输入给网络。若想做确定性消融，可以显式传
        ``add_noise=False``。
        """

        dev = lq.device
        target_size = (int(self.config.model.image_size), int(self.config.model.image_size))
        c, x1 = self.build_endpoint(lq, target_size=target_size, add_noise=add_noise)
        bz = x1.shape[0]

        if y is None:
            y = torch.zeros(bz, dtype=torch.long, device=dev)

        t = torch.ones(bz, device=dev)
        r = torch.zeros(bz, device=dev)
        h = t - r
        omega = torch.ones(bz, device=dev)
        t_min = torch.zeros(bz, device=dev)
        t_max = torch.ones(bz, device=dev)

        u, _ = self.model(x1, t, h, omega, t_min, t_max, y, c)
        x0_pred = x1 - t.reshape(-1, 1, 1, 1) * u
        return x0_pred.clamp(-1, 1)
