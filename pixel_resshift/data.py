import os
from copy import deepcopy
from functools import partial
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .compat import ensure_resshift_path
from .config import to_plain_dict


@contextmanager
def resshift_imports(resshift_root):
    # pMF_torch 和 ResShift 都存在顶层 ``utils`` 包。
    # 如果先 import 了 pMF_torch，再去 import ResShift 的 datapipe.datasets，
    # ``from utils import util_sisr`` 会错误地命中 pMF_torch 的 utils。
    # 这里在导入 ResShift 入口前，临时把冲突模块从 sys.modules 里挪开。
    saved_utils = {}
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            saved_utils[name] = sys.modules.pop(name)
    ensure_resshift_path(resshift_root)
    try:
        yield
    finally:
        # 只恢复之前已经存在的 pMF_torch utils，不动 ResShift 已经导入完成的 datapipe/basicsr。
        for name in list(sys.modules):
            if (name == "utils" or name.startswith("utils.")) and name not in saved_utils:
                sys.modules.pop(name, None)
        sys.modules.update(saved_utils)


def _lsdir_realesrgan_config(dataset_config):
    """把 LSDIR 的 HR 图片目录展开成 ResShift/RealESRGAN 在线退化数据集。"""

    config = deepcopy(dataset_config)
    params = dict(config.get("params", {}))
    for stale_key in (
        "source_path",
        "source_txt_path",
        "degrade_kwargs",
        "transform_type",
        "transform_kwargs",
        "recursive",
        "need_path",
    ):
        params.pop(stale_key, None)
    params = {key: value for key, value in params.items() if value is not None}

    root = params.pop("root", "data/LSDIR")
    gt_dir = params.pop("gt_dir", "LSDIR_GT")
    gt_path = params.pop("gt_path", os.path.join(root, gt_dir))

    realesrgan_params = {
        "dir_paths": [gt_path],
        "txt_file_path": [],
        "im_exts": ["png", "PNG"],
        "gt_size": 256,
        "use_hflip": False,
        "use_rot": False,
        "crop_pad_size": 300,
        "rescale_gt": True,
        "io_backend": {"type": "disk"},
        "blur_kernel_size": 21,
        "kernel_list": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
        "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        "blur_sigma": [0.2, 3],
        "betag_range": [0.5, 4],
        "betap_range": [1, 2],
        "sinc_prob": 0.1,
        "blur_kernel_size2": 15,
        "kernel_list2": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
        "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        "blur_sigma2": [0.2, 1.5],
        "betag_range2": [0.5, 4],
        "betap_range2": [1, 2],
        "sinc_prob2": 0.1,
        "final_sinc_prob": 0.8,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "out_size": 256,
    }
    realesrgan_params.update(params)
    return {"type": "realesrgan", "params": realesrgan_params}


def _expand_dataset_config(dataset_config):
    """支持项目内数据集别名，最终仍交给 ResShift 原始 create_dataset。"""

    config = to_plain_dict(dataset_config)
    if config.get("type") in {"lsdir", "lsdir_realesrgan"}:
        return _lsdir_realesrgan_config(config)
    return config


def create_resshift_dataset(dataset_config, resshift_root):
    with resshift_imports(resshift_root):
        from datapipe.datasets import create_dataset

        # 数据集分发完全沿用 ResShift 原仓库。
        return create_dataset(_expand_dataset_config(dataset_config))


def _get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def worker_init_fn(worker_id, rank):
    # 与 pMF_torch 一致：不同 rank 的 DataLoader worker 使用不同随机种子。
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(config, phase="train"):
    dataset_config = config.data.get(phase)
    if dataset_config is None:
        return None

    dataset = create_resshift_dataset(dataset_config, config.resshift_root)
    shuffle = phase == "train"
    batch_size = int(config.training.micro_batch_size)
    sampler = None
    if phase == "train":
        # 与 pMF_torch 一致：每个 DDP rank 只读取自己的数据切片。
        sampler = DistributedSampler(
            dataset,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(config.data.num_workers),
        prefetch_factor=(
            int(getattr(config.data, "prefetch_factor", 2))
            if int(config.data.num_workers) > 0
            else None
        ),
        pin_memory=bool(config.data.pin_memory),
        persistent_workers=(
            bool(getattr(config.data, "persistent_workers", True))
            if int(config.data.num_workers) > 0
            else False
        ),
        drop_last=bool(config.data.drop_last) if phase == "train" else False,
        sampler=sampler,
        worker_init_fn=partial(worker_init_fn, rank=_get_rank()),
    )


def _move_batch(batch, device):
    """Move tensor fields to GPU/CPU while keeping path strings on host."""

    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


class ResShiftBatchPreparer:
    """Convert raw ResShift batches into unified ``{"gt", "lq"}`` tensors.

    对于 bicubic / paired / bsrgan 这类数据集，ResShift 本身已经直接返回
    ``lq`` 和 ``gt``。而对于 ``realesrgan``，数据集只返回 GT 与退化核，
    LR 图像要在训练阶段在线合成；这里复用了 ResShift 原来的退化流程。
    """

    def __init__(self, config, phase="train"):
        self.config = config
        self.phase = phase
        self.jpeger = None
        self.use_sharpener = None
        self._imports_ready = False

    def _lazy_import_resshift(self, device):
        if self._imports_ready:
            return

        ensure_resshift_path(self.config.resshift_root)
        from basicsr.data.degradations import (
            random_add_gaussian_noise_pt,
            random_add_poisson_noise_pt,
        )
        from basicsr.data.transforms import paired_random_crop
        from basicsr.utils import DiffJPEG, USMSharp
        from basicsr.utils.img_process_util import filter2D

        self.random_add_gaussian_noise_pt = random_add_gaussian_noise_pt
        self.random_add_poisson_noise_pt = random_add_poisson_noise_pt
        self.paired_random_crop = paired_random_crop
        self.filter2D = filter2D
        self.DiffJPEG = DiffJPEG
        self.USMSharp = USMSharp
        self._imports_ready = True

        # DiffJPEG / USMSharp 内部带 buffer，所以要和输入图像放到同一 device。
        self.jpeger = self.DiffJPEG(differentiable=False).to(device)
        self.use_sharpener = self.USMSharp().to(device)

    def prepare(self, batch, device):
        batch = _move_batch(batch, device)
        if {"kernel1", "kernel2", "sinc_kernel"}.issubset(batch.keys()):
            return self._prepare_realesrgan_train(batch, device)
        if "gt" not in batch or "lq" not in batch:
            raise KeyError("Expected a ResShift batch containing either lq/gt or gt+kernels.")
        return {"gt": batch["gt"].float(), "lq": batch["lq"].float(), **self._extra_fields(batch)}

    def _extra_fields(self, batch):
        """Pass through optional labels/paths without changing ResShift defaults."""

        extra = {}
        for key in ("y", "label", "gt_path", "lq_path", "path"):
            if key in batch:
                extra[key] = batch[key]
        return extra

    @torch.no_grad()
    def _prepare_realesrgan_train(self, batch, device):
        self._lazy_import_resshift(device)
        deg = self.config.degradation
        im_gt = batch["gt"].float()
        kernel1 = batch["kernel1"].float()
        kernel2 = batch["kernel2"].float()
        sinc_kernel = batch["sinc_kernel"].float()

        ori_h, ori_w = im_gt.size()[2:4]
        sf = deg.sf
        if not isinstance(sf, int):
            if len(sf) != 2:
                raise ValueError("degradation.sf must be an int or a two-value range.")
            sf = random.uniform(*sf)

        if deg.use_sharp:
            im_gt = self.use_sharpener(im_gt)

        # 第一阶段：blur -> resize -> noise -> JPEG
        out = self.filter2D(im_gt, kernel1)
        updown_type = random.choices(["up", "down", "keep"], deg.resize_prob)[0]
        if updown_type == "up":
            scale = random.uniform(1, deg.resize_range[1])
        elif updown_type == "down":
            scale = random.uniform(deg.resize_range[0], 1)
        else:
            scale = 1

        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        gray_noise_prob = deg.gray_noise_prob
        if random.random() < deg.gaussian_noise_prob:
            out = self.random_add_gaussian_noise_pt(
                out,
                sigma_range=deg.noise_range,
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = self.random_add_poisson_noise_pt(
                out,
                scale_range=deg.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        jpeg_p = out.new_zeros(out.size(0)).uniform_(*deg.jpeg_range)
        out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)

        # 第二阶段退化：与 ResShift/RealESRGAN 训练逻辑一致。
        if random.random() < deg.second_order_prob:
            if random.random() < deg.second_blur_prob:
                out = self.filter2D(out, kernel2)

            updown_type = random.choices(["up", "down", "keep"], deg.resize_prob2)[0]
            if updown_type == "up":
                scale = random.uniform(1, deg.resize_range2[1])
            elif updown_type == "down":
                scale = random.uniform(deg.resize_range2[0], 1)
            else:
                scale = 1

            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=mode,
            )

            gray_noise_prob = deg.gray_noise_prob2
            if random.random() < deg.gaussian_noise_prob2:
                out = self.random_add_gaussian_noise_pt(
                    out,
                    sigma_range=deg.noise_range2,
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = self.random_add_poisson_noise_pt(
                    out,
                    scale_range=deg.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

        # 最后阶段：随机交换 JPEG 与 final sinc 的顺序。
        if random.random() < 0.5:
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(ori_h // sf, ori_w // sf), mode=mode)
            out = self.filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*deg.jpeg_range2)
            out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*deg.jpeg_range2)
            out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(ori_h // sf, ori_w // sf), mode=mode)
            out = self.filter2D(out, sinc_kernel)

        if deg.resize_back:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
            crop_sf = 1
        else:
            crop_sf = sf

        # ResShift 最后会模拟 8-bit 量化，再做 paired crop，并归一化到 [-1, 1]。
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        im_gt, im_lq = self.paired_random_crop(im_gt, im_lq, deg.gt_size, crop_sf)

        return {
            "gt": (im_gt - 0.5) / 0.5,
            "lq": (im_lq - 0.5) / 0.5,
            **self._extra_fields(batch),
        }
