import os
from copy import deepcopy

from .compat import default_project_root


class AttrDict(dict):
    """A small dict wrapper that supports ``config.foo.bar`` access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = _wrap(value)

    def copy(self):
        return AttrDict({key: deepcopy(value) for key, value in self.items()})


def _wrap(value):
    if isinstance(value, AttrDict):
        return value
    if isinstance(value, dict):
        return AttrDict({key: _wrap(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def to_plain_dict(value):
    """Convert nested AttrDict/list structures back to plain Python containers."""

    if isinstance(value, AttrDict):
        return {key: to_plain_dict(val) for key, val in value.items()}
    if isinstance(value, dict):
        return {key: to_plain_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


def deep_update(base, updates):
    """Recursively merge ``updates`` into ``base``."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = _wrap(value)
    return base


def _default_config():
    project_root = default_project_root()
    return _wrap(
        {
            "project_root": project_root,
            "pmf_torch_root": os.path.join(project_root, "pMF_torch"),
            "resshift_root": os.path.join(project_root, "ResShift"),
            "output_dir": os.path.join(project_root, "outputs", "pixel_resshift"),
            "checkpoint_dir": os.path.join(project_root, "outputs", "pixel_resshift", "checkpoints"),
            "log_dir": os.path.join(project_root, "outputs", "pixel_resshift", "logs"),
            "image_dir": os.path.join(project_root, "outputs", "pixel_resshift", "images"),
            "seed": 42,
            "model": {
                # 这里直接沿用 pMF_torch 的模型命名。
                "model_str": "pmfDiT_B_16",
                "image_size": 256,
                "in_channels": 3,
                "cond_channels": 3,
                # LR cross-attention 的初始门控；小正值能让条件分支从第一步就有梯度。
                "cross_scale_init": 0.01,
                # 使用 pMF 预训练权重时保持 1000 类 embedding 形状，但默认喂 null class。
                "num_classes": 1000,
                "eval_mode": False,
            },
            "meanflow": {
                # 这些字段来自 pMF_torch/pMF 的 meanflow 训练超参。
                "P_mean": 0.8,
                "P_std": 0.8,
                "cfg_max": 7.0,
                "noise_scale": 1.0,
                "data_proportion": 0.5,
                "cfg_beta": 1.0,
                "class_dropout_prob": 0.0,
                "norm_p": 1.0,
                "norm_eps": 0.01,
                "tr_uniform": False,
                # SR 特有：把 LR 上采样为条件图 c，再在退化端点额外加噪。
                "condition_upsample_mode": "bicubic",
                # 对齐 ResShift bicubic x4 的“模型实际看到的”端点噪声强度：
                # ResShift prior 是 y + kappa * sqrt_eta_T * noise，raw std = 2.0 * 0.99 = 1.98；
                # 但 ResShift 送入网络前会除以 sqrt(1 + raw_std^2)，等效 std 约为 0.8927。
                # 我们的 pixel DiT 没有这一步 _scale_input，因此直接使用等效 std。
                "endpoint_noise_std": 0.8927,
                "clip_endpoint": False,
                # 这里保留 pMF_torch 的 LPIPS 使用方式：只在较小 t 区间启用感知损失。
                "lpips": True,
                "lpips_lambda": 0.4,
                "convnext": True,
                "convnext_lambda": 0.1,
                "perceptual_max_t": 0.8,
                # SR 锚定项：直接约束模型的一步 x_pred 不要偏离 GT 内容。
                "pixel_l1_lambda": 1.0,
                "lr_consistency_lambda": 1.0,
                "lr_consistency_mode": "bicubic",
                # pMF 权重仍保留 1000 类 embedding，但 SR 训练默认不再使用真实类别。
                # False 时统一喂 null class token，避免类别语义把超分内容带偏。
                "use_label_condition": False,
            },
            "degradation": {
                # 这些参数与 ResShift trainer.prepare_data / RealESRGAN 二阶退化一致。
                "sf": 4,
                "resize_prob": [0.2, 0.7, 0.1],
                "resize_range": [0.15, 1.5],
                "gaussian_noise_prob": 0.5,
                "noise_range": [1, 30],
                "poisson_scale_range": [0.05, 3.0],
                "gray_noise_prob": 0.4,
                "jpeg_range": [30, 95],
                "second_order_prob": 0.5,
                "second_blur_prob": 0.8,
                "resize_prob2": [0.3, 0.4, 0.3],
                "resize_range2": [0.3, 1.2],
                "gaussian_noise_prob2": 0.5,
                "noise_range2": [1, 25],
                "poisson_scale_range2": [0.05, 2.5],
                "gray_noise_prob2": 0.4,
                "jpeg_range2": [30, 95],
                "gt_size": 256,
                "resize_back": False,
                "use_sharp": False,
            },
            "data": {
                "num_workers": 4,
                # 与 pMF_torch 的 input_pipeline 保持一致：worker 预取多个 batch，
                # 减少 GPU 等待下一批数据和在线退化结果的时间。
                "prefetch_factor": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "drop_last": True,
                "train": {
                    # 默认先给一个最常用的 bicubic 配置；正式 real-world SR 可以换成 realesrgan。
                    "type": "bicubic",
                    "params": {
                        "source_path": None,
                        "source_txt_path": None,
                        "degrade_kwargs": {
                            "scale": 0.25,
                            "activate_matlab": True,
                            "resize_back": False,
                            "pch_size": 256,
                            "pass_smallmaxresize": False,
                            "pass_aug": False,
                            "pass_crop": False,
                        },
                        "transform_type": "default",
                        "transform_kwargs": {"mean": 0.5, "std": 0.5},
                        "length": None,
                        "need_path": False,
                        "im_exts": ["png", "jpg", "jpeg", "JPEG"],
                        "recursive": False,
                    },
                },
                "val": None,
            },
            "training": {
                # batch_size 保留 pMF_torch 的“全局 batch”语义；
                # micro_batch_size 是单次 DataLoader/显存批大小。
                "batch_size": 8,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": None,
                "num_epochs": 1,
                "max_steps": None,
                # 这里先保留字段，但训练默认按 pMF_torch 的稳定性考虑走 fp32。
                # 如果后面要尝试 bf16，需要额外验证 torch.func.jvp 在当前环境下是否稳定。
                "precision": "fp32",
                # TF32 不改变张量 dtype，但会让 Ampere/Hopper 上的 fp32 matmul/conv
                # 使用 Tensor Core；这是比 bf16 更稳的第一档提速开关。
                "allow_tf32": True,
                "optimizer": "muon",
                "learning_rate": 1e-3,
                "adam_b2": 0.95,
                "weight_decay": 0.0,
                "lr_schedule": "warmup_const",
                "warmup_epochs": 0,
                "lr_min_factor": 0.0,
                "grad_clip_norm": 1.0,
                "log_per_step": 20,
                # 训练日志里额外输出最近 N 个 optimizer step 的平均 loss，
                # 用来观察 raw loss 的整体趋势，避免被单个 batch 的抖动误导。
                "log_avg_window": 100,
                # 每隔多少个 optimizer step 保存一张可视化图；0 表示不保存。
                # 图片包含 GT、LR 上采样条件图、当前模型恢复结果。
                "save_image_per_step": 0,
                # 固定第一批样本做可视化，避免每次随机 batch 不同导致趋势不可比。
                "fixed_visual_batch": True,
                "visual_num_samples": 4,
                "save_per_step": 1000,
                # 默认保留旧行为：每个 epoch 末尾也保存一次。
                # 长训练时可以在 yaml 里关掉，只按 save_per_step 保存 checkpoint。
                "save_epoch_end": True,
                "resume_from": None,
                # 可选：从原始 pMF_torch checkpoint 初始化 DiT 主干。
                # 新增的 LR 条件/cross-attention 模块没有对应权重，会自动跳过。
                "pretrained_pmf_path": None,
                "pretrained_pmf_ema": 2000,
            },
        }
    )


class Config(AttrDict):
    def __init__(self, config_dict=None):
        super().__init__()
        self.update(_default_config())
        if config_dict:
            deep_update(self, _wrap(config_dict))


def load_config(path=None):
    if path is None:
        path = os.environ.get("PIXEL_RESSHIFT_CONFIG")
    if path is None:
        return Config()
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return Config(loaded)


def get_config(path=None):
    return load_config(path)
