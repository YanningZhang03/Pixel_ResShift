import argparse
import glob
import logging
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .config import get_config, to_plain_dict
from .data import ResShiftBatchPreparer, build_dataloader
from .dit import build_model
from .mean_flow import PixelResShiftMeanFlow
from .optimizer import count_trainable_parameters, create_optimizer_and_scheduler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None


logger = logging.getLogger(__name__)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize DDP from torchrun env vars, following pMF_torch/main.py."""

    if "RANK" in os.environ and not is_distributed():
        dist.init_process_group(backend="nccl")


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process() else logging.WARNING,
    )


def configure_torch_backend(config):
    """打开不改变训练语义的 CUDA 后端优化。"""

    allow_tf32 = bool(getattr(config.training, "allow_tf32", True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            # high 会允许 fp32 matmul 使用 TF32；比 medium 更保守。
            torch.set_float32_matmul_precision("high")
    logger.info("CUDA TF32 enabled: %s", allow_tf32)


def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda", get_local_rank())
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def compute_accumulation_steps(config):
    """Infer gradient accumulation steps from global batch and micro batch.

    pMF_torch 的配置习惯里 ``batch_size`` 更接近“目标总 batch”。
    DDP 下每张卡各跑一个 micro batch，所以自动推断时会除以 world size。
    """

    explicit = config.training.gradient_accumulation_steps
    if explicit not in (None, 0):
        return int(explicit)

    global_batch = int(config.training.batch_size)
    micro_batch = int(config.training.micro_batch_size)
    return max(1, math.ceil(global_batch / (micro_batch * get_world_size())))


def get_log_avg_window(config):
    """Return the rolling metric window used only for logging.

    这个窗口不参与反向传播，只是把最近一段训练的 raw loss 平均起来，
    方便判断趋势；没有显式配置时，默认等于 log_per_step。
    """

    return max(
        1,
        int(getattr(config.training, "log_avg_window", config.training.log_per_step)),
    )


def average_metric_history(metric_history):
    """Average scalar metrics stored in the rolling history."""

    if not metric_history:
        return {}

    keys = sorted(metric_history[0].keys())
    averaged = {}
    for key in keys:
        values = [metrics[key] for metrics in metric_history if key in metrics]
        if values:
            averaged[key] = float(sum(values) / len(values))
    return averaged


def median_metric_history(metric_history):
    """Median scalar metrics stored in the rolling history."""

    if not metric_history:
        return {}

    keys = sorted(metric_history[0].keys())
    medians = {}
    for key in keys:
        values = [metrics[key] for metrics in metric_history if key in metrics]
        if values:
            medians[key] = float(np.median(values))
    return medians


def format_metrics(metrics):
    """Format metrics into a compact one-line string for text logs."""

    return " ".join(f"{key}={value:.4g}" for key, value in sorted(metrics.items()))


def ensure_dirs(config):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    if int(getattr(config.training, "save_image_per_step", 0) or 0) > 0:
        os.makedirs(get_image_dir(config), exist_ok=True)


def get_image_dir(config):
    """Return visualization directory, defaulting to output_dir/images."""

    return getattr(config, "image_dir", None) or os.path.join(config.output_dir, "images")


def _tensor_to_pil(image_tensor):
    """Convert one CHW tensor in [-1, 1] to a PIL RGB image."""

    array = image_tensor.detach().float().cpu().clamp(-1, 1)
    array = ((array + 1.0) * 127.5).round().to(torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_triplet_image(path, gt, lq_up, pred):
    """Save a GT / LR / prediction triplet into one readable PNG."""

    if Image is None or ImageDraw is None:
        logger.warning("Pillow is not installed; skip visualization image %s", path)
        return

    images = [_tensor_to_pil(gt), _tensor_to_pil(lq_up), _tensor_to_pil(pred)]
    labels = ("GT", "LR", "Pred")
    width, height = images[0].size
    label_height = 24
    gap = 8
    canvas = Image.new("RGB", (width * 3 + gap * 2, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (label, image) in enumerate(zip(labels, images)):
        x_offset = index * (width + gap)
        draw.text((x_offset + 4, 4), label, fill=(0, 0, 0))
        canvas.paste(image, (x_offset, label_height))

    canvas.save(path)


def save_triplet_grid(path, gt, lq_up, pred):
    """Save several fixed GT / LR / Pred rows for stable visual comparison."""

    if Image is None or ImageDraw is None:
        logger.warning("Pillow is not installed; skip visualization image %s", path)
        return

    num_rows = int(gt.shape[0])
    if num_rows <= 1:
        save_triplet_image(path, gt[0], lq_up[0], pred[0])
        return

    width, height = _tensor_to_pil(gt[0]).size
    label_height = 24
    gap = 8
    row_gap = 8
    canvas = Image.new(
        "RGB",
        (width * 3 + gap * 2, label_height + num_rows * height + (num_rows - 1) * row_gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(("GT", "LR", "Pred")):
        draw.text((index * (width + gap) + 4, 4), label, fill=(0, 0, 0))

    for row in range(num_rows):
        y_offset = label_height + row * (height + row_gap)
        for col, tensor in enumerate((gt[row], lq_up[row], pred[row])):
            canvas.paste(_tensor_to_pil(tensor), (col * (width + gap), y_offset))

    canvas.save(path)


def clone_visual_batch(batch, max_items):
    """Clone a small fixed batch for repeated visualizations."""

    fixed = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            fixed[key] = value[:max_items].detach().clone()
        elif isinstance(value, (list, tuple)):
            fixed[key] = value[:max_items]
        else:
            fixed[key] = value
    return fixed


@torch.no_grad()
def save_training_visualization(config, flow, batch, global_step, writer=None):
    """Save one visual sample from the current training batch.

    这里不参与训练，只用当前 batch 的第一张图做快速体检：
    GT 是目标高清图，LR 会先按训练条件同样上采样。
    Pred 按 ResShift 推理方式从加噪 LR prior 恢复，条件 c 仍是干净 LR 上采样图。
    """

    image_dir = get_image_dir(config)
    os.makedirs(image_dir, exist_ok=True)

    num_samples = max(1, int(getattr(config.training, "visual_num_samples", 1)))
    gt = batch["gt"][:num_samples]
    lq = batch["lq"][:num_samples]
    label = None
    if not bool(getattr(config.meanflow, "use_label_condition", False)):
        label = None
    elif "y" in batch:
        label = batch["y"][:num_samples]
    elif "label" in batch:
        label = batch["label"][:num_samples]

    was_training = flow.model.training
    flow.model.eval()
    pred = flow.restore(lq, y=label, add_noise=True)
    if was_training:
        flow.model.train()

    lq_up = flow.prepare_condition(lq, target_size=gt.shape[-2:])
    image_path = os.path.join(image_dir, f"step-{global_step:07d}.png")
    save_triplet_grid(image_path, gt, lq_up, pred)

    if writer is not None:
        writer.add_images("visual/gt", ((gt.clamp(-1, 1) + 1) * 0.5), global_step)
        writer.add_images("visual/lr_up", ((lq_up.clamp(-1, 1) + 1) * 0.5), global_step)
        writer.add_images("visual/pred", ((pred.clamp(-1, 1) + 1) * 0.5), global_step)

    logger.info("Saved visualization image to %s", image_path)


def save_checkpoint(config, model, optimizer, scheduler, step):
    path = os.path.join(config.checkpoint_dir, f"pixel-resshift-step-{step}.pt")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": to_plain_dict(config),
    }
    torch.save(state, path)
    logger.info("Saved checkpoint to %s", path)


def _select_checkpoint_file(path):
    """从文件或目录里选出一个 pMF checkpoint 文件。"""

    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "checkpoint_*.pt")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.pt")))
        if not files:
            raise FileNotFoundError(f"No checkpoint file found under {path}")
        return files[-1]
    return path


def _select_ema_state(ckpt, ema_key):
    """优先从 pMF_torch checkpoint 里取指定 EMA 权重。"""

    ema_params = ckpt.get("ema_params", {})
    for key in (ema_key, float(ema_key), str(ema_key)):
        if key in ema_params:
            return ema_params[key], f"ema_{key}"
    return ckpt.get("model_state_dict", ckpt.get("model", {})), "online"


def load_pmf_pretrained(config, model, device):
    """用原始 pMF_torch 权重 warm-start v4 的 DiT 主干。

    这里只加载“参数名存在且形状完全一致”的张量。v4 新增的 LR cross-attention
    和条件分支没有 pMF 对应权重，会保持当前初始化；这样既能继承 pMF 的自然图像先验，
    又不会错误地把无条件生成权重硬塞进 SR 条件模块。
    """

    ckpt_file = _select_checkpoint_file(getattr(config.training, "pretrained_pmf_path", None))
    if ckpt_file is None:
        logger.info("No pretrained pMF checkpoint configured.")
        return
    if is_distributed() and get_rank() != 0:
        # DDP 构造时会从 rank0 广播参数；非主进程不重复读取 3GB+ checkpoint，
        # 避免把共享磁盘和系统内存同时打满。
        return

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    ema_key = getattr(config.training, "pretrained_pmf_ema", None)
    src_state, src_name = _select_ema_state(ckpt, ema_key)

    target_state = model.state_dict()
    loadable = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in src_state.items():
        # pMF_torch 的完整 checkpoint 保存的是 pixelMeanFlow，内部 DiT 参数带 net. 前缀。
        model_key = key[4:] if key.startswith("net.") else key
        if model_key not in target_state:
            skipped_missing.append(model_key)
            continue
        if tuple(target_state[model_key].shape) != tuple(value.shape):
            skipped_shape.append((model_key, tuple(value.shape), tuple(target_state[model_key].shape)))
            continue
        loadable[model_key] = value.to(device=device, dtype=target_state[model_key].dtype)

    merged = dict(target_state)
    merged.update(loadable)
    model.load_state_dict(merged, strict=True)

    logger.info(
        "Loaded %d tensors from pMF checkpoint %s (%s). Skipped missing=%d shape=%d.",
        len(loadable),
        ckpt_file,
        src_name,
        len(skipped_missing),
        len(skipped_shape),
    )
    if skipped_shape:
        logger.info("Shape-skipped examples: %s", skipped_shape[:8])


def maybe_resume(config, model, optimizer, scheduler, device):
    resume_path = config.training.resume_from
    if not resume_path:
        return 0
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    step = int(ckpt.get("step", 0))
    logger.info("Resumed from %s at step %d", resume_path, step)
    return step


def train(config_path=None):
    setup_distributed()
    setup_logging()
    config = get_config(config_path)
    ensure_dirs(config)
    seed_everything(config.seed)
    configure_torch_backend(config)

    device = get_device()
    logger.info(
        "DDP process: rank %d / %d, local_rank %d",
        get_rank(),
        get_world_size(),
        get_local_rank(),
    )
    logger.info("Using device: %s", device)
    logger.info(
        "Training precision is kept in fp32 by default because torch.func.jvp "
        "is the critical path and is usually the first thing to get unstable under AMP."
    )

    train_loader = build_dataloader(config, "train")
    if train_loader is None:
        raise RuntimeError("config.data.train is required.")

    accumulation_steps = compute_accumulation_steps(config)
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))

    # 模型主干来自 pMF_torch，SR 扩展只在条件图分支和 loss/path 上。
    model = build_model(config).to(device)
    load_pmf_pretrained(config, model, device)
    flow = PixelResShiftMeanFlow(model, config).to(device)

    optimizer, scheduler = create_optimizer_and_scheduler(
        config,
        model,
        optimizer_steps_per_epoch,
    )

    logger.info("Trainable parameters: %s", f"{count_trainable_parameters(model):,}")
    logger.info(
        "Global batch: %d, local micro batch: %d, world: %d",
        int(config.training.batch_size),
        int(config.training.micro_batch_size),
        get_world_size(),
    )
    logger.info("Gradient accumulation steps: %d", accumulation_steps)

    start_step = maybe_resume(config, model, optimizer, scheduler, device)
    ddp_flow = (
        DDP(flow, device_ids=[get_local_rank()])
        if is_distributed() and get_world_size() > 1
        else flow
    )
    raw_flow = ddp_flow.module if isinstance(ddp_flow, DDP) else ddp_flow
    raw_model = raw_flow.model
    batch_preparer = ResShiftBatchPreparer(config, phase="train")
    writer = (
        SummaryWriter(log_dir=config.log_dir)
        if SummaryWriter is not None and is_main_process()
        else None
    )

    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    log_avg_window = get_log_avg_window(config)
    metric_history = deque(maxlen=log_avg_window)
    image_interval = int(getattr(config.training, "save_image_per_step", 0) or 0)
    last_image_step = None
    fixed_visual_batch = None
    use_fixed_visual_batch = bool(getattr(config.training, "fixed_visual_batch", False))
    logger.info("Averaged training metrics use a rolling window of %d optimizer steps.", log_avg_window)
    if image_interval > 0:
        logger.info(
            "Training visualizations will be saved every %d optimizer steps to %s.",
            image_interval,
            get_image_dir(config),
        )
        if use_fixed_visual_batch:
            logger.info(
                "Visualizations will reuse a fixed batch of %d samples.",
                int(getattr(config.training, "visual_num_samples", 1)),
            )

    for epoch in range(int(config.training.num_epochs)):
        if hasattr(train_loader.sampler, "set_epoch"):
            # 与 pMF_torch 一样，每个 epoch 重新洗牌，同时保证各 rank 不重复取样。
            train_loader.sampler.set_epoch(epoch)

        ddp_flow.train()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=True,
            disable=not is_main_process(),
        )
        metric_buffer = {}
        micro_in_accum = 0

        for raw_batch in progress:
            # ResShift 的不同 dataset 输出格式不完全一样，这里统一成 gt/lq。
            batch = batch_preparer.prepare(raw_batch, device)
            if image_interval > 0 and use_fixed_visual_batch and fixed_visual_batch is None and is_main_process():
                fixed_visual_batch = clone_visual_batch(
                    batch,
                    max(1, int(getattr(config.training, "visual_num_samples", 1))),
                )
            # 从 checkpoint 续跑时，如果当前 step 正好是可视化间隔点，先保存一次恢复点效果。
            if (
                image_interval > 0
                and global_step > 0
                and global_step % image_interval == 0
                and last_image_step != global_step
            ):
                if is_main_process():
                    visual_batch = fixed_visual_batch if fixed_visual_batch is not None else batch
                    save_training_visualization(config, raw_flow, visual_batch, global_step, writer)
                last_image_step = global_step

            # 这里保持 pMF_torch 的朴素 DDP 方式：每个 micro batch 都走 DDP forward，
            # 由 PyTorch 在 backward 时自动做梯度同步。
            loss, metrics = ddp_flow(batch)
            (loss / accumulation_steps).backward()
            micro_in_accum += 1

            for key, value in metrics.items():
                metric_buffer[key] = metric_buffer.get(key, 0.0) + float(value)

            should_step = micro_in_accum == accumulation_steps
            if should_step:
                if config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(),
                        float(config.training.grad_clip_norm),
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                averaged = {
                    key: value / micro_in_accum
                    for key, value in metric_buffer.items()
                }
                averaged["lr"] = scheduler.get_last_lr()[0]
                # latest 是当前 step；window_averaged 是最近 N 步平均，更适合看 loss 走势。
                metric_history.append(averaged)
                window_averaged = average_metric_history(metric_history)
                window_median = median_metric_history(metric_history)
                postfix = dict(averaged)
                for key in (
                    "loss",
                    "loss_u",
                    "loss_v",
                    "loss_lpips",
                    "loss_convnext",
                    "loss_l1",
                    "loss_lr",
                ):
                    if key in window_averaged:
                        postfix[f"avg_{key}"] = window_averaged[key]
                    if key in window_median:
                        postfix[f"med_{key}"] = window_median[key]
                progress.set_postfix(**postfix)

                if writer is not None and global_step % int(config.training.log_per_step) == 0:
                    for key, value in averaged.items():
                        writer.add_scalar(f"train/{key}", value, global_step)
                    for key, value in window_averaged.items():
                        writer.add_scalar(f"train_avg/{key}", value, global_step)
                    for key, value in window_median.items():
                        writer.add_scalar(f"train_median/{key}", value, global_step)

                if global_step % int(config.training.log_per_step) == 0:
                    log_metrics = {f"avg_{key}": value for key, value in window_averaged.items()}
                    log_metrics.update({f"med_{key}": value for key, value in window_median.items()})
                    logger.info(
                        "step=%d avg_window=%d %s",
                        global_step,
                        len(metric_history),
                        format_metrics(log_metrics),
                    )

                if (
                    int(config.training.save_per_step) > 0
                    and global_step % int(config.training.save_per_step) == 0
                    and is_main_process()
                ):
                    save_checkpoint(config, raw_model, optimizer, scheduler, global_step)

                if image_interval > 0 and global_step % image_interval == 0:
                    if is_main_process():
                        visual_batch = fixed_visual_batch if fixed_visual_batch is not None else batch
                        save_training_visualization(config, raw_flow, visual_batch, global_step, writer)
                    last_image_step = global_step

                metric_buffer = {}
                micro_in_accum = 0

                if config.training.max_steps is not None and global_step >= int(config.training.max_steps):
                    if is_main_process():
                        save_checkpoint(config, raw_model, optimizer, scheduler, global_step)
                    if writer is not None:
                        writer.close()
                    return

        # 处理一个 epoch 末尾可能剩下的不足 accumulation_steps 的 micro batch。
        if micro_in_accum > 0:
            if config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    float(config.training.grad_clip_norm),
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            averaged = {
                key: value / micro_in_accum
                for key, value in metric_buffer.items()
            }
            averaged["lr"] = scheduler.get_last_lr()[0]
            metric_history.append(averaged)
            logger.info("Epoch %d tail-step metrics: %s", epoch, averaged)

            if writer is not None and global_step % int(config.training.log_per_step) == 0:
                for key, value in averaged.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                for key, value in average_metric_history(metric_history).items():
                    writer.add_scalar(f"train_avg/{key}", value, global_step)
                for key, value in median_metric_history(metric_history).items():
                    writer.add_scalar(f"train_median/{key}", value, global_step)

            if config.training.max_steps is not None and global_step >= int(config.training.max_steps):
                if is_main_process():
                    save_checkpoint(config, raw_model, optimizer, scheduler, global_step)
                if writer is not None:
                    writer.close()
                return

        # 长训练会按 save_per_step 定期落盘；是否额外在每个 epoch 末尾保存由配置控制。
        if bool(getattr(config.training, "save_epoch_end", True)) and is_main_process():
            save_checkpoint(config, raw_model, optimizer, scheduler, global_step)

    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    try:
        train(args.config)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
