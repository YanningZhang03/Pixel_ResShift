"""Training and evaluation for pixel MeanFlow — pure PyTorch with DDP."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import ml_collections

from pmf import pixelMeanFlow
import utils.input_pipeline as input_pipeline
from utils.ckpt_util import save_checkpoint, restore_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer
from utils.vis_util import make_grid_visualization
from utils.lr_utils import lr_schedules
from utils.sample_util import get_fid_evaluator, run_sample
from utils.trainstate_util import create_model_and_optimizer
from utils.auxloss_util import init_auxloss


def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def _local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def _get_precision_context(config: ml_collections.ConfigDict, device: torch.device):
    precision = str(config.training.get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp32' or 'bf16'.")

    amp_enabled = device.type == "cuda" and precision == "bf16"

    def _ctx():
        if amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    return precision, _ctx


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    writer = Writer(config, workdir)
    device = torch.device("cuda", _local_rank())
    torch.cuda.set_device(device)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    precision, autocast_ctx = _get_precision_context(config, device)
    log_for_0(f"Training precision: {precision}")
    if precision == "bf16":
        log_for_0(
            "bf16 is enabled for eval/sampling only; training forward uses fp32 for"
            " stability with torch.func.jvp."
        )

    global_batch = config.training.batch_size
    local_batch = global_batch // world_size
    log_for_0(f"Global batch: {global_batch}, local: {local_batch}, world: {world_size}")

    train_loader, steps_per_epoch = input_pipeline.create_imagenet_split(
        config.dataset, local_batch, split="train")
    use_flip = config.dataset.use_flip
    log_for_0(f"Steps per epoch: {steps_per_epoch}")

    model = pixelMeanFlow(**config.model.to_dict())
    lr_lambda, base_lr = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    model, optimizer, scheduler, ema_params = create_model_and_optimizer(
        config, model, lr_lambda, base_lr, device)

    ddp_model = DDP(model, device_ids=[_local_rank()]) if dist.is_initialized() and world_size > 1 else model
    raw_model = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model

    step_offset = 0
    if config.load_from:
        step_offset, ema_loaded = restore_checkpoint(config.load_from, raw_model, optimizer, scheduler)
        if ema_loaded:
            ema_params = ema_loaded
    epoch_offset = step_offset // steps_per_epoch

    aux_fn = init_auxloss(config, device) if (config.model.convnext or config.model.lpips) else None
    if aux_fn:
        log_for_0("Using perceptual auxiliary loss")
    else:
        log_for_0("Not using perceptual auxiliary loss")

    fid_evaluator = get_fid_evaluator(config, writer, raw_model, device)
    sample_kwargs = dict(omega=config.sampling.omega, t_min=config.sampling.t_min, t_max=config.sampling.t_max)

    metrics_tracker = MetricsTracker()
    timer = Timer()

    for epoch in range(epoch_offset, config.training.num_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        log_for_0(f"Epoch {epoch}...")

        if (epoch + 1) % config.training.sample_per_epoch == 0:
            raw_model.eval()
            with torch.no_grad():
                with autocast_ctx():
                    vis = run_sample(raw_model, 16, config, device, sample_idx=0, **sample_kwargs)
            vis_grid = make_grid_visualization(vis, grid=4)
            writer.write_images(step_offset + 1, {"vis_sample": vis_grid[0]})
            raw_model.train()

        ddp_model.train()
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            step = epoch * steps_per_epoch + n_batch
            images, labels = batch
            if isinstance(images, torch.Tensor) and images.dtype == torch.uint8:
                images = input_pipeline.process_images(images.to(device), use_flip=use_flip)
            else:
                images = images.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad(set_to_none=True)
            with nullcontext():
                loss, dict_losses = ddp_model(images, labels, aux_fn=aux_fn)
            loss.backward()
            optimizer.step()
            scheduler.step()

            for k, ema_sd in ema_params.items():
                ema_params[k] = update_ema(ema_sd, raw_model.state_dict(), ema_fn(step, k))

            dict_losses["lr"] = scheduler.get_last_lr()[0]
            metrics_tracker.update(dict_losses)
            if (step + 1) % config.training.log_per_step == 0:
                summary = metrics_tracker.finalize()
                summary["steps_per_second"] = config.training.log_per_step / timer.elapse_with_reset()
                summary["epoch"] = epoch
                writer.write_scalars(step + 1, summary)
            step_offset = step

        if _is_main() and ((epoch + 1) % config.training.checkpoint_per_epoch == 0 or (epoch + 1) == config.training.num_epochs):
            save_checkpoint(workdir, step_offset + 1, raw_model, optimizer, scheduler, ema_params)

        if (epoch + 1) % config.training.fid_per_epoch == 0 or (epoch + 1) == config.training.num_epochs:
            raw_model.eval()
            with autocast_ctx():
                fid_evaluator(step_offset, ema_params, **sample_kwargs)
            raw_model.train()

    if dist.is_initialized():
        dist.barrier()
    log_for_0("Training complete.")


def just_evaluate(config: ml_collections.ConfigDict, workdir: str):
    assert config.eval_only and config.load_from
    writer = Writer(config, workdir)
    device = torch.device("cuda", _local_rank())
    torch.cuda.set_device(device)
    precision, autocast_ctx = _get_precision_context(config, device)
    log_for_0(f"Evaluation precision: {precision}")

    config.training.ema_val = config.sampling.emas
    model = pixelMeanFlow(**config.model.to_dict(), eval_mode=True).to(device)
    lr_lambda, base_lr = lr_schedules(config, 1000)
    # For eval-only, use a simple AdamW to satisfy checkpoint loading; optimizer state is unused
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    step, ema_params = restore_checkpoint(config.load_from, model, optimizer)
    fid_evaluator = get_fid_evaluator(config, writer, model, device)

    best_fid, best_is, best_config = float("inf"), float("-inf"), None
    model.eval()
    for ema in config.sampling.emas:
        for interval in config.sampling.interval:
            t_min_v, t_max_v = interval
            for omega in config.sampling.omegas:
                with autocast_ctx():
                    fid, is_score = fid_evaluator(step, ema_params, ema_only=True,
                                                  omega=omega, t_min=t_min_v, t_max=t_max_v, ema=ema)
                if fid < best_fid:
                    best_fid, best_is, best_config = fid, is_score, (omega, t_min_v, t_max_v, ema)

    omega, t_min_v, t_max_v, ema = best_config
    log_for_0(f"Best FID: {best_fid:.2f}, IS: {best_is:.2f}")
    writer.write_scalars(step + 1, dict(best_fid=best_fid, best_is=best_is,
                                         omega=omega, t_min=t_min_v, t_max=t_max_v, ema=ema))
    if dist.is_initialized():
        dist.barrier()
