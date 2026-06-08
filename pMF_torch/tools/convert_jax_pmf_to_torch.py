"""Convert an official JAX/Orbax pMF checkpoint to pMF_torch checkpoints.

The official full checkpoints are saved as Flax/Orbax pytrees.  This script
maps the JAX parameter tree into the PyTorch module names used by pMF_torch.
It writes two checkpoints:

* ``eval/checkpoint_<step>.pt``: matches ``pixelMeanFlow(..., eval_mode=True)``.
* ``full/checkpoint_<step>.pt``: matches ``pixelMeanFlow(..., eval_mode=False)``.

The eval checkpoint is the one expected by ``pMF_torch/train.py --eval_only``.
The full checkpoint is useful for warm-starting or resuming finetuning.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp
import torch


def _add_repo_to_path(repo: Path) -> None:
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr)).float().contiguous()


def _linear_weight(arr: np.ndarray) -> torch.Tensor:
    # Flax Dense kernel is [in, out]; torch Linear weight is [out, in].
    return _tensor(np.asarray(arr).T)


def _conv_weight(arr: np.ndarray) -> torch.Tensor:
    # Flax Conv kernel is [H, W, in, out]; torch Conv2d is [out, in, H, W].
    return _tensor(np.asarray(arr).transpose(3, 2, 0, 1))


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for item in path.split("/"):
        cur = cur[item]
    return cur


def _copy_if_present(dst: OrderedDict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    dst[key] = value


def convert_net(net: dict[str, Any], include_v: bool) -> OrderedDict[str, torch.Tensor]:
    """Convert one JAX ``params['net']`` tree into a torch state dict."""

    out: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name in ("time_tokens", "class_tokens", "omega_tokens", "t_min_tokens", "t_max_tokens", "pos_embed"):
        _copy_if_present(out, f"net.{name}", _tensor(net[name]))

    # Patch embedding.
    _copy_if_present(out, "net.x_embedder.proj1.weight", _conv_weight(_get(net, "x_embedder/proj1/kernel")))
    _copy_if_present(out, "net.x_embedder.proj1.bias", _tensor(_get(net, "x_embedder/proj1/bias")))
    _copy_if_present(out, "net.x_embedder.proj2.weight", _conv_weight(_get(net, "x_embedder/proj2/kernel")))
    _copy_if_present(out, "net.x_embedder.proj2.bias", _tensor(_get(net, "x_embedder/proj2/bias")))

    # Time / CFG / interval embedders.
    for module in ("h_embedder", "omega_embedder", "cfg_t_start_embedder", "cfg_t_end_embedder"):
        for jax_layer, torch_layer in (("layers_0", "0"), ("layers_2", "2")):
            base = f"{module}/mlp/{jax_layer}/_flax_linear"
            _copy_if_present(out, f"net.{module}.mlp.{torch_layer}.weight", _linear_weight(_get(net, f"{base}/kernel")))
            _copy_if_present(out, f"net.{module}.mlp.{torch_layer}.bias", _tensor(_get(net, f"{base}/bias")))

    # Class embedding: JAX stores an extra leading axis.
    y_embed = np.asarray(_get(net, "y_embedder/embedding_table/embedding"))[0]
    _copy_if_present(out, "net.y_embedder.embedding_table.embedding.weight", _tensor(y_embed))

    def convert_block(jax_prefix: str, torch_prefix: str) -> None:
        block = net[jax_prefix]
        _copy_if_present(out, f"{torch_prefix}.attn_scale", _tensor(block["attn_scale"]))
        _copy_if_present(out, f"{torch_prefix}.mlp_scale", _tensor(block["mlp_scale"]))
        _copy_if_present(out, f"{torch_prefix}.norm1.weight", _tensor(_get(block, "norm1/kernel")))
        _copy_if_present(out, f"{torch_prefix}.norm2.weight", _tensor(_get(block, "norm2/kernel")))
        _copy_if_present(out, f"{torch_prefix}.attn.q_norm.weight", _tensor(_get(block, "attn/q_norm/kernel")))
        _copy_if_present(out, f"{torch_prefix}.attn.k_norm.weight", _tensor(_get(block, "attn/k_norm/kernel")))
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            _copy_if_present(
                out,
                f"{torch_prefix}.attn.{proj}.weight",
                _linear_weight(_get(block, f"attn/{proj}/_flax_linear/kernel")),
            )
        for proj in ("w1", "w2", "w3"):
            _copy_if_present(
                out,
                f"{torch_prefix}.mlp.{proj}.weight",
                _linear_weight(_get(block, f"mlp/{proj}/_flax_linear/kernel")),
            )

    for i in range(8):
        convert_block(f"shared_blocks_{i}", f"net.shared_blocks.{i}")
    for i in range(8):
        convert_block(f"u_heads_{i}", f"net.u_heads.{i}")
    if include_v:
        for i in range(8):
            convert_block(f"v_heads_{i}", f"net.v_heads.{i}")

    def convert_final(jax_prefix: str, torch_prefix: str) -> None:
        final = net[jax_prefix]
        _copy_if_present(out, f"{torch_prefix}.norm.weight", _tensor(_get(final, "norm/kernel")))
        _copy_if_present(out, f"{torch_prefix}.linear.weight", _linear_weight(_get(final, "linear/_flax_linear/kernel")))
        _copy_if_present(out, f"{torch_prefix}.linear.bias", _tensor(_get(final, "linear/_flax_linear/bias")))

    convert_final("u_final_layer", "net.u_final_layer")
    if include_v:
        convert_final("v_final_layer", "net.v_final_layer")

    return out


def filter_to_model(converted: OrderedDict[str, torch.Tensor], model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Keep only keys that exist in the target model and have matching shapes."""

    target = model.state_dict()
    filtered: OrderedDict[str, torch.Tensor] = OrderedDict()
    missing: list[str] = []
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, target_value in target.items():
        value = converted.get(key)
        if value is None:
            missing.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            mismatched.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        filtered[key] = value

    if missing or mismatched:
        raise RuntimeError(
            "Converted checkpoint does not match target model. "
            f"missing={missing[:10]} mismatched={mismatched[:10]}"
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-ckpt", required=True, type=Path)
    parser.add_argument("--torch-repo", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-str", default="pmfDiT_B_16")
    parser.add_argument("--ema", nargs="*", default=["250", "500", "1000", "2000"])
    args = parser.parse_args()

    _add_repo_to_path(args.torch_repo)
    from pmf import pixelMeanFlow

    print(f"Restoring JAX checkpoint: {args.jax_ckpt}")
    ckpt = ocp.PyTreeCheckpointer().restore(str(args.jax_ckpt.resolve()))
    step = int(np.asarray(ckpt.get("step", 0)))
    print(f"JAX checkpoint step: {step}")
    print(f"EMA keys: {sorted(ckpt.get('ema_params', {}).keys())}")

    eval_model = pixelMeanFlow(model_str=args.model_str, eval_mode=True)
    full_model = pixelMeanFlow(model_str=args.model_str, eval_mode=False)

    online_full = filter_to_model(convert_net(ckpt["params"]["net"], include_v=True), full_model)
    online_eval = filter_to_model(convert_net(ckpt["params"]["net"], include_v=False), eval_model)

    eval_ema: dict[int, OrderedDict[str, torch.Tensor]] = {}
    full_ema: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for ema_key in args.ema:
        ema_tree = ckpt["ema_params"][str(ema_key)]["net"]
        eval_ema[int(ema_key)] = filter_to_model(convert_net(ema_tree, include_v=False), eval_model)
        full_ema[int(ema_key)] = filter_to_model(convert_net(ema_tree, include_v=True), full_model)

    eval_dir = args.out_dir / "eval"
    full_dir = args.out_dir / "full"
    eval_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    eval_path = eval_dir / f"checkpoint_{step}.pt"
    full_path = full_dir / f"checkpoint_{step}.pt"

    torch.save({"step": step, "model_state_dict": online_eval, "ema_params": eval_ema}, eval_path)
    torch.save({"step": step, "model_state_dict": online_full, "ema_params": full_ema}, full_path)

    print(f"Wrote eval checkpoint: {eval_path}")
    print(f"Wrote full checkpoint: {full_path}")
    print(f"Eval params: {len(online_eval)} tensors; full params: {len(online_full)} tensors")


if __name__ == "__main__":
    main()
