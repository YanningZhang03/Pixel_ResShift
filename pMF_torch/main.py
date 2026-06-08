"""Main entry point for pixel MeanFlow training/evaluation — pure PyTorch."""

import importlib.util
import os
import logging
from pathlib import Path
import warnings

import torch
import torch.distributed as dist
import ml_collections
import yaml

from absl import app, flags

import train
from configs.default import get_config as get_default_config
from utils.logging_util import log_for_0

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("debug", False, "Debugging mode.")
flags.DEFINE_string(
    "config",
    None,
    "Path to config file (.yml/.yaml/.py).",
)


def _load_config(path: str) -> ml_collections.ConfigDict:
    def _merge_dict_into_config(dst: ml_collections.ConfigDict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict):
                if k in dst and isinstance(dst[k], ml_collections.ConfigDict):
                    _merge_dict_into_config(dst[k], v)
                else:
                    dst[k] = ml_collections.ConfigDict(v)
            else:
                dst[k] = v

    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        config = get_default_config()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must be a mapping at top-level: {path}")
        _merge_dict_into_config(config, data)
        return config

    if suffix == ".py":
        spec = importlib.util.spec_from_file_location("pmf_runtime_config", str(config_path))
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python config file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "get_config"):
            raise ValueError(f"Python config must provide get_config(): {path}")
        cfg = module.get_config()
        return cfg if isinstance(cfg, ml_collections.ConfigDict) else ml_collections.ConfigDict(cfg)

    raise ValueError(f"Unsupported config extension: {suffix}. Use .yml/.yaml/.py")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    config = _load_config(FLAGS.config)

    # Initialize DDP via torchrun environment variables
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log_for_0("PyTorch DDP process: %d / %d", rank, world_size)
    log_for_0("CUDA devices: %s", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    log_for_0("Config:\n%s", config)

    if config.get("eval_only", False):
        train.just_evaluate(config, FLAGS.workdir)
    else:
        train.train_and_evaluate(config, FLAGS.workdir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
