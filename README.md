# Pixel ResShift

Pixel ResShift combines a pMF-style pixel MeanFlow objective with ResShift/RealESRGAN-style low-resolution degradation for super-resolution research.

## What is included

- `pixel_resshift/`: the main training, model, data, optimization, evaluation, and visualization code.
- `pMF_torch/`: local pMF_torch source dependency used by the DiT backbone, Muon optimizer, ConvNeXt auxiliary loss, and pMF utilities.
- `ResShift/`: local ResShift source dependency used for dataset construction and online RealESRGAN-style degradation.

The repository does not include datasets, checkpoints, generated outputs, or experiment logs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Depending on your environment, ResShift/BasicSR may require additional packages such as `opencv-python`, `lmdb`, `facexlib`, or `realesrgan`.

## Smoke Test

Edit `pixel_resshift/examples/smoke_cpu.yaml` so that `source_path` points to a small image folder, then run:

```bash
python -m pixel_resshift.train --config pixel_resshift/examples/smoke_cpu.yaml
```

## Training

For LSDIR with online RealESRGAN-style degradation:

```bash
torchrun --nproc_per_node=4 -m pixel_resshift.train \
  --config pixel_resshift/examples/train_b16_lsdir.yaml
```

Before training, edit the example YAML paths:

- `data.train.params.root`: LSDIR root.
- `training.pretrained_pmf_path`: optional pMF full checkpoint directory or file.
- `output_dir`, `checkpoint_dir`, `log_dir`, `image_dir`: where experiment artifacts should be written.

## Evaluation and Visualization

The `pixel_resshift/examples/` directory contains scripts for:

- ImageNet-style SR evaluation.
- FID/MUSIQ/CLIPIQA evaluation.
- Degradation endpoint visualization.
- MeanFlow `z_t` / `x_pred` grids.
- Residual maps against LR-up or GT.

All scripts expect an explicit config and checkpoint path.

## Notes

- LR image conditioning is injected through cross-attention blocks in the DiT.
- The training endpoint uses `x1 = LR_up + sigma * noise`, with `sigma=0.8927` by default to match the effective ResShift input noise scale.
- Real ImageNet labels are disabled by default; the model uses the pMF class-token shape for checkpoint compatibility while feeding the null class token.

