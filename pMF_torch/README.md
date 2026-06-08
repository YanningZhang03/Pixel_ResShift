# Pixel Mean Flows (PyTorch)

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2601.22158-b31b1b.svg)](https://arxiv.org/abs/2601.22158)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&nbsp;
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-pMF-blue.svg)](https://huggingface.co/Lyy0725/pMF)&nbsp;


<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

This is a **pure PyTorch** implementation of the paper [One-step Latent-free Image Generation with Pixel Mean Flows](https://arxiv.org/abs/2601.22158), converted from the original JAX/Flax codebase. It supports multi-GPU training via PyTorch Distributed Data Parallel (DDP).

For the original JAX implementation (TPU), please refer to the [main branch](https://github.com/Lyy-iiis/pMF).
For HSDP implementation, please refer to the [hsdp branch](https://github.com/Lyy-iiis/pMF/tree/hsdp).

## Installation

Install the required dependencies:

```bash
bash scripts/install.sh
wandb login YOUR_WANDB_API_KEY  # optional, for experiment tracking
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.1 with CUDA support
- Key dependencies: `timm`, `lpips`, `wandb`, `ml-collections`

## Inference

You can quickly verify your setup with the provided checkpoints.

<table><tbody>
<td valign="bottom">ImageNet 256x256</td>
<td valign="bottom" align="center">pMF-B/16</td>
<td valign="bottom" align="center">pMF-L/16</td>
<td valign="bottom" align="center">pMF-H/16</td>
<tr><td align="left">pre-trained checkpoint (inference) </td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-B-16.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-L-16.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-H-16.zip">download</td>
</tr>
<tr><td align="left">pre-trained checkpoint (full) </td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-B-16-full.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-L-16-full.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-H-16-full.zip">download</td>
</tr>
<tr><td align="left">FID (this repo / original paper)</td>
<td align="center">3.11/3.12</td>
<td align="center">2.50/2.52</td>
<td align="center">2.11/2.22</td>
</tr>
<tr><td align="left">IS (this repo / original paper)</td>
<td align="center">256.4/254.6</td>
<td align="center">266.0/262.6</td>
<td align="center">270.5/268.8</td>
</tr>
</tbody></table>

<table><tbody>
<td valign="bottom">ImageNet 512x512</td>
<td valign="bottom" align="center">pMF-B/32</td>
<td valign="bottom" align="center">pMF-L/32</td>
<td valign="bottom" align="center">pMF-H/32</td>
<tr><td align="left">pre-trained checkpoint (inference) </td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-B-32.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-L-32.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-H-32.zip">download</td>
</tr>
<tr><td align="left">pre-trained checkpoint (full) </td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-B-32-full.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-L-32-full.zip">download</td>
<td align="center"><a href="https://huggingface.co/Lyy0725/pMF/blob/main/pMF-H-32-full.zip">download</td>
</tr>
<tr><td align="left">FID (this repo / original paper)</td>
<td align="center">3.64/3.70</td>
<td align="center">2.73/2.75</td>
<td align="center">2.37/2.48</td>
</tr>
<tr><td align="left">IS (this repo / original paper)</td>
<td align="center">274.4/271.9</td>
<td align="center">276.6/276.8</td>
<td align="center">285.3/284.9</td>
</tr>
</tbody></table>

> **Note:** Slight differences in FID/IS may arise due to different computation setups. The reference results above were computed on TPU v5p-64 with the original JAX code.

#### Sanity Check

1. **Download the checkpoint and FID stats:**
    - Download a pre-trained checkpoint from the table above.
    - Download the FID stats file: [256x256](https://huggingface.co/Lyy0725/pMF/blob/main/imagenet_256_fid_stats.npz) or [512x512](https://huggingface.co/Lyy0725/pMF/blob/main/imagenet_512_fid_stats.npz). You can also recompute FID stats using `prepare_ref.py` if needed.

2. **Unzip the checkpoint:**
    ```bash
    unzip <downloaded_checkpoint.zip> -d <your_ckpt_dir>
    ```

3. **Set up the config:**
    - Set `load_from` in `configs/eval_config.yml` to point to `<your_ckpt_dir>`.
    - Set `fid.cache_ref` to the path of the downloaded FID stats file.
    - Set the model parameters accordingly, e.g., `model.model_str` and `sampling`.

4. **Launch evaluation:**
    ```bash
    bash scripts/eval.sh JOB_NAME
    ```
    The default evaluation script generates 50,000 samples using pMF-B/16 for FID and IS evaluation.

## Data Preparation

#### Option 1: Interactive Download Tool

We provide a script that handles downloading, extracting, and organizing ImageNet-1k automatically. It uses `aria2c` for fast multi-connection downloading with resume support.

```bash
# Requires aria2c: sudo apt install aria2
python scripts/prepare_imagenet.py
```

The interactive menu provides the following options:
1. **Download** — downloads the training set (~138 GB), validation set (~6.3 GB), and devkit via `aria2c`
2. **Extract training set** — unpacks the main archive and organizes images into 1000 per-class directories
3. **Extract & classify validation set** — unpacks validation images and sorts them into per-class directories using ground truth labels
4. **Full pipeline** — runs all of the above in sequence

> **Note:** You need a valid ImageNet account to access the download URLs. Visit [image-net.org](http://image-net.org/download) to register.

#### Option 2: Manual Download

If you already have the ImageNet dataset or prefer to download it manually, simply ensure the directory structure matches:

```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Configuration

Edit your config file (e.g., `configs/pMF_B_16_config.yml`):

```yaml
dataset:
    root: YOUR_DATA_ROOT        # path to your ImageNet directory

fid:
    cache_ref: YOUR_FID_CACHE   # path to FID statistics file

logging:
    wandb_project: 'YOUR_PROJECT'  # WandB project name (optional)
```

**Available config files:**
- `configs/pMF_B_16_config.yml` — pMF-B/16 model (recommended starting point)
- `configs/pMF_B_32_config.yml` — pMF-B/32 model
- `configs/pMF_L_16_config.yml` — pMF-L/16 model
- `configs/pMF_L_32_config.yml` — pMF-L/32 model
- `configs/default.py` — Base configuration with all default hyperparameters

The config system uses a hierarchical approach: YAML config files override specific parameters from `default.py`.

## Training

Launch multi-GPU training with DDP:

```bash
bash scripts/train.sh JOB_NAME
```

**Before running**, update the environment variables in `scripts/train.sh`:
- `DATA_ROOT`: Path to your ImageNet directory
- `LOG_DIR`: Path for saving training logs and checkpoints
- `NGPU`: Number of GPUs to use (defaults to all available)

The training script uses `torchrun` for DDP launch. To customize the number of GPUs:

```bash
NGPU=4 bash scripts/train.sh my_experiment
```

#### Custom Experiments

1. Create a new config file (e.g., `configs/my_exp_config.yml`):

```yaml
training:
    num_epochs: 80

model:
    model_str: pmfDiT_B_16
    noise_scale: 1.0
```

2. Update the launch script to use your config:
   ```bash
   --config=configs/load_config.py:my_exp
   ```

For a complete list of configuration options, refer to `configs/default.py`.

## Optimizer

This repo uses the **Muon** (MomentUm Orthogonalized by Newton-schulz) optimizer by default, matching the original JAX codebase which uses `optax.contrib.muon`. The implementation is in `utils/muon.py`, based on [Keller Jordan's reference](https://github.com/KellerJordan/Muon).

Muon automatically partitions parameters into two groups:
- **2D hidden weight matrices** — optimized with SGD-momentum + Newton-Schulz orthogonalization (5-step quintic iteration in bfloat16)
- **Embeddings, biases, layernorms, output heads** — optimized with AdamW

To switch to pure AdamW, set in your config:

```yaml
training:
    optimizer: adamw
```

## Key Differences from the JAX Version

| Component | JAX (original) | PyTorch (this repo) |
|---|---|---|
| Framework | JAX / Flax | PyTorch |
| JVP computation | `jax.jvp` | `torch.func.jvp` + `functional_call` |
| Distributed training | `jax.pmap` | `torch.nn.parallel.DistributedDataParallel` |
| Optimizer | `optax.contrib.muon` | `MuonAdamW` (Muon + AdamW hybrid) |
| Perceptual loss | `lpips_j` (JAX) | `lpips` (PyTorch) |
| ConvNeXt features | Custom JAX impl | `timm` pretrained model |
| FID computation | Custom JAX InceptionV3 | `torchvision` InceptionV3 |
| Hardware target | TPU | GPU (CUDA) |

## License

This repo is under the MIT license. See [LICENSE](./LICENSE) for details.

## Citation

If you find this work useful in your research, please consider citing:

```bib
@article{pixelmeanflows,
  title={One-step Latent-free Image Generation with Pixel Mean Flows},
  author={Lu, Yiyang and Lu, Susie and Sun, Qiao and Zhao, Hanhong and Jiang, Zhicheng and Wang, Xianbang and Li, Tianhong and Geng, Zhengyang and He, Kaiming},
  journal={arXiv preprint arXiv:2601.22158},
  year={2026}
}
```

## Contributors

This repository is a collaborative effort by Kaiming He, Hanhong Zhao, Qiao Sun and Yiyang Lu, developed in support of several research projects, including [MeanFlow](https://arxiv.org/abs/2505.13447), [improved MeanFlow](https://arxiv.org/abs/2512.02012), and [BiFlow](https://arxiv.org/abs/2512.10953).

## Acknowledgement

We gratefully acknowledge the Google TPU Research Cloud (TRC) for granting TPU access for the original JAX implementation.
We hope this work will serve as a useful resource for the open-source community.
