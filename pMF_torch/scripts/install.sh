#!/bin/bash
# pMF PyTorch environment installation
conda create -n pMF python=3.12 -y
conda activate pMF

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install timm lpips wandb ml-collections absl-py pillow numpy scipy
pip install pyyaml tqdm
