#!/usr/bin/env bash
set -euo pipefail

CONFIG=/home/lkshpc/pixel_resshift/pixel_resshift_4/examples/train_b16_epoch_imagenet_server.yaml
ROOT=/data/share/imagenet
RUN_ROOT=/data/ynzhang/PixelResShift4
TRAIN_LOG="$RUN_ROOT/train_b16_epoch_imagenet.log"

mkdir -p "$RUN_ROOT"

echo "[$(date)] waiting for standard ImageNet extraction to finish"
while pgrep -f /tmp/prepare_imagenet_standard.sh >/dev/null; do
  train_dirs=$(find "$ROOT/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  train_images=$(find "$ROOT/train" -type f -name '*.JPEG' 2>/dev/null | wc -l)
  echo "[$(date)] ImageNet preparation is still running: train_dirs=$train_dirs train_images=$train_images"
  sleep 300
done

train_dirs=$(find "$ROOT/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
train_images=$(find "$ROOT/train" -type f -name '*.JPEG' 2>/dev/null | wc -l)

if [ "$train_dirs" -lt 1000 ] || [ "$train_images" -lt 1280000 ]; then
  echo "[$(date)] ImageNet train is not ready: train_dirs=$train_dirs train_images=$train_images"
  exit 1
fi

echo "[$(date)] ImageNet is ready: train_dirs=$train_dirs train_images=$train_images"
echo "[$(date)] launching B/16 one sampled epoch"

cd /home/lkshpc/pixel_resshift
source .venv/bin/activate

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m pixel_resshift_4.train --config "$CONFIG" >> "$TRAIN_LOG" 2>&1
