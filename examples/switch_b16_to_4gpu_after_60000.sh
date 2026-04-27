#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data/ynzhang/PixelResShift4/train_b16_20epoch_ddp_bs12_imagenet_train_rsnoise_img1000_ckpt10000"
CKPT="${OUT_DIR}/checkpoints/pixel-resshift-4-step-60000.pt"
CONFIG="pixel_resshift_4/examples/train_b16_epoch_ddp_bs12_rsnoise_server_4gpu_from60000.yaml"
FALLBACK_CONFIG="pixel_resshift_4/examples/train_b16_epoch_ddp_bs12_rsnoise_server_2gpu_from60000.yaml"
OLD_PID="2432670"
GPUS="0,1,6,7"
FALLBACK_GPUS="1,6"

echo "[$(date)] watcher started; waiting for ${CKPT}"

while true; do
  if [[ -f "${CKPT}" ]]; then
    size_1="$(stat -c '%s' "${CKPT}")"
    sleep 20
    size_2="$(stat -c '%s' "${CKPT}")"
    if [[ "${size_1}" == "${size_2}" && "${size_1}" -gt 1000000000 ]]; then
      echo "[$(date)] checkpoint is stable: ${CKPT} (${size_2} bytes)"
      break
    fi
    echo "[$(date)] checkpoint exists but is still changing: ${size_1} -> ${size_2}"
  fi
  sleep 60
done

echo "[$(date)] checking spare GPUs before stopping the current job"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

for gpu in 0 7; do
  used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v g="${gpu}" '$1 == g {gsub(/ /, "", $2); print $2}')"
  if [[ -z "${used}" || "${used}" -gt 4096 ]]; then
    echo "[$(date)] GPU ${gpu} is not free enough (${used:-unknown} MiB used); keep old job running and abort switch"
    exit 1
  fi
done

echo "[$(date)] stopping old two-GPU torchrun pid ${OLD_PID}"
if kill -0 "${OLD_PID}" 2>/dev/null; then
  pkill -TERM -P "${OLD_PID}" 2>/dev/null || true
  kill -TERM "${OLD_PID}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${OLD_PID}" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[$(date)] old pid did not exit after TERM; sending KILL"
    pkill -KILL -P "${OLD_PID}" 2>/dev/null || true
    kill -KILL "${OLD_PID}" 2>/dev/null || true
  fi
fi

sleep 10
echo "[$(date)] GPU state after stopping old job"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

cd /home/lkshpc/pixel_resshift
echo "[$(date)] launching four-GPU DDP on CUDA_VISIBLE_DEVICES=${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" nohup .venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m pixel_resshift_4.train \
  --config "${CONFIG}" \
  >> "${OUT_DIR}/train.log" 2>&1 &
new_pid="$!"
echo "[$(date)] launched four-GPU torchrun pid ${new_pid}"

sleep 180
if ! kill -0 "${new_pid}" 2>/dev/null; then
  echo "[$(date)] four-GPU launch exited early; falling back to two GPUs ${FALLBACK_GPUS}"
  CUDA_VISIBLE_DEVICES="${FALLBACK_GPUS}" nohup .venv/bin/torchrun --standalone --nproc_per_node=2 \
    -m pixel_resshift_4.train \
    --config "${FALLBACK_CONFIG}" \
    >> "${OUT_DIR}/train.log" 2>&1 &
  fallback_pid="$!"
  echo "[$(date)] launched fallback two-GPU torchrun pid ${fallback_pid}"
fi
