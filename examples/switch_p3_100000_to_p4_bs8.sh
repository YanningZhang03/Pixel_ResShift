#!/usr/bin/env bash
set -u

cd /home/lkshpc/pixel_resshift

P3_OUT=/data/ynzhang/PixelResShift3/train_b16_20epoch_ddp_bs12_imagenet_train_rsnoise_img1000_ckpt10000
P3_CKPT="$P3_OUT/checkpoints/pixel-resshift-3-step-100000.pt"

P4_OUT=/data/ynzhang/PixelResShift4/train_b16_2000_ddp_bs8_imagenet_train_rsnoise_img1000_ckpt10000
P4_CFG=pixel_resshift_4/examples/train_b16_2000_ddp_bs8_rsnoise_server_4gpu.yaml
P4_LOG="$P4_OUT/train.log"

mkdir -p "$P4_OUT"

echo "[$(date)] waiting for $P3_CKPT"
while [ ! -s "$P3_CKPT" ]; do
  tail -n 80 "$P3_OUT/train.log" | grep -a 'step=' | tail -1 || true
  sleep 30
done

# checkpoint 可能刚被 torch.save 创建，等文件大小稳定后再停第三版。
prev=-1
while true; do
  size=$(stat -c%s "$P3_CKPT")
  sleep 10
  size2=$(stat -c%s "$P3_CKPT")
  if [ "$size" = "$size2" ] && [ "$size" != "$prev" ]; then
    break
  fi
  prev=$size
done

echo "[$(date)] checkpoint is stable: $(ls -lh "$P3_CKPT")"

main_pid=$(pgrep -f 'torchrun .*pixel_resshift_3.train' | head -1 || true)
if [ -n "$main_pid" ]; then
  echo "[$(date)] stopping pixel_resshift_3 torchrun pid=$main_pid"
  kill -TERM "$main_pid" || true
  for _ in $(seq 1 60); do
    if ! pgrep -f 'pixel_resshift_3.train' >/dev/null; then
      break
    fi
    sleep 1
  done
fi

if pgrep -f 'pixel_resshift_3.train' >/dev/null; then
  echo "[$(date)] force stopping remaining pixel_resshift_3 workers"
  pkill -KILL -f 'pixel_resshift_3.train' || true
fi

sleep 5

echo "[$(date)] starting pixel_resshift_4 bs8 4gpu test"
CUDA_VISIBLE_DEVICES=0,1,6,7 nohup .venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m pixel_resshift_4.train \
  --config "$P4_CFG" \
  > "$P4_LOG" 2>&1 &

pid=$!
echo "$pid" > "$P4_OUT/torchrun.pid"
echo "[$(date)] pixel_resshift_4 torchrun pid=$pid log=$P4_LOG"

sleep 45

echo "[$(date)] p4 process check"
pgrep -af 'torchrun.*pixel_resshift_4|pixel_resshift_4.train' | head -20 || true

echo "[$(date)] p4 log tail"
tail -n 80 "$P4_LOG" || true
