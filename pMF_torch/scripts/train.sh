#!/bin/bash

# Note: Update dataset.root and fid.cache_ref in your config YAML.
export DATA_ROOT="YOUR_OUTPUT_DIR_FROM_DATA_PREPARATION"
export LOG_DIR="YOUR_LOG_DIR"

export now=$(date '+%Y%m%d_%H%M%S')
export salt=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
export JOBNAME=${now}_${salt}_$1
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${LOG_DIR}

# Number of GPUs — adjust as needed
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}

torchrun --nproc_per_node=${NGPU} main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:pMF_B_16 \
    2>&1 | tee -a $LOG_DIR/output.log
