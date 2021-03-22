#!/bin/bash
export WANDB_MODE=dryrun

python3 train.py \
    --batch_size 32 \
    --num_workers 4 \
    --gpus 1 \
    ${@:1}
