#!/bin/bash
#

python3 /home/clionaodoherty/CMC/anna_get_acts.py \
    --model_path /data/movie-associations/saves/temporal/finetune5min/movie-training-distorted/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_300_view_temporal/ckpt_epoch_80.pth \
    --save_path /home/clionaodoherty/CMC/activations/supervised/alexnet_superv.pickle \
    --transform distort \
    --supervised True