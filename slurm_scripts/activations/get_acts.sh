#!/bin/bash
#

python3 /home/clionaodoherty/CMC/anna_get_acts.py \
    --model_path /data/movie-associations/saves/temporal/1min/distorted/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_60_view_temporal/ckpt_epoch_200.pth \
    --save_path /home/clionaodoherty/CMC/activations/full_temporal_distort/60sec_full_distort.pickle \
    --transform distort
