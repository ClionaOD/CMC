#!/bin/bash
#

python3 /home/clionaodoherty/CMC/anna_get_acts.py \
    --model_path /data/movie-associations/saves/temporal/finetune1sec/movie-training-distorted/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_1_view_temporal/ckpt_epoch_80.pth \
    --save_path /home/clionaodoherty/CMC/activations/temporal_lab/act_only/1sec.pickle \
    --transform Lab