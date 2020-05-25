#!/bin/bash
#

python3 /home/clionaodoherty/CMC/anna_get_acts.py \
    --model_path /data/movie-associations/saves/Lab/movie-pretrain/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_200.pth \
    --save_path /home/clionaodoherty/CMC/activations/movie_lab.pickle \
    --transform Lab
