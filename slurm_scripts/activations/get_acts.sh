#!/bin/bash
#

python3 /home/clionaodoherty/CMC/anna_get_acts.py \
    --model_path /data/movie-associations/saves/temporal/1min/movie-pretrain-1min/ckpt_epoch_220.pth \
    --save_path /home/clionaodoherty/CMC/activations/1min_fullytrained.pickle \
    --transform Lab
