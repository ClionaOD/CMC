#!/bin/bash
#
#Use original Lab Training but on 5 min timelag

CUDA_VISIBLE_DEVICES=0,1 python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/movie-pretrain-5min \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --time_lag 300