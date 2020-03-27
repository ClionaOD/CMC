#!/bin/bash
#
#Continue Chen training with adjust learning rate

CUDA_VISIBLE_DEVICES=0,1 python3 /home/clionaodoherty/CMC/train_temporal.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/chen_updates/movie-pretrain \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --learning_rate 0.3 \
    --batch_size 256