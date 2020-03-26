#!/bin/bash
#
#Train the temporal CMC using implementations in Chen et al. 2020

CUDA_VISIBLE_DEVICES=0,1 python3 /home/clionaodoherty/CMC/train_temporal.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/chen_updates/movie-pretrain \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --resume /home/clionaodoherty/movie-associations/saves/temporal/chen_updates/movie-pretrainmemory_nce_16384_alexnet_lr_0.18_decay_1e-05_bsz_156_sec_60_view_temporal/ckpt_epoch_280.pth \
    --batch_size 156 \
    --learning_rate 0.18 \
    --epochs 400