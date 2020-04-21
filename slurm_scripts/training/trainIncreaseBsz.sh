#!/bin/bash
#
#Train the temporal CMC using original lr scaling and higher bsz

CUDA_VISIBLE_DEVICES=0,1 python3 /home/clionaodoherty/CMC/train_temporal.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/chen_updates/movie-pretrain \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --batch_size 512 \
    --learning_rate 0.6 \
    --epochs 140 \
    --lr_decay_epochs 60,100,120 \
    --resume /home/clionaodoherty/movie-associations/saves/temporal/chen_updates/movie-pretrain/memory_nce_16384_alexnet_lr_0.6_decay_1e-05_bsz_512_sec_60_view_temporal/ckpt_epoch_50.pth