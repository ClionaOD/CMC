#!/bin/bash
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/val_distort_pre60sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/val_distort_pre60sec%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

python3 /home/clionaodoherty/CMC/LinearProbing.py \
    --data_folder /data/ILSVRC2012 \
    --save_path /data/movie-associations/saves/temporal/finetune60sec/imgnet-val \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/60sec_distorted/val \
    --model_path /data/movie-associations/saves/temporal/finetune60sec/movie-training-distorted/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_60_view_temporal/ckpt_epoch_80.pth \
    --view temporal \
    --distort True 