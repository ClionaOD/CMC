#!/bin/bash
#
#Use original full Temporal Training but on 1 sec timelag, distorted
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/1sec_fulldistort%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/1sec_fulldistort%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /data/movie-associations/saves/temporal/1sec/distorted \
    --tb_path /home/clionaodoherty/CMC/tensorboard/1sec \
    --time_lag 1 \
    --view temporal \
    --lr_decay_epochs 120,160,200 \
    --epochs 220 \
    --distort True \
    --resume /data/movie-associations/saves/temporal/1sec/distorted/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_1_view_temporal/ckpt_epoch_114.pth