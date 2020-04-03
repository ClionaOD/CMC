#!/bin/bash
#
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/val_5min%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/val_5min%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12

python3 /home/clionaodoherty/CMC/LinearProbing.py \
    --data_folder /data/ILSVRC2012 \
    --save_path /home/clionaodoherty/movie-associations/saves/temporal/imgnet-val-5min \
    --tb_path /home/clionaodoherty/CMC/tensorboard/val \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/movie-pretrain-5min/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_300_view_temporal/ckpt_epoch_220.pth