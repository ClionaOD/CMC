#!/bin/bash
#
#Use original Temporal Training but on 5 min timelag
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/5min%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/5min%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/movie-pretrain-5min \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --time_lag 300 \
    --view temporal \
    --resume /home/clionaodoherty/movie-associations/saves/temporal/movie-pretrain-5min/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_300_view_temporal/ckpt_epoch_61.pth
