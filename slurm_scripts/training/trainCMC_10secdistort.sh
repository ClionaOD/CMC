#!/bin/bash
#
#Use original full Temporal Training but on 10 sec timelag, distorted
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/10sec_fulldistort%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/10sec_fulldistort%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /data/movie-associations/saves/temporal/10sec/distorted \
    --tb_path /home/clionaodoherty/CMC/tensorboard/10sec \
    --time_lag 10 \
    --view temporal \
    --lr_decay_epochs 120,160,200 \
    --epochs 220 \
    --distort True 