#!/bin/bash
#
#Use original Temporal Training but on 60 min timelag
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/slurm%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/slurm%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/movie-pretrain-60min \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --time_lag 3600 \
    --view temporal \