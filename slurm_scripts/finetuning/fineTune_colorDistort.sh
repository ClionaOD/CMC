#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet, use color distortion
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_distort_5min%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_distort_5min%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /data/movie-associations/saves/temporal/finetune5min/movie-training-distorted \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/5min_distorted \
    --time_lag 300 \
    --view temporal \
    --pretrained /data/movie-associations/saves/Lab_pretrained_fullAlexNet.pth \
    --lr_decay_epochs 30,50,70 \
    --epochs 80 \
    --distort True 