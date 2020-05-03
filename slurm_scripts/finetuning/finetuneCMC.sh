#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_30sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_30sec%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /data/movie-associations/saves/temporal/finetune30sec/movie-training-30sec \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/30sec \
    --time_lag 30 \
    --view temporal \
    --pretrained /data/movie-associations/saves/Lab_pretrained_fullAlexNet.pth \
    --lr_decay_epochs 30,50,70 \
    --epochs 80 