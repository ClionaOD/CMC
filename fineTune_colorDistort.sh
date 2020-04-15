#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet, use color distortion
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_distort_1sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_distort_1sec%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/movie-training-distorted \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/1sec/distorted \
    --time_lag 1 \
    --view temporal \
    --pretrained /home/clionaodoherty/movie-associations/saves/Lab_pretrained_fullAlexNet.pth \
    --lr_decay_epochs 30,50,70 \
    --epochs 80 \
    --distort True