#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet, split Lab across time
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_split_1sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_split_1sec%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/Lab_Split \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/1sec/lab_split \
    --time_lag 1 \
    --view temporal \
    --split_Lab True \
    --pretrained /home/clionaodoherty/CMC/models/CMC_alexnet.pth \
    --lr_decay_epochs 30,50,70 \
    --epochs 80 