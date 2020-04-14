#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_5sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_5sec%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/finetune5sec \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/5sec \
    --time_lag 5 \
    --view temporal \
    --pretrained /home/clionaodoherty/movie-associations/saves/Lab_pretrained_fullAlexNet.pth \
    --lr_decay_epochs 30,50,70 \
    --epochs 80 