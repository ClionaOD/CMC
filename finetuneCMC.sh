#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_5min%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_5min%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/finetune5min \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune \
    --time_lag 300 \
    --view temporal \
    --pretrained /home/clionaodoherty/movie-associations/saves/Lab/movie-pretrain/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_200.pth