#!/bin/bash
#
#Use original Temporal Training but on Lab pretrained AlexNet, use color distortion
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/fineTune_distort_10sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/fineTune_distort_10sec%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12


python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /data/movie-associations/saves/temporal/finetune10sec/movie-training-distorted \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/10sec_distorted \
    --time_lag 10 \
    --view temporal \
    --lr_decay_epochs 30,50,70 \
    --epochs 80 \
    --distort True \
    --resume /data/movie-associaions/saves/temporal/finetune10sec/movie-training-distorted/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_10_view_temporal/ckpt_epoch_78.pth