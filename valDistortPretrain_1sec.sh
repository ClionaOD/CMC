#!/bin/bash
#
#SBATCH --output=/home/clionaodoherty/movie-associations/logs/val_distort_pre1sec%j.out
#SBATCH --error=/home/clionaodoherty/movie-associations/logs/val_distort_pre1sec%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12

python3 /home/clionaodoherty/CMC/LinearProbing.py \
    --data_folder /data/ILSVRC2012 \
    --save_path /home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/imgnet-val-1sec \
    --tb_path /home/clionaodoherty/CMC/tensorboard/finetune/val \
    --model_path /home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/movie-training-1sec/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_1_view_temporal/ckpt_epoch_80.pth \
    --view temporal \
    --distort True 