#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J train_CMC
#SBATCH --output=/movie-associations/logs/slurm-%j.out
#SBATCH --error=/movie-associations/logs/slurm-%j.err


python /home/ubuntu/CMC/train_CMC.py --data_folder /movie-associations \
    --model_path /movie-associations/saves/model-saves \
    --tb_path /movie-associations/saves/tb-saves \
    --resume /movie-associations/saves/model-saves/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_Lab/ckpt_epoch_71.pth \
    --view Lab
