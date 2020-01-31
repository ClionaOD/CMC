#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J val_movie-trained
#SBATCH --output=/movie-associations/logs/val-movie-trained-%j.out
#SBATCH --error=/movie-associations/logs/val-movie-trained-%j.err


python /home/ubuntu/CMC/LinearProbing.py --data_folder /movie-associations/imagenet \
 --save_path /movie-associations/saves/val-saves/movie-trained \
 --tb_path /movie-associations/saves/tb-saves/val/movie-trained \
 --model_path /movie-associations/saves/model-saves/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_Lab/ckpt_epoch_74.pth
