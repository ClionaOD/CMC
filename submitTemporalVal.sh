#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J val_temporal-trained
#SBATCH --output=/movie-associations/logs/val-temporal-trained-%j.out
#SBATCH --error=/movie-associations/logs/val-temporal-trained-%j.err


python /home/ubuntu/CMC/LinearProbing.py --data_folder /movie-associations/imagenet \
 --save_path /movie-associations/saves/val-saves/temporal-trained \
 --tb_path /movie-associations/saves/tb-saves/val/temporal-trained \
 --model_path /movie-associations/saves/model-saves/temporal/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_view_temporal/ckpt_epoch_70.pth \
 --view temporal \
 --time_lag 60