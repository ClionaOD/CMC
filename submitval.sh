#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J train_CMC
#SBATCH --output=/movie-associations/logs/valid-%j.out
#SBATCH --error=/movie-associations/logs/valid-%j.err


python /home/ubuntu/CMC/LinearProbing.py --batch_size 20 \
 --data_folder /movie-associations \
 --save_path /movie-associations/saves/val-saves \
 --tb_path /movie-associations/saves/tb-saves/val \
 --model_path /movie-associations/saves/model-saves/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_Lab/ckpt_epoch_74.pth
