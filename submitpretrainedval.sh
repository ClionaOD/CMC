#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J train_CMC
#SBATCH --output=/movie-associations/logs/valid-%j.out
#SBATCH --error=/movie-associations/logs/valid-%j.err


python /home/ubuntu/CMC/LinearProbing.py --batch_size 20 \
 --data_folder /movie-associations \
 --save_path /movie-associations/saves/val-saves/pretrained \
 --tb_path /movie-associations/saves/tb-saves/val/pretrained \
 --model_path /movie-associations/pretrained/CMC_alexnet.pth
