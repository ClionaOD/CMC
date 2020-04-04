#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J val_imagenet-trained
#SBATCH --output=/movie-associations/logs/val-imagenet-trained-%j.out
#SBATCH --error=/movie-associations/logs/val-imagenet-trained-%j.err


python /home/ubuntu/CMC/LinearProbing.py --data_folder /movie-associations/imagenet \
 --save_path /movie-associations/saves/val-saves/imagenet-trained \
 --tb_path /movie-associations/saves/tb-saves/val/imagenet-trained \
 --model_path /movie-associations/pretrained/CMC_alexnet.pth