#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J temporal_training
#SBATCH --output=/movie-associations/logs/temporal-training-%j.out
#SBATCH --error=/movie-associations/logs/temporal-training-%j.err


python /home/ubuntu/CMC/train_CMC.py --data_folder /movie-associations/ \
 --model_path /movie-associations/saves/model-saves/temporal \
 --tb_path /movie-associations/saves/tb-saves/temporal \
 --view temporal \
 --time_lag 60 \
 --epochs 1 \