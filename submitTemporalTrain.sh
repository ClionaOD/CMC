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
 --resume /movie-associations/saves/model-saves/temporal/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_view_temporal/ckpt_epoch_9.pth