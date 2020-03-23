#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J train_CMC_1
#SBATCH --output=/movie-associations/logs/slurm-%j.out
#SBATCH --error=/movie-associations/logs/slurm-%j.err

#This will submit the pretraining on movie dataset with Lab objective,
#   resuming from epoch 71 in an attempt to get better acc performance comparable
#   to that of CMC on imagenet. In my first attempt, it had not fully converged
#   and the lr had not decayed. Previous Top-1 was 32.38 and Top-5 54.28.

python3 /home/clionaodoherty/CMC/train_CMC.py \
    --data_folder /data/movie-associations \
    --model_path /home/clionaodoherty/movie-associations/saves/Lab/movie-pretrain \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --resume /home/clionaodoherty/movie-associations/saves/Lab/movie-pretrain/ckpt_epoch_74.pth \
    --view Lab \
    --epochs 200 \
    --lr_decay_epochs '80, 120, 160' 