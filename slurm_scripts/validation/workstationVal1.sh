#!/bin/bash
#
#SBATCH --cpus-per-task=32  
#SBATCH -J val_ImgNet_moviepretrained
#SBATCH --output=/movie-associations/logs/val-imagenet-trained-%j.out
#SBATCH --error=/movie-associations/logs/val-imagenet-trained-%j.err

#Redo ImageNet val with fully pretrained on movie images.

#TODO: Add model_path after training finished.

CUDA_VISIBLE_DEVICES=0,1 python3 /home/clionaodoherty/CMC/LinearProbing.py \
    --data_folder /data/ILSVRC2012 \
    --save_path /home/clionaodoherty/movie-associations/saves/Lab/imgnet-val/fully_pretrained \
    --tb_path /home/clionaodoherty/CMC/tensorboard \
    --model_path /home/clionaodoherty/movie-associations/saves/Lab/movie-pretrain/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_200.pth