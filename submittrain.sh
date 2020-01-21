#!/bin/bash
#
#SBATCH --cpus-per-task = 32  

python /home/ubuntu/CMC/train_CMC.py --batch_size 256 --num_workers 32 --data_folder /movie-associations --model_path /movie-associations/saves/model-saves --tb_path /movie-associations/saves/tb-saves

aws s3 sync /movie-associations/saves s3://movie-associations/parallelcluster/saves/