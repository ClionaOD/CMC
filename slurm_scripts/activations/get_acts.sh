#!/bin/bash
#

python3 /home/clionaodoherty/CMC/anna_get_acts.py \
    --model_path /data/movie-associations/movie_Lab.pth \
    --save_path /home/clionaodoherty/CMC/activations/movie_lab.pickle \
    --transform Lab
