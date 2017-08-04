#!/bin/bash

cd ..

name=recon_camera
model=seq2seq_recon

python train_model.py --name $name --model $model --train_session all \
    --batch_size 32 --max_time 15 --n_hidden 200 --test_session 201704151140 \
    --modality_X camera --modality_Y camera --n_input 1536 --n_output 1536 \
    --n_epochs 1
