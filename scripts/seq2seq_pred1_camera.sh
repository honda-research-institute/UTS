#!/bin/bash

cd ..

name=pred1_camera
model=seq2seq_basic

# train model
python train_model.py --name $name --model $model --train_session all \
    --batch_size 32 --max_time 15 --n_pred 1 --n_hidden 200 --iter_mode pred \
    --modality_X camera --modality_Y camera --n_input 1536 --n_output 1536 \
    --n_epochs 2 --isTrain

# extract feature
python train_model.py --name $name --model $model --iter_mode pred \
    --batch_size 32 --max_time 15 --n_pred 1 --n_hidden 200 --test_session all \
    --modality_X camera --modality_Y camera --n_input 1536 --n_output 1536 \
