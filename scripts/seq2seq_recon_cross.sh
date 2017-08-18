#!/bin/bash

cd ..

name=recon_cross_adam
model=seq2seq_basic

python train_model.py --name $name --model $model \
    --batch_size 32 --max_time 15 --n_predict 15 --n_hidden 500 \
    --train_session all --test_session all \
    --modality_X camera --modality_Y can --n_input 1536 --n_output 8 \
    --X_feat feat_fc --Y_feat feat --iter_mode recon \
    --n_epochs 3 --isTrain --gpu 0 --learning_rate 1e-5 --optimizer adam

# extract feature
#python train_model.py --name $name --model $model \
#    --batch_size 32 --max_time 15 --n_predict 1 --n_hidden 500 \
#    --train_session all --test_session all \
#    --modality_X camera --modality_Y can --n_input 1536 --n_output 8 \
#    --X_feat feat_fc --Y_feat feat --gpu 0\
