#!/bin/bash

cd ..

name=recon_camera
model=seq2seq_basic

#python train_model.py --name $name --model $model \
#    --batch_size 32 --max_time 15 --n_predict 15 --n_hidden 500 \
#    --train_session all --test_session all \
#    --modality_X camera --modality_Y camera --n_input 1536 --n_output 1536 \
#    --X_feat feat_fc --Y_feat feat_fc --iter_mode recon \
#    --n_epochs 5 --isTrain --gpu 1

# extract feature
python train_model.py --name $name --model $model \
    --batch_size 32 --max_time 15 --n_predict 15 --n_hidden 500 \
    --train_session all --test_session all \
    --modality_X camera --modality_Y camera --n_input 1536 --n_output 1536 \
    --X_feat feat_fc --Y_feat feat_fc --gpu 1\
