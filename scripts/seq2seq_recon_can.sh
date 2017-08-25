#!/bin/bash

cd ..

max_time=15
n_hidden=500

name=recon_can_mt${max_time}_nh${n_hidden}
model=seq2seq_basic

python train_model.py --name $name --model $model \
    --batch_size 32 --max_time $max_time --n_predict $max_time --n_hidden $n_hidden \
    --train_session all --test_session all \
    --modality_X can --modality_Y can --n_input 8 --n_output 8 \
    --X_feat feat --Y_feat feat --iter_mode recon \
    --n_epochs 30 --isTrain --gpu 1

# extract feature
#python train_model.py --name $name --model $model \
#    --batch_size 32 --max_time 15 --n_predict 15 --n_hidden 500 \
#    --train_session all --test_session all \
#    --modality_X camera --modality_Y camera --n_input 1536 --n_output 1536 \
#    --X_feat feat_fc --Y_feat feat_fc --gpu 1\
