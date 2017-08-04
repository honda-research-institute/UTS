#!/bin/bash

cd ..

name=kmeans_recon_camera
model=kmeans
feat_name=recon_camera

python run_clustering.py --name $name --model $model --train_session 201704151140 \
    --test_session 201704151140 --feat_name $feat_name \
    --modality_X camera --K 18 \
    --isTrain
