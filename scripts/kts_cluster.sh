#!/bin/bash

cd ..

name=kts_recon_camera
model=kts
feat_name=recon_camera

python run_clustering.py --name $name --model $model \
    --test_session 201704151140 --feat_name $feat_name \
    --modality_X camera --K 18 --D 1000 --m 200 \
    --isTrain

#python run_evaluate.py --name $name --save_result
