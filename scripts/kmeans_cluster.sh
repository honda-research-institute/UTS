#!/bin/bash

cd ..


model=kmeans
feat_name='feat'
test_session="201704151140"
name=kmeans_${feat_name}
modality_X=can

python run_clustering.py --name $name --model $model --train_session $test_session \
    --test_session $test_session --X_feat $feat_name \
    --modality_X $modality_X --K 18 \
    --isTrain

python run_evaluate.py --name $name --save_result
