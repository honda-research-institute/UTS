#!/bin/bash

cd ..


name=kmeans_pred_cross
model=kmeans
feat_name=pred_cross
test_session="201704151140"

python run_clustering.py --name $name --model $model --train_session $test_session \
    --test_session $test_session --X_feat $feat_name \
    --modality_X camera --K 18 \
    --isTrain

python run_evaluate.py --name $name --save_result
