#!/bin/bash

cd ../

event=1
encoder=avgpool
#modality_X=camera
#X_feat=pred_convnn_prediction_label_b0.01-14
modality_X=can
X_feat=feat_norm

gpu=1

name=retrieval_${modality_X}_${X_feat}_${encoder}


python run_retrieval.py --name $name --gpu $gpu \
    --encoder $encoder --modality_X $modality_X --X_feat $X_feat --event $event
#    --test_session train_database.txt --val_session train_query.txt

python run_evaluate.py --name $name

