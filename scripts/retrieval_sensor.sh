#!/bin/bash

cd ../

event=1
encoder=pool
modality_X=can
X_feat=feat_quan

gpu=1

name=retrieval_${modality_X}_${X_feat}_${encoder}


python run_retrieval.py --name $name --gpu $gpu \
    --encoder $encoder --modality_X $modality_X --X_feat $X_feat --event $event
#    --test_session train_database.txt --val_session train_query.txt

python run_evaluate.py --name $name

#cd utils

#python convseq2seq_pred.py --name $model_name --gpu $gpu \
#    --event $event  --max_time $max_time --batch_size $batch_size \
#    --modality_Y $modality_Y --Y_feat $Y_feat \
#    --n_output $n_output --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat \
#    --optimizer ${optimizer}

