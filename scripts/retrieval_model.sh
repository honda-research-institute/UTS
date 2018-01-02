#!/bin/bash

cd ../

event=1
encoder=pred
#encoder=feat
model_name=$1

gpu=0

name=retrieval_${encoder}_${model_name}


python run_retrieval.py --name $name --gpu $gpu \
    --encoder $encoder --model_name $model_name --event $event

#name=retrieval_train_${encoder}_${model_name}
#python run_retrieval.py --name $name --gpu $gpu \
#    --encoder $encoder --model_name $model_name --event $event \
#    --test_session train_database.txt --val_session train_query.txt

python run_evaluate.py --name $name

#cd utils

#python convseq2seq_pred.py --name $model_name --gpu $gpu \
#    --event $event  --max_time $max_time --batch_size $batch_size \
#    --modality_Y $modality_Y --Y_feat $Y_feat \
#    --n_output $n_output --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat \
#    --optimizer ${optimizer}

