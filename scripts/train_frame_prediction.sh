#!/bin/bash

cd ../sources

n_hidden=2000
n_C=20
n_output=13
prob=0.5
beta_l2=0
optimizer=adam
lr=5e-4

modality_X=camera
X_feat=feat_conv
#modality_Y=can
#Y_feat=feat_cluster20
modality_Y=label
Y_feat=label
model_type=convnn

n_epochs=100
batch_size=64

gpu=$1
n_threads=4
buffer_size=1000

#name=${model_type}_${Y_feat}_b${beta_l2}_focalloss
#name=prediction_nobinary_${model_type}_${Y_feat}_h${n_hidden}

#python train_frame_prediction.py --name $name --gpu $gpu --snapshot_num 14 --val_session train_session.txt \
python train_frame_prediction.py --name $name --isTrain --gpu $gpu --val_session test_session.txt \
    --n_epochs $n_epochs --batch_size $batch_size --n_C $n_C --learning_rate $lr \
    --n_output $n_output --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat \
    --modality_Y $modality_Y --Y_feat $Y_feat --n_threads $n_threads --buffer_size $buffer_size \
    --input_keep_prob ${prob} --output_keep_prob $prob --optimizer ${optimizer} \
    --model_type ${model_type}  --beta_l2 ${beta_l2} --is_classify --focal_loss #--trimmed 

#echo $name | mail -s "Task complete" yangxitongbob@gmail.com
