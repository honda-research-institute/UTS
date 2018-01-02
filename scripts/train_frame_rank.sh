#!/bin/bash

cd ../sources

n_hidden=2000
n_C=20
beta_l2=0.0
optimizer=adam
lr=1e-4
margin=10.0

modality_X=camera
X_feat=feat_conv
#modality_Y=can
#Y_feat=feat_cluster20
modality_Y=label
Y_feat=label
model_type=convnn_rank

n_epochs=50
batch_size=64

gpu=$1
n_threads=4
buffer_size=10000

name=${model_type}_${Y_feat}_b${beta_l2}

#python convlstm_unsupervised.py --name $name --gpu $gpu --data_mode $data_mode --snapshot_num 28 --continue_train --val_session train_session.txt \
#python convlstm_unsupervised.py --name $name --gpu $gpu --data_mode $data_mode --snapshot_num 28 --val_session val_session.txt \
#python train_frame_rank.py --name $name --gpu $gpu --snapshot_num 14 --val_session train_session.txt \
python train_frame_rank.py --name $name --isTrain --gpu $gpu \
    --n_epochs $n_epochs --batch_size $batch_size --n_C $n_C --learning_rate $lr\
    --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat --margin $margin \
    --modality_Y $modality_Y --Y_feat $Y_feat --n_threads $n_threads --buffer_size $buffer_size \
    --model_type ${model_type}  --beta_l2 ${beta_l2} --optimizer ${optimizer}

#echo $name | mail -s "Task complete" yangxitongbob@gmail.com
