#!/bin/bash

cd ../sources

n_hidden=2000
n_C=20
max_time=30
n_output=13
prob=0.9
beta_l2=0.01
optimizer=rmsprop

modality_X=camera
modality_Y=label
X_feat=feat_conv
Y_feat=label
model_type=convuntrimmedlstm
data_mode=pool

n_epochs=2    # 30 for untrimmed, 40 for Trimmed version
batch_size=64

gpu=$1
n_threads=4
buffer_size=10000

name=${model_type}_prediction_${Y_feat}_h${n_hidden}_b${beta_l2}

#python convlstm_unsupervised.py --name $name --gpu $gpu --data_mode $data_mode --snapshot_num 28 --continue_train --val_session train_session.txt \
#python convlstm_unsupervised.py --name $name --gpu $gpu --data_mode $data_mode --snapshot_num 28 --val_session val_session.txt \
python train_seq_prediction.py --name $name --isTrain --gpu $gpu --data_mode $data_mode \
    --n_epochs $n_epochs  --max_time $max_time --batch_size $batch_size --n_C $n_C \
    --n_output $n_output --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat \
    --modality_Y $modality_Y --Y_feat $Y_feat --n_threads $n_threads --buffer_size $buffer_size \
    --input_keep_prob ${prob} --output_keep_prob $prob --optimizer ${optimizer} \
    --model_type ${model_type} --beta_l2 $beta_l2 --is_classify

echo $name | mail -s "Task complete" yangxitongbob@gmail.com
