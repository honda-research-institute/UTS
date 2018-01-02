#!/bin/bash

cd ../sources

n_hidden=2000
n_C=20
max_time=30
n_output=13
prob=0.9
beta_l2=0
optimizer=adam
lr=5e-4

modality_X=camera
modality_Y=label
X_feat=feat_conv
Y_feat=label
model_type=convuntrimmedlstm

n_epochs=100    # 30 for untrimmed, 40 for Trimmed version
batch_size=64

gpu=$1
n_threads=4
buffer_size=10000

#name=${model_type}_prediction_${Y_feat}_h${n_hidden}_b${beta_l2}
#name=debug_${model_type}_focalloss_untrimmed_b${beta_l2}
name=debug_${model_type}_focalloss_untrimmed_b${beta_l2}_olddata

#python train_untrimmedseq_prediction.py --name $name --gpu $gpu --val_session test_session.txt --snapshot_num 6 \
python train_untrimmedseq_prediction.py --name $name --isTrain --gpu $gpu --val_session test_session.txt \
    --n_epochs $n_epochs  --max_time $max_time --batch_size $batch_size --n_C $n_C \
    --n_output $n_output --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat \
    --modality_Y $modality_Y --Y_feat $Y_feat --n_threads $n_threads --buffer_size $buffer_size \
    --input_keep_prob ${prob} --output_keep_prob $prob --optimizer ${optimizer} --learning_rate $lr\
    --model_type ${model_type} --beta_l2 $beta_l2 --is_classify --focal_loss

#echo $name | mail -s "Task complete" yangxitongbob@gmail.com
