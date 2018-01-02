#!/bin/bash

cd ../sources

n_hidden=2000
n_C=20
max_time=30
n_predict=1
n_output=8
prob=0.7
margin=1
optimizer=adam

modality_X=camera
modality_Y=can
X_feat=feat_conv
Y_feat=feat_norm
data_mode=pool
K=50
cluster_name=kmeans_featnorm_${K}.pkl

n_epochs=10    # 30 for untrimmed, 40 for Trimmed version
batch_size=64

gpu=$1
n_threads=4
buffer_size=20

#name=convseq2seq_nocond_h${n_hidden}_${Y_feat}_${optimizer}
#python convseq2seq_encoder4.py --name $name --isTrain --gpu $gpu --sampling $sampling \

name=convlstm_rank_${Y_feat}_K${K}
#python convlstm_rank.py --name $name --isTrain --gpu $gpu --data_mode $data_mode \
#python convlstm_rank.py --name $name --gpu $gpu --data_mode $data_mode --val_session train_session.txt\
python convlstm_rank.py --name $name --gpu $gpu --data_mode $data_mode --snapshot_num 6 --continue_train --val_session train_session.txt \
    --margin $margin --cluster_name $cluster_name \
    --n_epochs $n_epochs  --max_time $max_time --batch_size $batch_size --n_C $n_C \
    --n_output $n_output --n_hidden $n_hidden --modality_X $modality_X --X_feat $X_feat \
    --modality_Y $modality_Y --Y_feat $Y_feat --n_threads $n_threads --buffer_size $buffer_size \
    --input_keep_prob ${prob} --output_keep_prob $prob --optimizer ${optimizer} 
#    --is_classify
