#!/bin/bash

cd ..

# declare an array variable
m=30
test_session="201704150933"

prefix="kts${m}"
echo $prefix

declare -a name=(${prefix}"_featfc_camera" ${prefix}"_pred_cross1")
declare -a X_feat=("feat_fc" "pred_cross1")
declare -a modality_X=("camera" "camera")
#declare -a name=(${prefix}"_pred_cross")
#declare -a X_feat=("pred_cross")
#declare -a modality_X=("camera")

model=kts


# get length of an array
len=${#name[@]}

for (( i=0; i<${len}; i++ ))
do
    python run_clustering.py --name ${name[$i]} --model $model \
        --test_session $test_session --X_feat ${X_feat[$i]} \
        --modality_X ${modality_X[$i]} --m $m --K 18 --D 200 \
        --isTrain &    # run in parrallel
#        --isTrain --is_clustered &    # run in parrallel
done 

wait
