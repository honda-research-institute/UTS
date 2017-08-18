#!/bin/bash

cd ..

# declare an array variable
m=350
test_session="201704151140"

prefix="kts${m}"
echo $prefix

declare -a name=(${prefix}"_featfc_camera" ${prefix}"_pred_cross")
declare -a X_feat=("feat_fc" "pred_cross")
declare -a modality_X=("camera" "camera")
#declare -a name=(${prefix}"_pred_cross")
#declare -a X_feat=("pred_cross")
#declare -a modality_X=("camera")

model=kts


# get length of an array
len=${#name[@]}

for (( i=0; i<${len}; i++ ))
do
    python run_evaluate.py --name ${name[$i]} --save_result &
done 

wait

