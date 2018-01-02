#!/bin/bash

event=1
model_name=$1
snapshot_num=$2

gpu=1

cd ../utils

python convnn_pred.py --model_name $model_name --gpu $gpu --snapshot_num $snapshot_num
