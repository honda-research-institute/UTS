#python clustering.py --camera --train --PCA 128 --output_name camera_kmeans_model.pkl

#python clustering.py --camera --test --PCA 128 --input_name camera_kmeans_model.pkl --output_name camera

data_path='/home/xyang/UTS/Data/general_sensors'
list='/home/xyang/UTS/Data/session_list'

while read line
do
    id=${line//-}
    id=${id:0:12}

    echo $id
    mkdir ${data_path}/${id}
    cp ${data_path}/general_sensors_${id}.pkl ${data_path}/${id}/feats.pkl
#    python visualize.py can_lstm_train.csv --train --can --method frames --session_id $id
done < $list
