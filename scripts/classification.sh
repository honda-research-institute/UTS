
model=logistic
X_feat=feat
modality_X=can
PCA_dim=0
n_neighbors=5

name=lg_pred_cross1
name=${model}_${modality_X}_${X_feat}_PCA${PCA_dim}

cd ..

python run_classification.py --name $name --train_session train_session.txt --test_session all \
                             --X_feat $X_feat --modality_X $modality_X --isTrain --model $model \
                             --n_neighbors $n_neighbors --PCA_dim $PCA_dim
