import os
import sys
import glob
import h5py
import numpy as np

sys.path.append("../configs")
from base_config import BaseConfig

cfg = BaseConfig().parse()

for session_id in cfg.all_session:
    with h5py.File(os.path.join(cfg.sensor_root, session_id+'/feat.h5'), 'r') as fin:
        feat = fin['feats'][:]
    
    new_feat = np.zeros((feat.shape[0], 22), dtype='float32')

    new_feat[feat[:,0]<1, 0] = 1
    new_feat[np.all([feat[:,0]>=1,feat[:,0]<15], axis=0), 1] = 1
    new_feat[feat[:,0]>=15, 2] = 1

    new_feat[feat[:,1]<-5, 3] = 1
    new_feat[np.all([feat[:,1]>=-5,feat[:,1]<5], axis=0), 4] = 1
    new_feat[feat[:,1]>=5, 5] = 1

    new_feat[feat[:,2]<-15, 6] = 1
    new_feat[np.all([feat[:,2]>=-15,feat[:,2]<15], axis=0), 7] = 1
    new_feat[feat[:,2]>=15, 8] = 1

    new_feat[feat[:,3]<1, 9] = 1
    new_feat[np.all([feat[:,3]>=1,feat[:,3]<40], axis=0), 10] = 1
    new_feat[feat[:,3]>=40, 11] = 1

    new_feat[feat[:,4]<90, 12] = 1
    new_feat[np.all([feat[:,4]>=90,feat[:,4]<1200], axis=0), 13] = 1
    new_feat[feat[:,4]>=1200, 14] = 1

    new_feat[feat[:,5]==0, 15] = 1
    new_feat[feat[:,5]==1, 16] = 1

    new_feat[feat[:,6]==0, 17] = 1
    new_feat[feat[:,6]==1, 18] = 1

    new_feat[feat[:,7]<-0.5, 19] = 1
    new_feat[np.all([feat[:,7]>=-0.5,feat[:,7]<0.5], axis=0), 20] = 1
    new_feat[feat[:,7]>=0.5, 21] = 1

    with h5py.File(os.path.join(cfg.sensor_root, session_id+'/feat_quan.h5'), 'w') as fout:
        fout.create_dataset(name='feats', data=new_feat, dtype='float32')
