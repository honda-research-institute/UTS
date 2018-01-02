import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb
from sklearn.metrics import average_precision_score

from configs.retrieval_config import RetrievalConfig

def main():
    cfg = RetrievalConfig().parse()

    fin = pkl.load(open(os.path.join(cfg.result_root, 'result_'+cfg.name+'.pkl'), 'r'))
    result = fin['result']
    label_target = fin['label_target']
    num2label = pkl.load(open(os.path.join(cfg.annotation_root, 'num2label.pkl'), 'r'))
    event = result[0][1]

    # database statistics
    print ("%d target videos in database" % label_target.shape[0])
    for key in num2label[event]:
        print ("%d %s: count=%d, ratio=%f" % (key, num2label[event][key], 
            np.sum(label_target==key), np.mean(label_target==key)))

    mAP = {}
    mean_ap = 0
    count = 0
    for i in range(len(result)):
        dist = result[i][4]
        g = result[i][3]

        score = np.max(dist) - dist
        if np.sum(label_target==g) == 0:    # actually should not happen because they are not saved
            print "Warning ap =0!"
            ap = 0
        else:
            ap = average_precision_score(np.squeeze(label_target==g), np.squeeze(score))
        mean_ap += ap
        count += 1

        if g in mAP:
            mAP[g]['ap'] += ap
            mAP[g]['count'] += 1
        else:
            mAP[g] = {'ap':ap, 'count':1}

    print ("Mean AP = %f" % (mean_ap / count))
    new_mAP = 0
    for key in mAP:
        label = num2label[event][key]
        ap = mAP[key]['ap'] / mAP[key]['count']
        new_mAP += ap

        print ("%d %s: mAP = %f, query count=%d" % (key, label, ap, mAP[key]['count']))
        print ("Random mAP = %f" % average_precision_score(np.squeeze(label_target==key), np.squeeze(np.random.rand(label_target.shape[0]))))
    print ("Non-weighted mAP: ", new_mAP/len(mAP.keys()))

    ######## Precision @ top k #########
    top = 10

    mAP = {}
    mean_ap = 0
    count = 0
    for i in range(len(result)):
        dist = result[i][4]
        idx = np.argsort(dist)
        dist = dist[idx[:top]]
        l = label_target[idx[:top]]

        score = np.max(dist) - dist
        g = result[i][3]

        if np.sum(l==g) == 0:
            ap = 0
        else:
            ap = np.sum(np.squeeze(l==g)) / float(top)
        mean_ap += ap
        count += 1

        if g in mAP:
            mAP[g]['ap'] += ap
            mAP[g]['count'] += 1
        else:
            mAP[g] = {'ap':ap, 'count':1}

    print ("Mean Precision @ %d = %f" % (top, (mean_ap / count)))
    for key in mAP:
        label = num2label[event][key]
        ap = mAP[key]['ap'] / mAP[key]['count']

        print ("%d %s: P@%d = %f, query count=%d" % (key, label, top, ap, mAP[key]['count']))

if __name__ == '__main__':
    main()
