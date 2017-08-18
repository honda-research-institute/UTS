import pickle
import os
import numpy as np
import argparse
import pdb
import time
import sys

from configs.evaluate_config import EvaluateConfig
from utils.utils import convert_seg, genConMatrix


def main():

    cfg = EvaluateConfig().parse()

    # load result file
    if cfg.result_path is None:
        result_path = os.path.join(cfg.result_root, cfg.name+'/result.pkl')
    else:
        result_path = cfg.result_path
    result_dict = pickle.load(open(result_path, 'r'))
    
    if cfg.save_result:
        new_result_dict = {}
        new_seg_dict = {}
    
    for test_id in result_dict.keys():
        if not cfg.silent_mode:
            print "Evaluating ", test_id

        # load annotation file
        gt_path = os.path.join(cfg.annotation_root, test_id+'/annotations.pkl')
        gt = pickle.load(open(gt_path ,'r'))
        #temporary operation, join two labels
        gt = np.max(gt, axis=1)
    
        result = result_dict[test_id]
    
        # compute the confusion matrix
        C0 = genConMatrix(gt, result)
    
        # run hungarian algorithm (maximum assignment problem
        sys.path.append(cfg.UTS_ROOT+'3rd-party/hungarian-algorithm')
        from hungarian import Hungarian
    
        h = Hungarian(C0.tolist(), is_profit_matrix=True)
        h.calculate()
    
        ass = h.get_results()
        P = np.zeros((np.max(gt)+1, np.max(result)+1), dtype='int')
        for i in range(len(ass)):
            P[ass[i][0], ass[i][1]] = 1

        if not cfg.silent_mode:
            print P
    
        # transfer the confusion matrix according to matching result
        C = C0.dot( P.T )
    
        if cfg.save_result:
            # save transformed result
            label_map = {}
            for i in range(len(ass)):
                label_map[ass[i][1]] = ass[i][0]
    
            for i in range(len(result)):
                result[i] = label_map[result[i]]

            new_result_dict[test_id] = result
            s, G = convert_seg(result)
            new_seg_dict[test_id] = {}
            new_seg_dict[test_id]['s'] = s
            new_seg_dict[test_id]['G'] = G
    
    
        # Evaluation
        f1 = []
        for i in range(C.shape[0]):
            c = C[i]
            if c[i] == 0:
                f1.append(0)
            else:
                precision = float(c[i]) / np.sum(C[:,i])
                recall = float(c[i]) / np.sum(c)
                f1.append(2*precision*recall / (precision+recall))
    
        if not cfg.silent_mode:
            print "Avg F1-score: %f" % np.mean(np.vstack(f1))
            print "F1-score:"
            print f1

    if cfg.save_result:
        pickle.dump(new_result_dict, open(result_path.replace('.pkl', '_hungarian.pkl'),'w'))
        pickle.dump(new_seg_dict, open(result_path.replace('.pkl', '_seg_hungarian.pkl'),'w'))
    
    #    # calculate accuracy
    #    C1 = C.astype(float)
    #    C1 /= np.sum(C1)
    #    acc = np.trace(C1)
    #    print "Acc: ", acc
    




if __name__ == '__main__':
    main()
