import pickle
import os
import numpy as np
import argparse
import pdb
import time
import sys

from configs.evaluate_config import EvaluateConfig


def main():

    cfg = EvaluateConfig().parse()

    test_id = cfg.test_session[0]

    # load annotation file
    gt_path = os.path.join(cfg.annotation_root, test_id+'/annotations.pkl')
    gt = pickle.load(open(gt_path ,'r'))
    #temporary operation, join two labels
    gt = np.max(gt, axis=1)

    # load result file
    result = pickle.load(open(cfg.result_path, 'r'))
    result = result[test_id]

    # compute the confusion matrix
    C0 = genConMatrix(gt, result)

    #C0 = C0.T

    # run hungarian algorithm (maximum assignment problem
    sys.path.append(cfg.UTS_ROOT+'3rd-party/hungarian-algorithm')
    from hungarian import Hungarian

    h = Hungarian(C0.tolist(), is_profit_matrix=True)
    h.calculate()

    ass = h.get_results()
    P = np.zeros((np.max(gt)+1, np.max(result)+1), dtype='int')
    for i in range(len(ass)):
        P[ass[i][0], ass[i][1]] = 1
    print P

    # transfer the confusion matrix according to matching result
    C = C0.dot( P.T )

    f1 = []
    for i in range(C.shape[0]):
        c = C[i]
        if c[i] == 0:
            f1.append(0)
        else:
            precision = float(c[i]) / np.sum(C[:,i])
            recall = float(c[i]) / np.sum(c)
            f1.append(2*precision*recall / (precision+recall))

    print "Avg F1-score: %f" % np.mean(np.vstack(f1))
    print "F1-score:"
    print f1

#    # calculate accuracy
#    C1 = C.astype(float)
#    C1 /= np.sum(C1)
#    acc = np.trace(C1)
#    print "Acc: ", acc

    print "Confusion matrix after matching:"
    print C


def genConMatrix(gt, result):
    """
    Generate the confusion matrix of two segmentations

    Input
        gt   -   ground truth segmentation, vector with size N
        result - cluster result segmentation, vector with size N

    Output
        C    -   the confusion matrix (class by class), size k1 x k2
    """

    s1, G1 = convert_seg(gt)
    s2, G2 = convert_seg(result)

    print "Number of segments of gt: ", len(s1)
    print "Number of segments of result: ", len(s2)

    C = np.zeros((np.max(gt)+1, np.max(result)+1), dtype='int32')
    for i in range(len(s1)-1):
        for j in range(len(s2)-1):
            a = max(s1[i], s2[j])
            b = min(s1[i+1], s2[j+1])

            if a < b:
                C[G1[i], G2[j]] += b - a

    return C


def convert_seg(seg, k=0):
    """
    Convert original segmentation vector

    Input
        seg    -   original segmentation vector with size N

    Output
        s  -  starting position of each segment, list with size m+1, m is the number of segment
        G  -  label of each segment, list with size m 
    """

    if k == 0:
        k = np.max(seg) + 1

    N = seg.shape[0]

    s = [0]
    G = [seg[0]]
    for i in range(1, N):
        if not seg[i] == seg[i-1]:
            s.append(i)
            G.append(seg[i])
    s.append(N)

    return s, G

if __name__ == '__main__':
    main()
