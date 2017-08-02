import pickle
import os
import numpy as np
import argparse
import pdb
import time

from configs.base_config import BaseConfig
import hungarian

def main(args):

    cfg = BaseConfig()

    # load annotation file
    gt = pickle.load(open(os.path.join(cfg.annotation_root, args.test_id+'/annotations.pkl'),'r'))
    #temporary operation, join two labels
    gt = np.max(gt, axis=1)

    # load result file
    result = pickle.load(open(args.result_path, 'r'))
    result = result[args.test_id]

    # compute the confusion matrix
    C0 = genConMatrix(result, gt)

    # convert to minimum-cost assignment problem
    ma = np.max(C0)
    C2 = ma - C0
    pdb.set_trace()

    # run hungarian algorithm
    start = time.time()
    h = hungarian.HungarianSolver(C2.tolist())
    h.solve()
    end = time.time()
    print ("Hungarian Time: %d secs" % (end-start))

    ass = h.get_assignment()
    P = np.zeros((np.max(result)+1, np.max(gt)+1))
    for i in range(len(ass)):
        P[i, ass] = 1

    # normalization
    C2 /= np.sum(C2)

    C = C2 * P.T
    acc = np.trace(C)

    print "Acc: ", acc

    print "Confusion matrix after matching:"
    print C


def genConMatrix(result, gt):
    """
    Generate the confusion matrix of two segmentations

    Input
        gt   -   ground truth segmentation, vector with size N
        result - cluster result segmentation, vector with size N

    Output
        C    -   the confusion matrix (class by class), size k1 x k2
    """

    s1, G1 = convert_seg(result)
    s2, G2 = convert_seg(gt)

    print "Number of segments of result: ", s1
    print "Number of segments of gt: ", s2

    C = np.zeros((np.max(result)+1, np.max(gt)+1), dtype='int32')
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
    parser = argparse.ArgumentParser(description='Script for model training/testing')

    parser.add_argument('--test_id', type=str, help='testing session id')
    parser.add_argument('--result_path', help='path to the result file')
    parser.add_argument('--method', type=str, default='hungarian', help='evaulation method')

    args = parser.parse_args()

    main(args)
