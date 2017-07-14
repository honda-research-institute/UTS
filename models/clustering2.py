import pickle as pkl
import h5py
import os
from os import path
import numpy as np
import argparse
import time
import pdb

from sklearn.cluster import KMeans
from scipy.stats import mode


def predict_data(args, fname, session_ids):

    output_path = args.root + 'visualize/sensor'
    
    # load model
    kmeans = pkl.load(open(args.input_name, 'r'))

    with open(fname, 'w') as fout:
        for session_id in session_ids:

            fin = h5py.File(path.join(output_path, "{0}/2/feats.h5").format(session_id), 'r')
            d = fin['feats'][:]
            
            if args.PCA_dim > 0:
                from sklearn.decomposition import PCA
                PCA_dim = args.PCA_dim
            
                pca = pkl.load(open(path.join(output_path, 'pca_{0}.pkl').format(PCA_dim), 'r'))
                d = pca.transform(d)

            labels = kmeans.predict(d)
    
            if args.smoothing:
                labels = smoothing(labels)
    
            np.savetxt(fout, np.reshape(labels, [1, -1]), fmt="%i", delimiter=",")

def smoothing(labels):

    # smooth the labels by majority voting within sliding window
    half_win = 7    # window size = 15

    new_labels = np.zeros(labels.shape)
    for i in range(len(labels)):
        start = i-half_win if i-half_win >= 0 else 0
        end = i + half_win if i+half_win < len(labels) else len(labels)

        window = labels[start:end]
        new_labels[i] = mode(window)[0][0]

    return new_labels


def main(args):

    root = args.root
    output_path = args.root + 'visualize/sensor'

    # load session ids
    session_list = open(root+'session_list', 'r')
    session_ids = session_list.read().strip().split('\n')
    session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]

    train_ids = session_ids[:-10]
    test_ids = session_ids[-10:]


    # training stage
    if args.train_stage:

        allsessions = []
        for session_id in train_ids:
            fin = h5py.File(path.join(output_path, "{0}/2/feats.h5").format(session_id), 'r')
            d = fin['feats'][:]
            allsessions.append(d)
    
        data = np.vstack(allsessions)
    
        if args.PCA_dim > 0:
            from sklearn.decomposition import PCA
            PCA_dim = args.PCA_dim
    
            print ("PCA to %d dims ..." % PCA_dim)
    
            # look for cached model
            if not os.path.exists(path.join(output_path, 'pca_{0}.pkl').format(PCA_dim)):
                pca = PCA(n_components = PCA_dim)
    
                start = time.time()
                data = pca.fit_transform(data)
                end = time.time()
                print ("PCA Time: %d secs" % (end-start))
    
                pkl.dump(pca, open(path.join(output_path, 'pca_{0}.pkl').format(PCA_dim), 'w'))
            else:
                print ("Using cached PCA model")
                pca = pkl.load(open(path.join(output_path, 'pca_{0}.pkl').format(PCA_dim), 'r'))
                data = pca.transform(data)
            print ("Explained variance ratio: ", np.sum(pca.explained_variance_ratio_))
    
        # cluster data
        K = args.K
        
        print ("K-means clustering: %d data samples with dim=%d clustered to %d clusters" % 
                (data.shape[0], data.shape[1], K))
    
        start = time.time()
        kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
        end = time.time()
        print ("Clustering Time: %d secs" % (end-start))
    
        # save the model
        pkl.dump(kmeans, open(args.output_name, 'w'))

    # testing stage
    if args.test_stage:
        predict_data(args, args.output_name+'_train.csv', train_ids)
#        predict_data(args, args.output_name+'_test.csv', test_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a model')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', dest='train_stage', action='store_true',
                       help='Training')
    group.add_argument('--test', dest='test_stage', action='store_true',
                       help='Testing')


    parser.add_argument('--root', dest='root', default="/home/xyang/project/data/", type=str,
                       help='Root path')
    parser.add_argument('--input_name', dest='input_name', default='kmeans_model.pkl',
                       help='Input model file name')
    parser.add_argument('--output_name', dest='output_name', default='kmeans_model.pkl',
                       help='Output model file name)')
    parser.add_argument('--K', dest='K', default=10, type=int,
                       help='Number of clusters')
    parser.add_argument('--smoothing', dest='smoothing', action='store_true',
                       help='Smooth the output labels')
    parser.add_argument('--PCA', dest='PCA_dim', default=0, type=int,
                       help='Whether to use PCA and the dimensions to keep')


    args = parser.parse_args()

    main(args)
