import os
import numpy as np
import time
import pdb
import pickle as pkl

from sklearn.cluster import KMeans
from scipy.stats import mode

class KMeansModel():

    def __init__(self, PCA=-1):
        self.PCA = PCA
        self.kmeans_model = None
        self.pca_model = None

    def train(self, X, K):
        self.K = K

        if self.PCA > 0:
            print ("PCA to %d dims ..." % self.PCA)
            self.pca_model = PCA(n_components = self.PCA)
            X = self.pca_model.fit_transform(data)
            print ("Explained variance ratio: ",
                    np.sum(self.pca_model.explained_variance_ratio_))

        print ("K-means clustering: %d data samples with dim=%d clustered to %d clusters" % 
                (X.shape[0], X.shape[1], K))
    
        start = time.time()
        self.kmeans_model = KMeans(n_clusters=K, random_state=0).fit(X)
        end = time.time()
        print ("KMeans done! Clustering Time: %d secs" % (end-start))

    def save_model(self, output_path, suffix=''):

        # save the model
        pkl.dump(self.kmeans_model, open(os.path.join(output_path, 'kmeans_'+suffix+'.pkl'), 'w'))

        if self.PCA > 0:
            pkl.dump(self.pca_model, open(os.path.join(output_path, 'pca_'+suffix+'.pkl'), 'w'))

    def load_model(self, input_path, suffix=''):

        # load the model
        self.kmeans_model = pkl.load(open(os.path.join(input_path, 'kmeans_'+suffix+'.pkl'), 'r'))

        if self.PCA > 0:
            self.pca_model = pkl.load(open(os.path.join(input_path, 'pca_'+suffix+'.pkl'), 'r'))

    def predict(self, X):
        
        if self.kmeans_model is None:
            print "Error! You need to load a kmeans model!"
            raise

        if self.PCA > 0 and self.pca_model is None:
            print "Error! We cannot find the PCA model!"
            raise

        if self.PCA > 0:
            X = self.pca_model.transform(X)

        return self.kmeans_model.predict(X)


