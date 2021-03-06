import os
import numpy as np
import time
import pdb
import pickle as pkl

from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.decomposition import PCA

class KMeansModel():

    def __init__(self, PCA=-1):
        self.PCA = PCA
        self.kmeans_model = None
        self.pca_model = None
        self.feasibility = False    # flag for whether model is trained

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

        self.feasibility = True
        print ("KMeans done! Clustering Time: %d secs" % (end-start))

    def save_model(self, output_path):

        # save the model
        pkl.dump({'kmeans': self.kmeans_model,
                'pca': self.pca_model},
                open(output_path+'kmeans_model.pkl', 'w'))


    def load_model(self, input_path):

        # load the model
        fin = pkl.load(open(input_path+'kmeans_model.pkl', 'r'))

        self.kmeans_model = fin['kmeans']
        self.pca_model = fin['pca']
        self.feasibility = True


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

    def train_and_predict(self, X, K):
        self.K = K

        if self.PCA > 0:
            print ("PCA to %d dims ..." % self.PCA)
            self.pca_model = PCA(n_components = self.PCA)
            X = self.pca_model.fit_transform(X)
            print ("Explained variance ratio: ",
                    np.sum(self.pca_model.explained_variance_ratio_))

        print ("K-means clustering: %d data samples with dim=%d clustered to %d clusters" % 
                (X.shape[0], X.shape[1], K))
    
        self.kmeans_model = KMeans(n_clusters=K, random_state=0).fit(X)

        return self.kmeans_model.predict(X)

