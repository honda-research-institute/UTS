import sys
import os
import numpy as np
import time
import pdb
import pickle as pkl

from sklearn.decomposition import PCA
sys.path.append('3rd-party/kts/')
from cpd_auto import cpd_auto
from cpd_nonlin import cpd_nonlin


class KTSModel():

    def __init__(self, PCA=-1, is_clustered=True, **kwargs):
        self.PCA = PCA
        self.pca_model = None
        self.is_clustered = is_clustered

        if self.is_clustered:
            self.aggregate_method = kwargs.get('aggregate_method', 'bow')
            self.cluster_method = kwargs.get('cluster_method', 'kmeans')
            self.K = kwargs.get('K', 10) 
            self.D = kwargs.get('D', 100)
            self.bow_model = kwargs.get('bow_model', None)
            self.cluster_model = kwargs.get('cluster_model', None)

    def save_model(self, output_path):

        # save the model
        pkl.dump({'kts': self.kmeans_model,
                'pca': self.pca_model},
                open(output_path+'kts_model.pkl', 'w'))

    def load_model(self, input_path):

        # load the model
        fin = pkl.load(open(input_path+'kmeans_model.pkl', 'r'))

        self.kts_model = fin['kts']
        self.pca_model = fin['pca']

#    def train(self, X, m=1000):
#        """
#        X - a list of data
#        """
#
#        self.m = m
#        
#        if self.PCA > 0:
#            X_array = np.vstack(X)
#            print ("PCA to %d dims ..." % self.PCA)
#            self.pca_model = PCA(n_components = self.PCA)
#            self.pca_model.fit(X_array)
#            print ("Explained variance ratio: ",
#                    np.sum(self.pca_model.explained_variance_ratio_))
#
#            for i in range(len(X)):
#                X[i] = self.pca_model.transform(X[i])
#
#        # KTS for each video
#        for i in range(X):
#            K = np.dot(X[i], X[i].T)
#            cps, _ cpd_auto(K, 2*m, 1)

    def train_and_predict(self, X, m=100):

        self.m = m
        
        if self.PCA > 0:
            print ("PCA to %d dims ..." % self.PCA)
            self.pca_model = PCA(n_components = self.PCA)
            X = self.pca_model.fit_transform(data)
            print ("Explained variance ratio: ",
                    np.sum(self.pca_model.explained_variance_ratio_))

        K = np.dot(X, X.T)
        #cps, _ = cpd_auto(K, 2*m, 1) 
        cps, _ = cpd_nonlin(K, m, 1) 

        if self.is_clustered:

            ################ Feature aggregation ################

            if self.aggregate_method.lower() == "bow":
                print "BoW Model..."

                if self.bow_model is not None:
                    self.bow_model = pkl.load(open(self.bow_model, 'r'))

                from utils.utils import BoWModel
                bow = BoWModel(self.D, self.bow_model)
                new_X = bow.fit_bow(X, cps)

            else:
                raise NotImplementedError


            ################ Clustering method ################

            label = None
            if self.cluster_method.lower() == 'kmeans':
                from .kmeans import KMeansModel

                self.kts_model = KMeansModel()
                label = self.kts_model.train_and_predict(new_X, self.K)
            else:
                raise NotImplementedError

            # transfer back to original label shape
            new_label = np.zeros(X.shape[0], dtype=int)
            cps = [0] + cps.tolist() + [X.shape[0]]

            for i in range(1, len(cps)):
                new_label[cps[i-1]:cps[i]] = label[i-1]

        else:
            # if not cluster it, then just assign distinct numbers
            new_label = np.zeros(X.shape[0], dtype=int)
            cps = [0] + cps.tolist() + [X.shape[0]]
            for i in range(1, len(cps)):
                new_label[cps[i-1]:cps[i]] = i - 1

        return cps, new_label


