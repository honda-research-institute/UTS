import numpy as np
from sklearn.metrics import average_precision_score

class Retriever(object):
    def __init__(self, method):
        self.method = method
        self.N = 0

    def load_data(self, feats, label):
        self.feats = feats
        self.label = label
        self.N = self.feats.shape[0]

    def retrieve(self, feat):

        if self.method == 'cos':
            dist, output = self.cos_retrieve(feat)
        else:
            raise NotImplementedError

        return dist, output

    def evaluate(self, dist, l):
        score = np.max(dist) - dist    # convert distance to score
        
        if np.sum(self.label==l) == 0:
            return 0
        else:
            return average_precision_score(np.squeeze(self.label==l), np.squeeze(score))

    def retrieve_and_evaluate(self, feat, l):
        dist, output = self.retrieve(feat)
        ap = self.evaluate(dist, l)
        return ap, dist, output

    def cos_retrieve(self, feat):
        sim = np.dot(self.feats, feat.T)
        dist = np.squeeze(np.max(sim) - sim)
        idx = np.argsort(dist)
        return dist, idx
