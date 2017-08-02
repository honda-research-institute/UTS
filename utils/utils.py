from models.kmeans import KMeansModel
import numpy as np

def iterate_minibatch1(x, vid=None, batch_size=0, n_steps=0, shuffle=False):
    """
    Iterator for creating batch data for sequence to scalar prediction 
    x.shape = [N, dim]

    return shape:
    x_batch.shape = [batch_size, n_step, dim]
    y_batch.shape = [batch_size, dim]
    """

    if vid is None:
        vid = np.zeros(X.shape[0])
    
    valid = vid[n_steps:] == vid[:-n_steps]
    indices = np.where(valid)[0]

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, indices.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i: i+batch_size]

        temp = []
        for j in range(n_steps):
            temp.append(np.expand_dims(x[excerpt + j, :], axis=1))    # add axis for n_step

        yield np.concatenate(temp, axis=1), x[excerpt + n_steps, :]

def iterate_minibatch2(x, vid=None, y=None, batch_size=0, n_steps=0, shuffle=False):
    """
    Iterator for creating batch data for sequence to sequence reconstruction
    (including cross sequence reconstruction)
    x.shape = [N, dim]

    return shape:
    x_batch.shape = [batch_size, n_step, dim]
    y_batch.shape = [batch_size, n_step, dim]
    """

    if vid is None:
        vid = np.zeros(X.shape[0])
    if y is None:
        y = x
    
    valid = vid[n_steps:] == vid[:-n_steps]
    indices = np.where(valid)[0]

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, indices.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i: i+batch_size]

        temp = []
        for j in range(n_steps):
            temp.append(np.expand_dims(y[excerpt + j, :], axis=1))    # add axis for n_step

        yield np.concatenate(temp, axis=1), np.concatenate(temp, axis=1)

def iterate_minibatch3(x, vid=None, y=None, batch_size=0, n_steps=0, n_predict=0, shuffle=False):
    """
    Iterator for creating batch data for sequence to sequence prediction
    (including cross sequence prediction)
    x.shape = [N, dim]

    return shape:
    x_batch.shape = [batch_size, n_step, dim]
    y_batch.shape = [batch_size, n_predict, dim]
    """

    if vid is None:
        vid = np.zeros(X.shape[0])
    
    valid = vid[n_steps+n_predict:] == vid[:-n_steps-n_predict]
    indices = np.where(valid)[0]

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, indices.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i: i+batch_size]

        temp = []
        for j in range(n_steps):
            temp.append(np.expand_dims(x[excerpt + j, :], axis=1))    # add axis for n_step

        temp2 = []
        for j in range(n_steps, n_steps+n_predict):
            temp2.append(np.expand_dims(y[excerpt + j, :], axis=1))    # add axis for n_predict
        if y is None:    # no cross prediction
            temp2 = temp

        yield np.concatenate(temp, axis=1), np.concatenate(temp2, axis=1)


class BoWModel():
    """
    Build Bag-of-Word representation
    X - data with N*dim
    D - BOW cluster dimensions
    cps - change points
    model_path - path to previously-obtained kmeans model
    """
    
    def __init__(self, D=100, model=None):

        self.D = D
        self.model = model

    def fit_bow(self, X, cps):
        
        if self.model is None:
            # Build dictionary for BoW from scratch using X
            model = KMeansModel()
            model.train(X, self.D)
        else:
            model = self.model

        label = model.predict(X)

        # Build BoW
        bow = np.zeros((cps.shape[0]+1, self.D))
        cps = [0] + cps.tolist() + [X.shape[0]]

        for i in range(1, len(cps)):
            for j in range(cps[i-1], cps[i]):
                bow[i-1, label[j]] += 1

        return bow
