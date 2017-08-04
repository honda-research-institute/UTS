import numpy as np

def iterate_minibatch(x, batch_size=0, n_steps=0, shuffle=False):
    """
    Iterator for creating batch data from a sequence (same video)
    x.shape = [N, dim]

    return shape:
    x_batch.shape = [batch_size, n_step, dim]
    seq_batch.shape = (batch_size,)
    """

    N, dim = x.shape

    # for convenient implementation of the boundary case
    x = np.vstack((x, np.zeros((n_steps-1,dim),dtype=x.dtype)))

    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, N, batch_size):
        length = min(i+batch_size, N) - i
        excerpt = indices[i: i+length]

        temp_batch = np.zeros((batch_size, n_steps, dim), dtype=x.dtype)
        seq_batch = np.zeros((batch_size, n_steps), dtype='int32')
        for j in range(n_steps):
            # "clever" way to get length
            valid = (excerpt+j) < N
            valid_pad = np.zeros((batch_size,),dtype=bool)
            valid_pad[:length] = valid
            seq_batch[valid_pad, j] = 1

            temp_batch[:length, j, :] = x[excerpt+j, :]

        seq_batch = np.sum(seq_batch, axis=1)

        yield temp_batch, seq_batch

def recon_minibatch(x, vid=None, y=None, batch_size=0, n_steps=0, shuffle=False):
    """
    Iterator for creating batch data for sequence to sequence reconstruction 
    (fixed sequence length)

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

    seq_len = n_step * np.ones((batch_size,), dtype='int32')

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, indices.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i: i+batch_size]

        temp_x = []
        temp_y = []
        for j in range(n_steps):
            temp_x.append(np.expand_dims(x[excerpt + j, :], axis=1))    # add axis for n_step
            temp_y.append(np.expand_dims(y[excerpt + j, :], axis=1))    # add axis for n_step


        yield np.concatenate(temp_x, axis=1), np.concatenate(temp_y, axis=1), seq_len

def fixlength_minibatch(x, y=None, vid=None, batch_size=8, n_steps=0, n_predict=0, shuffle=False):
    """
    Iterator for creating fix-length batch data for sequence to sequence tasks
    x.shape = [N, dim]

    return shape:
    x_batch.shape = [batch_size, n_step, dim]
    y_batch.shape = [batch_size, n_predict, dim]
    """

    if y is None:
        y = x
    if vid is None:
        vid = np.zeros(X.shape[0])
    if n_steps == 0:
        raise ValueError("Sequence length cannot be 0!")
    if n_predict == 0:
        n_predict = n_steps
    
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
            from models.kmeans import KMeansModel
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
