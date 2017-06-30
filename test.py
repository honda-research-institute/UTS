import numpy as np
import pdb

def iterate_minibatch(x, vid, batch_size, n_steps, shuffle=False):
    """
    Iterator for creating batch data
    x.shape = [N, dim]

    return shape [batch_size, n_step, dim]
    """
    
    valid = vid[n_steps-1:] == vid[:-(n_steps-1)]
    indices = np.where(valid)[0]

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, indices.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i: i+batch_size]

        temp = []
        for j in range(n_steps):
            temp.append(np.expand_dims(x[excerpt + j, :], axis=1))    # add axis for n_step

        yield np.concatenate(temp, axis=1)


data = [np.ones((1,3)) * i for i in range(10)]
data = np.vstack(data)
vid = np.asarray([1,1,1,2,2,2,2,3,3,3])
n_steps = 2
batch_size = 3

for x_batch in iterate_minibatch(data, vid, batch_size, n_steps, shuffle=True):
    print "one batch:"

    print x_batch
