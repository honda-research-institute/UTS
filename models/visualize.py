import pickle as pkl
import os
import numpy as np
import argparse
import glob
import pdb

import matplotlib.pyplot as plt

def main(args):

    root = "/home/xyang/project/data/"
    if args.can:
###############
        output_path = root + 'visualize/sensor_lstm/'
    else:
        output_path = root + 'visualize/camera/'


    # load session ids
    session_list = open(root+'session_list', 'r')
    session_ids = session_list.read().strip().split('\n')
    session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]

    # load data
    # each row is a video
    if args.train:
        ids = session_ids[:-10]
    else:
        ids = session_ids[-10:]
    data = open(args.input, 'r').read().strip().split('\n')
    data = [d.split(',') for d in data]

    ############### Visualization #####################

    if args.method is None or args.method == 'plot':

        print "Visualization method: plot ..."

        if args.session_id is not None:
            row = ids.index(args.session_id)
        else:
            row = np.random.randint(len(ids))
        print ("Session_id: %s" % ids[row])

        d = data[row]
        plt.scatter(range(len(d)), d)
        plt.title("Session_id: %s" % ids[row])
        plt.show()

    if args.method == 'frames':
        
        print "Visualization method: frames ..."

        if args.session_id is not None:
            row = ids.index(args.session_id)
        else:
            row = np.random.randint(len(ids))
        print ("Session_id: %s" % ids[row])

##################
        frames = glob.glob(root+'camera/'+ids[row]+'/*.jpg')
#        frames = glob.glob(root+'visualize/sensor/'+ids[row]+'/2/*.jpg')
        frames.sort()
        label = data[row]

        for i in range(len(label)):
            outdir = output_path + ids[row] + '/' + label[i]
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            command = ["cp", frames[i], outdir]
            os.system(' '.join(command))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a model')

    parser.add_argument('input', help='input filename')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', dest='train', action='store_true',
                       help='Training data')
    group.add_argument('--test', dest='train', action='store_false',
                       help='Testing data')
    
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--can', dest='can', action='store_true',
                       help='CANbus input')
    group2.add_argument('--camera', dest='can', action='store_false',
                       help='camera input')

    parser.add_argument('--session_id', default=None,
                        help='session_id')

    parser.add_argument('--method', default=None,
            help='Visualization method: plot / frames')

    args = parser.parse_args()

    main(args)
