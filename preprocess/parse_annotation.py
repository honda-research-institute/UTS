import os
import glob
import pympi
import pickle
import h5py
import numpy as np
import pdb

class AnnotationReader():
    def __init__(self, root):

        self.root = root
        self.label_dict = {"background": 0}    # manually add a default background label

    def parse_all_annotation(self):

        with open(os.path.join(self.root, 'session_list'), 'r') as fin:
            for session in fin:
                session_id = session[:16].replace('-', '')
                annotation_path = os.path.join(self.root,
                        session[:10].replace('-', '_')+'_ITS_data_collection/' +
                        session_id+'_ITS/annotation/event/')
                annotation_files = glob.glob(annotation_path+"*.eaf")
                if len(annotation_files) > 0:
                    print session
                    d = h5py.File(os.path.join(self.root, 'camera/'+session_id+'/feats.h5'),'r')
                    label = self.parse_annotation(annotation_files[0], d['feats'][:].shape[0])

                    if not os.path.exists(os.path.join(self.root, 'annotation/'+session_id)):
                        os.mkdir(os.path.join(self.root, 'annotation/'+session_id))
                    pickle.dump(label, open(os.path.join(self.root, 'annotation/'+session_id+'/annotations.pkl'), 'w'))
                else:
                    print("Warning! Missing annotation for the session: " + annotation_path)

            print ("Save label dictionary")
            pickle.dump(self.label_dict, open(os.path.join(self.root, 'annotation/label_dict.pkl'), 'w'))

    def parse_annotation(self, file_path, N):
        """
        extract unoverlapped events, we are intereseted in the layers:
        u'\u88ab\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Stimuli-driven'
        u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented'

        N - number of frames of the video
        videos are down-sampled to 3fps
        """
        layers = [u'\u88ab\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Stimuli-driven',
                u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented']

        label = np.zeros((N, len(layers)), dtype='int32')
        eafob = pympi.Elan.Eaf(file_path)
        for i, layer in enumerate(layers):
            for annotation in eafob.get_annotation_data_for_tier(layer):
                name = annotation[2].strip()
                # manually fix some bug in annotation
                if name ==u'\u7a7f\u904eT\u5b57\u8def\u53e3 intersection passing':
                    name = u'\u7a7f\u904e\u5341\u5b57\u8def\u53e3 intersection passing'
                if name == '':
                    continue

                if not name in self.label_dict:
                    self.label_dict[name] = len(self.label_dict.keys())
                
                start = int(annotation[0] / 1000 * 3)
                end = int(annotation[1] / 1000 * 3)

                if start<0 or end>N:
                    print ("Length error!")
                    raise


                label[start-1: end, i] = self.label_dict[name]

        return label
