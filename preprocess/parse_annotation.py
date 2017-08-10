import os
import glob
import pympi
import pickle
import h5py
import numpy as np
import pdb
import pandas as pd

import sys
sys.path.append("../configs")
from base_config import BaseConfig

class AnnotationReader():
    def __init__(self):

        self.cfg = BaseConfig().parse()
        self.label_dict = {"background": 0}    # manually add a default background label
        self.session_template = "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"

    def get_start_end_index(self, session_id):
        """
        Decide start and end point by the alignment of video and accelerate sensor data
        """

        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        accel_filename = session_folder + "general/csv/accel_pedal_info.csv"
        video_filename = session_folder + "camera/center/png_timestamp.csv"

        accel_data = pd.read_csv(accel_filename)
        video_data = self.read_video_csv(video_filename)

        # for start point, set as roughly 10 seconds (300 lines for video file) before first movement (accel > 0)
        accel_start_index = accel_data[accel_data["pedalangle(percent)"]>0].index[0]
        accel_start = accel_data.iloc[accel_start_index]['unix_timestamp']
        video_start_index = (video_data['unix_timestamp']-accel_start).abs().argmin()
        video_start_index = max(0, video_start_index - 300)

        video_start = video_data.iloc[video_start_index]['unix_timestamp']
        accel_start = accel_data.iloc[0]['unix_timestamp']

        video_end = video_data.iloc[-1]['unix_timestamp']
        accel_end = accel_data.iloc[-1]['unix_timestamp']

        start_time = max(video_start, accel_start)
        end_time = min(video_end, accel_end)

        start_index = self.get_index(video_data, start_time)
        end_index = self.get_index(video_data, end_time)

        return start_index, end_index

    def read_video_csv(self, video_filename):
        """  deal with weird files """

        video_data = pd.read_csv(video_filename)
        if not video_data['filename'][0] == 'camera/center/png/000000.png':
            video_data = pd.read_csv(video_filename, usecols=[0, 1, 3])
        if not video_data['filename'][0] == 'camera/center/png/000000.png':
            video_data = pd.read_csv(video_filename, usecols=[0, 1, 2])

        assert(video_data['# timestamp'].dtype == 'float')
        assert(video_data['unix_timestamp'].dtype == 'float')
        assert(video_data['filename'][0] == 'camera/center/png/000000.png')
        return video_data

    def get_index(self, dataframe, time):
        """
        Get the closest index according to time
        """
        if (dataframe['unix_timestamp']-time).abs().min() > 1:    # differ too much
            raise ValueError("Differ too much!")

        return (dataframe['unix_timestamp']-time).abs().argmin()

    def parse_all_annotation(self):

        for session_id in self.cfg.test_session:    # test sessions contain annotation
            print "Aligning session: " + session_id

            start_index, end_index = self.get_start_end_index(session_id)
            d = h5py.File(os.path.join(self.cfg.video_root, session_id+'/feat_fc.h5'),'r')
            d = d['feats'][:]
            label = self.parse_annotation(session_id, d.shape[0], start_index)

            if not os.path.isdir(self.cfg.annotation_root+session_id):
                os.makedirs(self.cfg.annotation_root+session_id)
            pickle.dump(label, open(self.cfg.annotation_root+session_id+'/annotations.pkl', 'w'))

        print ("Save label dictionary")
        pickle.dump(self.label_dict, open(os.path.join(self.cfg.annotation_root, 'label2num.pkl'), 'w'))

        num2label = {}
        for key in self.label_dict:
            num2label[self.label_dict[key]] = key
        pickle.dump(num2label, open(os.path.join(self.cfg.annotation_root, 'num2label.pkl'), 'w'))


    def parse_annotation(self, session_id, N, start_index):
        """
        extract unoverlapped events, we are intereseted in the layers:
        u'\u88ab\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Stimuli-driven'
        u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented'

        N - number of frames of the video
        videos are down-sampled to 3fps

        Annotation is not so precise, +-3 seconds offset is possible
        """
        layers = [u'\u88ab\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Stimuli-driven',
                u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented']

        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        annotation_filename = glob.glob(session_folder + 'annotation/event/*eaf')[0]

        start_time = int(np.round(start_index / 30.))    # convert start index (line #) to secs (original video is 30fps)

        label = np.zeros((N, len(layers)), dtype='int32')
        eafob = pympi.Elan.Eaf(annotation_filename)
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
                
                start_offset = int(np.round(annotation[0] / 1000.)) - start_time    # offset in terms of secs
                start = start_offset * 3    # convert secs to frame number 

                end_offset = int(np.round(annotation[1] / 1000.)) - start_time    # offset in terms of secs
                end = end_offset * 3    # convert secs to frame number 

                if start>=0 and end<N:
                    label[start:end+1, i] = self.label_dict[name]
                elif start<N and end>0:    # partially overlapped
                    print "Partial adjustment: ", start, end, N
                    start = max(start, 0)
                    end = min(N-1, end)
                else:
                    print "Skip this: ", start, end, N
                    #raise ValueError("Length error!")

        return label

def main():

    annotation_reader = AnnotationReader()
    annotation_reader.parse_all_annotation()

if __name__ == "__main__":
    main()
