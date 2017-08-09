"""
    Align camera data and sensor data according to video time stamp
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import pdb
from scipy import signal
import h5py

sys.path.append("../configs")
from base_config import BaseConfig


class DataAlign(object):
    def __init__(self):
        self.cfg = BaseConfig().parse()
        self.session_template = "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"

    def get_start_end(self, session_id):
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
        video_data = pd.read_csv(video_filename)

        # for accel start point, set as roughly 10 seconds (3000 lines) before first movement
        accel_start_index = accel_data[accel_data["pedalangle(percent)"]>0].index[0]
        accel_start_index = max(0, accel_start_index - 3000)

        accel_start = accel_data.iloc[accel_start_index]['unix_timestamp']
        accel_end = accel_data.iloc[-1]['unix_timestamp']
        video_start = video_data.iloc[0]['unix_timestamp']
        video_end = video_data.iloc[-1]['unix_timestamp']

        start_time = max(video_start, accel_start)
        end_time = min(video_end, accel_end)

        return start_time, end_time

    def get_index(self, dataframe, time):
        """
        Get the closest index according to time
        """
        if (dataframe['unix_timestamp']-time).abs().min() > 1:    # differ too much
            raise ValueError("Differ too much!")

        return (dataframe['unix_timestamp']-time).abs().argmin()

    def get_accel(self, session_id, start_time, end_time):
        '''
        example:
        # timestamp,unix_timestamp,pedalangle(percent)
        20170410104105.117187500,1491846065.116573811,0.000000
        '''
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        filename = session_folder + "general/csv/accel_pedal_info.csv"
        dataframe = pd.read_csv(filename)
        start_index = self.get_index(dataframe, start_time)
        end_index = self.get_index(dataframe, end_time)

        return dataframe['pedalangle(percent)'][start_index:end_index+1].values

    def get_steer(self, session_id, start_time, end_time):
        '''
        example:
        # timestamp,unix_timestamp,steer_angle,steer_speed
        20170410104105.105468750,1491846065.104827166,-20.100000,0.000000
        '''
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        filename = session_folder + "general/csv/steer_info.csv"
        dataframe = pd.read_csv(filename)
        start_index = self.get_index(dataframe, start_time)
        end_index = self.get_index(dataframe, end_time)

        return dataframe[['steer_angle', 'steer_speed']][start_index:end_index+1].values

    def get_vel(self, session_id, start_time, end_time):
        '''
        example:
        # timestamp,unix_timestamp,speed
        20170410104105.117187500,1491846065.116569281,0.000000
        '''
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        filename = session_folder + "general/csv/vel_info.csv"
        dataframe = pd.read_csv(filename)
        start_index = self.get_index(dataframe, start_time)
        end_index = self.get_index(dataframe, end_time)

        return dataframe['speed'][start_index:end_index+1].values

    def get_brake(self, session_id, start_time, end_time):
        '''
        example:
        # timestamp,unix_timestamp,pedalpressure(kPa)
        20170410104105.109375000,1491846065.109828234,0.000000
        '''
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        filename = session_folder + "general/csv/brake_pedal_info.csv"
        dataframe = pd.read_csv(filename)
        start_index = self.get_index(dataframe, start_time)
        end_index = self.get_index(dataframe, end_time)

        return dataframe['pedalpressure(kPa)'][start_index:end_index+1].values

    def get_turn(self, session_id, start_time, end_time):
        '''
        example:
        # timestamp,unix_timestamp,lturn,rturn
        20170410104105.136718750,1491846065.135316610,0.000000,0.000000
        '''
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        filename = session_folder + "general/csv/turn_signal_info.csv"
        dataframe = pd.read_csv(filename)
        start_index = self.get_index(dataframe, start_time)
        end_index = self.get_index(dataframe, end_time)

        return dataframe[['lturn', 'rturn']][start_index:end_index+1].values

    def get_yaw(self, session_id, start_time, end_time):
        '''
        example:
        # timestamp,unix_timestamp,yaw(degree/s)
        20170410104105.109375000,1491846065.109578609,-0.244141
        '''
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        filename = session_folder + "general/csv/yaw_info.csv"
        dataframe = pd.read_csv(filename)
        start_index = self.get_index(dataframe, start_time)
        end_index = self.get_index(dataframe, end_time)

        return dataframe['yaw(degree/s)'][start_index:end_index+1].values

    def get_video(self, session_id, start_time, end_time):
        """
        Extract features for specific video frames
        """
        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        video_filename = session_folder + "camera/center/png_timestamp.csv"
        video_data = pd.read_csv(video_filename)

        # load actual frames
        frames_name = glob.glob(self.cfg.DATA_ROOT+'/frames/'+session_id+'/*jpg')
        frames_name = sorted(frames_name)

        # slice part of the video
        start_index = self.get_index(video_data, start_time)
        end_index = self.get_index(video_data, end_time)

        start_frame = int(os.path.basename(video_data['filename'][start_index]).strip('.png'))
        end_frame = int(os.path.basename(video_data['filename'][end_index]).strip('.png'))
        start_frame = start_frame / 10    # 3 fps
        end_frame = end_frame / 10    # 3 fps
        end_frame = min(end_frame+1, len(frames_name))

        feat_conv, feat_fc, prob = self.extract_feat(frames_name[start_frame:end_frame])
        feat_conv = feat_conv[:(end_frame-start_frame)]
        feat_fc = feat_fc[:(end_frame-start_frame)]
        prob = prob[:(end_frame-start_frame)]

        if not os.path.isdir(self.cfg.video_root+session_id):
            os.makedirs(self.cfg.video_root+session_id)
        with h5py.File(self.cfg.video_root+session_id+'/feat_conv.h5', 'w') as fout:
            fout.create_dataset("feats", data=feat_conv, dtype='float32')
        with h5py.File(self.cfg.video_root+session_id+'/feat_fc.h5', 'w') as fout:
            fout.create_dataset("feats", data=feat_fc, dtype='float32')
        with h5py.File(self.cfg.video_root+session_id+'/probs.h5', 'w') as fout:
            fout.create_dataset("feats", data=prob, dtype='float32')

        return feat_conv.shape[0]

    def extract_feat(self, frames_name):
        """
        Reference:
            1. Vasili's codes
            2. https://github.com/tensorflow/models/issues/429#issuecomment-277885861
        """

        slim_dir = "/home/xyang/anaconda2/lib/python2.7/site-packages/tensorflow/models/slim"
        checkpoints_dir = slim_dir + "/pretrained_models"
        checkpoints_file = checkpoints_dir + '/inception_resnet_v2_2016_08_30.ckpt'
        batch_size = 64

        sys.path.append(slim_dir)
        from PIL import Image
        from nets import inception
        import tensorflow as tf
        slim = tf.contrib.slim
        image_size = inception.inception_resnet_v2.default_image_size

        feat_conv = []
        feat_fc = []
        probs = []
        with tf.Graph().as_default():
            input_batch = tf.placeholder(dtype=tf.uint8,
                                         shape=(batch_size, 720, 1280, 3))
            resized_images = tf.image.resize_images(
                tf.image.convert_image_dtype(input_batch, dtype=tf.float32),
                [image_size, image_size]
                )
            preprocessed_images = tf.multiply(tf.subtract(resized_images, 0.5), 2.0)
            
            # Create the model, use the default arg scope to configure
            # the batch norm parameters.
            with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                logits, endpoints = inception.inception_resnet_v2(preprocessed_images,
                                                                  is_training=False)
            pre_pool = endpoints['Conv2d_7b_1x1']
            pre_logits_flatten = endpoints['PreLogitsFlatten']
            probabilities = endpoints['Predictions']

            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoints_file)

                for i in range(0, len(frames_name), batch_size):
                    current_batch = np.zeros((batch_size, 720, 1280, 3), dtype=np.uint8)
                    for j in range(batch_size):
                        if i+j == len(frames_name):
                            break
                        img = Image.open(frames_name[i+j]).convert('RGB')
                        current_batch[j] = np.array(img)

                    temp_conv, temp_fc, prob = sess.run([pre_pool,
                                                         pre_logits_flatten,
                                                         probabilities],
                                                         feed_dict={input_batch:
                                                             current_batch})
                    feat_conv.append(temp_conv)
                    feat_fc.append(temp_fc)
                    probs.append(prob)

        return np.concatenate(feat_conv, axis=0), np.concatenate(feat_fc, axis=0), np.concatenate(probs, axis=0)



    def align_data(self):

        for session_id in self.cfg.train_session:
            print "Aligning session: " + session_id
            start_time, end_time = self.get_start_end(session_id)

            # extract video features
            frame_num = self.get_video(session_id, start_time, end_time)

            # get sensor data
            sensor_data = []
            sensor_data.append(self.get_accel(session_id, start_time, end_time))
            sensor_data.append(self.get_steer(session_id, start_time, end_time))
            sensor_data.append(self.get_vel(session_id, start_time, end_time))
            sensor_data.append(self.get_brake(session_id, start_time, end_time))
            sensor_data.append(self.get_turn(session_id, start_time, end_time))
            sensor_data.append(self.get_yaw(session_id, start_time, end_time))

            # preprocessing of sensor data
            for i in range(len(sensor_data)):
                data = sensor_data[i]

                # smoothing (window size 101 is roughly 1/3 secs)
                data = signal.medfilt(data.reshape(data.shape[0],-1), [101, 1])

                # normalization
                mu = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                data = (data - mu) / std

                # resampling, consistent number with video frames
                samples = np.linspace(0, data.shape[0], endpoint=False, num=frame_num, dtype=int)
                data = data[samples]

                sensor_data[i] = data

            #pdb.set_trace()
            sensor = np.concatenate(sensor_data, axis=1)
            assert(sensor.shape[0] == frame_num)

            if not os.path.isdir(self.cfg.sensor_root+session_id):
                os.makedirs(self.cfg.sensor_root+session_id)
            with h5py.File(self.cfg.sensor_root+session_id+'/feat.h5', 'w') as fout:
                fout.create_dataset("feats", data=sensor, dtype='float32')

def main():

    data_aligner = DataAlign()
    data_aligner.align_data()

if __name__ == "__main__":
    main()
