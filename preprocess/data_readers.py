'''
A collection of various data readers for:

camera  (front camera video)                                          : mp4/png
general (accel_pedal, steer, brake_pedal, rtk_track, turn_signal, yaw): csv
lidar   (ibeo, velodyne)                                              : bag
mobileye                                                              : csv
route                                                                 : kmz

Author: Vasili Ramanishka

TODO: Check if all fields in pandas frames are correctly referenced
(to be save it is better to reference by field name since index is
presented sometimes)
'''

import math
import time

import glob
import numpy as np
import os
import pandas as pd
import subprocess
from functools import partial
from os import path

'''
Index data should be converted to more convenient format. pandas dataframe?

Every record then would have simple fields with locations of corresponding
files, event classes etc. After we have this convenient data structure it is
easy to pull necessary chunks out of other data files.
'''


'''
DATA LAYOUT
201704101041_ITS/
   camera/
        center/
           png/
               ...
        2017-04-10-10-41-02.mp4
        png_timestamp.csv
   general/
        csv/
           accel_pedal_info.csv
           rtk_pos_info.csv
           steer_info.csv
           vel_info.csv
           brake_pedal_info.csv
           rtk_track_info.csv
           turn_signal_info.csv
           yaw_info.csv
   lidar/
        ibeo/
            ibeo.bag
        velodyne/
            velodyne.bag
   mobileye/
        csv/
            mobileye_info.csv
            objA3_info.csv
            objB2_info.csv
            objC1_info.csv
            objC4_info.csv
            objA1_info.csv
            objA4_info.csv
            objB3_info.csv
            objC2_info.csv
            objA2_info.csv
            objB1_info.csv
            objB4_info.csv
            objC3_info.csv
   preview.mp4
   route.kmz
'''


class GeneralSensorsReader(object):
    '''
    pedal pressure sensor and left/right turn signal sensor work
    at completely different frequency;
    To get correct timestamps for input we need to perform lookup in reference
    csv for png files;
    since csv is quite slow to process (no seek), we need to convert input csv
    also. Read the first and the last timestamps to initialize a tick size for
    the fast lookup: only make sure that all these sensors have more or less
    uniformly sampled observations
    '''
    def __init__(self, base_data_path, cache_path):
        self.root = base_data_path
        self.cache_path = cache_path + "general_sensors/"
        self.sensor_dict = {
            "accel":    self.get_accel_pedal_info,
            "steer":    self.get_steer_info,
            "vel":      self.get_vel_info,
            "brake":    self.get_brake_pedal_info,
            "turn":     self.get_turn_signal_info,
            "yaw":      self.get_yaw_info
            }
        self.session_template = "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"
        # cached dataset
        if path.exists(path.join(self.cache_path,  "general_sensors.npy")):
            self.data = np.load(path.join(self.cache_path,
                                          "general_sensors.npy"))
        else:
            self.data = None
    '''
    this list of functions can be easily substituted by only one function
    '''
    def resample(self, end):
        return np.linspace(0, end, endpoint=False,
                           num=int(math.ceil(self.freq*(self.end-self.start))),
                           dtype=int)

    def get_accel_pedal_info(self):
        '''
        example:
        # timestamp,unix_timestamp,pedalangle(percent)
        20170410104105.117187500,1491846065.116573811,0.000000
        '''
        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "general/csv/accel_pedal_info.csv"

        data = pd.read_csv(csv_filename)
        result = data[(data["unix_timestamp"] >= self.start) &
                      (data["unix_timestamp"] <= self.end)]["pedalangle(percent)"]
        if result.empty:
            try:
                column = "pedalangle(percent)"
                result = data[(data["unix_timestamp"] >= self.start)].iloc[0][column]
            except IndexError:
                # invalid time reference
                # import ipdb; ipdb.set_trace()
                return [None]

        # df = df[df['unix_timestamp'].between(99, 101, inclusive=True)]
        # rescale/downsample
        samples = self.resample(result.shape[0])
        if (len(samples) == 1) and (result.shape[0] == 1):  # workaround against invalid data
            result = result.values
        else:
            result = result.iloc[samples].values
        assert len(result) != 0
        return result

    def get_steer_info(self):
        '''
        example:
        # timestamp,unix_timestamp,steer_angle,steer_speed
        20170410104105.105468750,1491846065.104827166,-20.100000,0.000000
        '''
        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "general/csv/steer_info.csv"
        data = pd.read_csv(csv_filename)
        result = data[(data["unix_timestamp"] >= self.start) &
                      (data["unix_timestamp"] <= self.end)][["steer_angle", "steer_speed"]]
        if result.empty:
            try:
                columns = ["steer_angle", "steer_speed"]
                result = data[(data["unix_timestamp"] >= self.start)].iloc[0][columns]
            except IndexError:
                return [[None, None]]

        # df = df[df['unix_timestamp'].between(99, 101, inclusive=True)]
        # rescale/downsample
        samples = self.resample(result.shape[0])
        if (len(samples) == 1) and (result.shape == (2,)):  # workaround against invalid data
            result = [result.values]
        else:
            result = result.iloc[samples].values
        assert len(result) != 0
        return result

    def get_vel_info(self):
        '''
        example:
        # timestamp,unix_timestamp,speed
        20170410104105.117187500,1491846065.116569281,0.000000
        '''
        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "general/csv/vel_info.csv"
        data = pd.read_csv(csv_filename)
        result = data[(data["unix_timestamp"] >= self.start) &
                      (data["unix_timestamp"] <= self.end)]["speed"]
        if result.empty:
            try:
                column = "speed"
                result = data[(data["unix_timestamp"] >= self.start)].iloc[0][column]
            except IndexError:
                return [None]

        # df = df[df['unix_timestamp'].between(99, 101, inclusive=True)]
        # rescale/downsample
        samples = self.resample(result.shape[0])
        if (len(samples) == 1) and (result.shape[0] == 1):  # workaround against invalid data
            result = result.values
        else:
            result = result.iloc[samples].values
        assert len(result) != 0
        return result

    def get_brake_pedal_info(self):
        '''
        example:
        # timestamp,unix_timestamp,pedalpressure(kPa)
        20170410104105.109375000,1491846065.109828234,0.000000
        '''

        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "general/csv/brake_pedal_info.csv"
        data = pd.read_csv(csv_filename)
        result = data[(data["unix_timestamp"] >= self.start) &
                      (data["unix_timestamp"] <= self.end)]["pedalpressure(kPa)"]
        if result.empty:
            try:
                result = data[(data["unix_timestamp"] >= self.start)].iloc[0]["pedalpressure(kPa)"]
            except IndexError:
                return [None]

        # df = df[df['unix_timestamp'].between(99, 101, inclusive=True)]
        # rescale/downsample
        samples = self.resample(result.shape[0])
        if (len(samples) == 1) and (result.shape[0] == 1):  # workaround against invalid data
            result = result.values
        else:
            result = result.iloc[samples].values
        assert len(result) != 0
        return result

    def get_rtk_track_info(self):
        '''
        example:
        ........................
        '''
        return None

    def get_rtk_pos_info(self):
        '''
        example:
        ........................
        '''
        return None

    def get_turn_signal_info(self):
        '''
        example:
        # timestamp,unix_timestamp,lturn,rturn
        20170410104105.136718750,1491846065.135316610,0.000000,0.000000
        '''
        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "general/csv/turn_signal_info.csv"
        data = pd.read_csv(csv_filename)
        result = data[(data["unix_timestamp"] >= self.start) &
                      (data["unix_timestamp"] <= self.end)][["lturn", "rturn"]]
        if result.empty:
            try:
                result = data[(data["unix_timestamp"] >= self.start)].iloc[0][["lturn", "rturn"]]
            except IndexError:
                return [[None, None]]

        # df = df[df['unix_timestamp'].between(99, 101, inclusive=True)]
        # rescale/downsample
        samples = self.resample(result.shape[0])
        # workaround against invalid data
        if (len(samples) == 1) and (result.shape == (2,)):
            result = [result.values]
        else:
            result = result.iloc[samples].values
        assert len(result) != 0
        return result

    def get_yaw_info(self):
        '''
        example:
        # timestamp,unix_timestamp,yaw(degree/s)
        20170410104105.109375000,1491846065.109578609,-0.244141
        '''
        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "general/csv/yaw_info.csv"
        data = pd.read_csv(csv_filename)
        # import ipdb; ipdb.set_trace()
        result = data[(data["unix_timestamp"] >= self.start) &
                      (data["unix_timestamp"] <= self.end)]["yaw(degree/s)"]
        if result.empty:
            try:
                result = data[(data["unix_timestamp"] >= self.start)].iloc[0]["yaw(degree/s)"]
            except IndexError:
                return [None]
        # df = df[df['unix_timestamp'].between(99, 101, inclusive=True)]
        # rescale/downsample
        samples = self.resample(result.shape[0])
        # workaround against invalid data
        if (len(samples) == 1) and (result.shape[0] == 1):
            result = result.values
        else:
            result = result.iloc[samples].values
        assert len(result) != 0
        return result

    def convert_time(self, event):
        '''Pandas(Index=3583, layer=5, event_type=68,
                  session_id='201704111540', start=6537340, end=6541750)
        event[3] - session_id
        event[4] - start
        event[5] - end
        '''
        csv_filename = self.session_template.format(self.root,
                                                    self.session_id[:4],
                                                    self.session_id[4:6],
                                                    self.session_id[6:8],
                                                    self.session_id)
        csv_filename += "camera/center/png_timestamp.csv"
        reference_frame_index = pd.read_csv(csv_filename, usecols=[0, 1, 2])

        # 30 frames per second
        start_frame = int(event["start"] * .03)
        end_frame = min(int(math.ceil(event["end"] * .03)), reference_frame_index.shape[0] - 1)
        assert start_frame < end_frame
        # print start_frame, end_frame, reference_frame_index.shape
        return reference_frame_index.iloc[start_frame][1], reference_frame_index.iloc[end_frame][1]

    def get_data(self, event, sensors, freq, max_len=20000):
        '''Pandas(Index=3583, layer=5, event_type=68,
                  session_id='201704111540', start=6537340, end=6541750)
        '''
        self.session_id = event["session_id"]
        self.freq = freq

        # in-memory and file cache
        if self.data:
            # the data is cached in this order
            # sensor_index = ["accel", "steer", "vel", "brake", "turn", "yaw"]
            # return [self.data[event.name][sensor_index.index(sensor)] for sensor in sensors]
            raise Exception("Not implemented")
        elif path.exists(path.join(self.cache_path, "general_sensors_{0}.npy").format(event.name)):
            filename = path.join(self.cache_path, "general_sensors_{0}.npy").format(event.name)
            data_record = np.load(filename)
            sample = (np.expand_dims(data_record[0], axis=1),
                      data_record[1],
                      np.expand_dims(data_record[2], axis=1),
                      np.expand_dims(data_record[3], axis=1),
                      data_record[4],
                      np.expand_dims(data_record[5], axis=1))
            sample = np.concatenate(sample, axis=1)[:max_len]
        else:
            # convert time based on csv data about timestamps
            self.start, self.end = self.convert_time(event)
            data_record = [self.sensor_dict[sensor]() for sensor in sensors]

            sample = (np.expand_dims(data_record[0], axis=1),
                      data_record[1],
                      np.expand_dims(data_record[2], axis=1),
                      np.expand_dims(data_record[3], axis=1),
                      data_record[4],
                      np.expand_dims(data_record[5], axis=1))


            sample = np.concatenate(sample, axis=1)[:max_len]
        if sample[0, 0] is None:
            sample = np.zeros((1, 8), dtype=np.float32)
            print("Invalid Sample")
        return sample


class CameraReader(object):
    # FIXME: the code is not thread-safe when out-of-cache data is used
    '''
    For the first time we need to extract CNN features in advance
    instead of employing CNN directly as a part of the model.

    This data reader takes an input a video stream instead of png files.
    In some cases it might be useful to switch to png reading because the
    average sequence duration is very small in comparison to the full-length
    video.

    The most important part:
    1. Read video file (potentially it can be preprocessed/rescaled in advance)
    2. Based on the file's metadata extract the target slice
    ( (end - start) x width x height)
    3. Send it to the tensorflow
    4. Tensorflow will perform resize and crop operations on images in batches

    Time alignment:
    Currently the code uses ffmpeg seeking so that it might produce sequences
    of frames with different length in comparison to those we have from
    general sensors.

    Training code is supposed to be aware of this and trim final
    representation appropriatly.
    '''

    def __init__(self, base_data_path, cache_path, headless=True):
        self.root = base_data_path
        self.cache_path = cache_path + "camera/"
        self.session_template = "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"
        # cached dataset
        if path.exists(path.join(self.cache_path,  "camera_data.npy")):
            self.data = np.load(open(path.join(self.cache_path,
                                               "camera_data.npy")))
        else:
            self.data = None

        """
        we should stop at this point to prevent an additional session construction
        """
        if headless:
            return
        '''
        Inception-ResnetV2 initialiazation
        TODO: Move parameters below into config
        '''

        slim_dir = "/home/vramanishka/repos/models/slim/"
        checkpoints_dir = 'experiments/slim_models'
        self.batch_size = 64
        import sys
        sys.path.insert(0, slim_dir)
        # FIXME: DEBUG
        from datasets import imagenet
        self.imagenet = imagenet
        import tensorflow as tf
        from nets import inception

        slim = tf.contrib.slim
        image_size = inception.inception_resnet_v2.default_image_size
        with tf.Graph().as_default():
            # TODO: remove extra data conversion we have below
            input_batch = tf.placeholder(dtype=tf.uint8,
                                         shape=(self.batch_size, 720, 1280, 3))
            resized_images = tf.image.resize_images(
                tf.image.convert_image_dtype(input_batch, dtype=tf.float32),
                [image_size, image_size]
                )
            self.resized_images = resized_images
            processed_images = tf.multiply(tf.subtract(resized_images,
                                                       0.5),
                                           2.0)
            # Create the model, use the default arg scope to configure
            # the batch norm parameters.
            with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                logits, endpoints = inception.inception_resnet_v2(processed_images,
                                                                  num_classes=1001,
                                                                  is_training=False)
            # for key in endpoints:
            #     print(key, endpoints[key].shape)
            # exit(1)
            self.pre_logits_flatten = endpoints['PreLogitsFlatten']
            self.pre_pool = endpoints['Conv2d_7b_1x1']
            self.probabilities = tf.nn.softmax(logits)

            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt'),
                slim.get_model_variables('InceptionResnetV2'))

            self.sess = tf.Session()
            init_fn(self.sess)
            self.input_batch = input_batch

    def get_frames(self, input_video, start, end, sample_rate):
        # FIXME: duration constraint should be added directly to the outer code
        '''
        input_video: absolute path
        start: start of the chunk in seconds
        end: end of the chunk in seconds
        sample_rate: fps for the frame extraction, e.g. "1/10" one frame every
        ten seconds
        '''
        # -vf "select='eq(pict_type,PICT_TYPE_I)'"
        command = ["ffmpeg",
                   '-ss',  str(start),
                   '-i', input_video,
                   '-t', str(end - start),
                   '-vf', 'fps=' + str(sample_rate),
                   # '-vf', 'scale=299:299', let it to be done by tensorflow
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo', '-']

        proc = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                stderr=open(os.devnull, 'w'),
                                bufsize=10 ** 7)
        nbytes = 3*720*1280
        nread = 0
        frame_range = []
        max_buf_len = 150

        while True:
            s = proc.stdout.read(nbytes)
            nread += 1
            if len(s) != nbytes:
                break
            if len(frame_range) > max_buf_len:
                print("Warning! Frame segment is too long.")
                continue
            else:
                result = np.fromstring(s, dtype='uint8')
                result = np.reshape(result,
                                    (720, 1280, len(s) // (1280 * 720)))
                frame_range.append(result)
        proc.wait()
        del proc
        return np.asarray(frame_range, dtype=np.uint8)

    def debug_feature_extraction(self, np_image, start, end,
                                 probabilities_batch):
        import cv2
        sorted_inds = []
        names = self.imagenet.create_readable_names_for_imagenet_labels()
        for image_ind in range(end-start):
            print(image_ind, probabilities_batch.shape)
            probabilities = probabilities_batch[image_ind, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                                key=lambda x:x[1])]
            # print sorted_inds
            for i in range(5):
                index = sorted_inds[i]
                prob = probabilities[index] * 100
                print('{0} Probability {1:0.2f}\% => {2}'.format(i,
                                                                 prob,
                                                                 names[index]))
            time.sleep(1 / 3.)
            cv2.imshow('i', cv2.cvtColor(np_image[image_ind],
                                         cv2.COLOR_BGR2RGB))
            if cv2.waitKey(30) == 27:
                exit(0)

    def load_segment(self):
        '''
        ffmpeg data reading/tensorflow image processing should be parallelized
        '''
        # 2017-04-11-12-02-32.mp4
        preview_template = self.session_template.format(self.root,
                                                        self.session_id[:4],
                                                        self.session_id[4:6],
                                                        self.session_id[6:8],
                                                        self.session_id) + "camera/center/*.mp4"
        video_full_path = glob.glob(preview_template)[0]
        # TODO: use a queue here to read video files asynchronously
        read_s = time.time()
        frames = self.get_frames(video_full_path, self.start / 1000.,
                                 self.end / 1000., self.freq)
        read_t = time.time() - read_s
        result = []
        bs = self.batch_size
        proc_s = time.time()
        for start, end in zip(range(0, frames.shape[0] + bs, bs),
                              range(self.batch_size, frames.shape[0] + bs, bs)):
            current_batch = np.zeros((bs, 720, 1280, 3), dtype=np.uint8)
            current_batch[:min(end, frames.shape[0]) - start] = frames[start:end]
            # which layer to use as descriptors?
            # TODO: increase throughput using larger batches and combining images
            # from different streams
            _, _, _, pre_pool = self.sess.run([self.resized_images,
                                               self.probabilities,
                                               self.pre_logits_flatten,
                                               self.pre_pool],
                                              feed_dict={
                                                  self.input_batch: current_batch
                                                  })
            valid_range = min(end, frames.shape[0]) - start
            result = result + list(pre_pool[:valid_range])

            # FIXME: DEBUG
            # debug_feature_extraction(np_image, start, end, probabilities_batch)
        # print(len(frames), "reading: {0} processing: {1}".format(read_t, time.time() - proc_s))


        # TODO: sanity check: compare resulting frame subset to the
        # original one in ELAN
        return np.asarray(result, np.float32)

    def get_data(self, event, freq, max_len=150):
        '''
        Pandas(Index=3583, layer=5, event_type=68,
               session_id='201704111540', start=6537340, end=6541750)
        '''
        self.session_id = event["session_id"]
        self.freq = freq
        # in-memory and file cache
        if self.data:
            result = self.data[event.name]
        elif path.exists(path.join(self.cache_path, "f_camera_{0}.npy").format(event.name)):
            result = np.load(path.join(self.cache_path,
                                       "f_camera_{0}.npy").format(event.name))
            # print("From cache")
        else:
            # start/end here represent a relative offset inside the video stream
            self.start, self.end = event["start"], event["end"]
            if (event["end"] - event["start"]) / 1000. * freq > max_len:
                print("Warning! Trimming event:" + str(event))
                self.end = self.start + (max_len / freq) * 1000
            result = self.load_segment()[:max_len]
        # directly use provided time to get frame ranges
        return result[:max_len]


class DataReader(object):
    def __init__(self, root, cache_dir, dr_type="both"):
        # GeneralSensorsReader(cfg.root, cfg.cache_dir)
        # CameraReader(cfg.root, cfg.cache_dir)
        self.type = dr_type
        if dr_type == "CANbus":
            self.data_reader = GeneralSensorsReader(root, cache_dir)
            self.get_data = partial(self.data_reader.get_data, sensors=["accel", "steer", "vel",
                                                                        "brake", "turn", "yaw"])
        elif dr_type == "camera":
            self.data_reader = CameraReader(root, cache_dir)
            self.get_data = self.data_reader.get_data
        elif dr_type == "both":
            def get_data(self, event, freq, max_len):
                # -------------------------------------------------------------------
                seq_CAN = self.data_reader.CAN.get_data(event,
                                                        ["accel", "steer", "vel",
                                                         "brake", "turn", "yaw"],
                                                        freq=freq, max_len=max_len)
                seq_cam = self.data_reader.camera.get_data(event, freq, max_len)
                s_len = min(len(seq_CAN), len(seq_cam), max_len)
                result = np.concatenate([seq_CAN[:s_len], seq_cam[:s_len]], axis=1)

                return result

            class Object():
                pass

            self.data_reader = Object()
            self.data_reader.CAN = GeneralSensorsReader(root, cache_dir)
            self.data_reader.camera = CameraReader(root, cache_dir)
            self.get_data = get_data
        else:
            raise Exception("Uknown data reader")
