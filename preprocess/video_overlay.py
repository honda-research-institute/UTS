"""
Frame reading/writing idea was taken from
http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
Warning: no signal handling, Ctrl+C may work improperly
"""
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import argparse
import numpy as np
import os
import pickle as pkl
import re
import subprocess as sp
from tqdm import tqdm

from video_writer import ffmpeg_video_writer
import sys
sys.path.append("../configs")
from base_config import BaseConfig

session_id = '201704150933'
#session_id = '201704141145'

output_size = [720, 480]
sample_rate = 3
preset = "medium"
caption_color = "white"
font_file = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"



def convert_time(time_str):
    # 00:00:55:555 or 00:00:55.555 -> seconds
    if time_str.find('.') != -1:
        time_str += '0'
        time_str = time_str.replace('.', ":")

    time_list = time_str.split(":")
    result = sum([float(unit) * int(60 ** (2 - idx)) for idx, unit in enumerate(time_list)])
    result += float(time_list[-1]) * 1e-3
    return result


def get_info(file_path):
    # returns info (e.g. resolution)
    # strings in the json are not formatted like 1043_Vantage_Point_00.43.08-00.43.59
    input_video = file_path

    command = ['ffmpeg', '-i', input_video]
    proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    proc.stdout.readline()
    proc.terminate()
    info = proc.stderr.read().decode("utf-8")
    print(info)

    match = re.search("\,[\s]+([0-9]+)x([0-9]+)", info)
    duration = re.search("Duration:\ (.+?)\, ", info)
    fps = re.search("\, ([0-9\.]+?)\ fps", info)

    if match is None or duration is None or fps is None:
        raise "Not found"

    return [[int(match.group(1)), int(match.group(2))], convert_time(duration.group(1)), float(fps.group(1))]


def filter_captions(captions, timestamp):
    result = []

    for caption in captions:
        if caption['start'] < timestamp < caption['end']:
            result.append(caption['text'])

    return result


def overlay_captions(cfg):

    video_filename = os.path.join(cfg.video_root, session_id+'/aligned_video.mp4')
    try:
        size, duration, fps = get_info(video_filename)
    except:
        print(size, duration, fps)
        print("File is not found or corrupted!")
        exit(1)

    output_filename = video_filename.replace('.mp4', '_'+cfg.name+'.mp4')
    print("Writing to: ", output_filename)

    '''
    font setup
    '''
    font = ImageFont.truetype(font_file, int(output_size[1] / 14.), encoding="unic")

    command = ["ffmpeg",
               '-i', video_filename,
               '-vf', 'fps=3',
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']

    proc = sp.Popen(command, stdout=sp.PIPE, stderr=open(os.devnull, 'w'), bufsize=10 ** 7)

    bs = 1  # currently slower with anything larger than batch_size=1
    nbytes = bs * 3 * size[0] * size[1]
    writer = ffmpeg_video_writer(output_filename, input_size=size, output_size=output_size,
                                 fps=10*fps, bitrate='2000k', codec='libx264', preset=preset)
    nframes = int(duration * sample_rate)
    nread = 0

    '''
    blocking part: single-threaded
    '''
    result_seg_name = cfg.name+'/result_seg.pkl'
    result_seg = pkl.load(open(os.path.join(cfg.result_root, result_seg_name), 'r'))
    seg = result_seg[session_id]['s']

    showing_shot_change = False
    show_counter = 0
    r = 100

    with tqdm(total=nframes) as pbar:
        while True:
            s = proc.stdout.read(nbytes)
            if len(s) != nbytes:
                # issue warning?
                pbar.close()
                break
            else:
                result = np.fromstring(s, dtype='uint8')
                result = np.reshape(result,
                                    (bs, size[1], size[0], len(s) // (size[0] * size[1] * bs)))
                for i in range(bs):
                    image = Image.fromarray(result[i])
                    draw = ImageDraw.Draw(image)

                    # current_captions = filter_captions(captions, nread / float(fps))

                    # ==== monkey patching
                    if (nread in seg):
                        draw.ellipse((size[0] - r, size[1] - r,
                                      size[0], size[1]), fill=(255, 0, 0))
                        draw.text((size[0] - 2 * r, size[1] - 30), 'NEW SHOT!',
                                  caption_color, font)
                        output = np.array(image)
                        for slowmo in range(30):
                            writer.write_frame(output)
                    try:
                        writer.write_frame(np.array(image))
                    except:
                        print("Cannot write anymore!")
                pbar.update(bs)
                nread += 1

    proc.wait()
    del proc
    writer.close()


def main():
    cfg = BaseConfig().parse()
    overlay_captions(cfg)


if __name__ == '__main__':

    main()
