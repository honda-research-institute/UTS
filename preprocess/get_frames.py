"""
    Extract video frames using ffmpeg
"""

import os
import sys
import glob
import pandas as pd

sys.path.append("../configs")
from base_config import BaseConfig

def main():
    cfg = BaseConfig().parse()
    sample_rate = 3    # 3 fps

    session_template = "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"

    for session_id in cfg.train_session:
        if os.path.isdir(cfg.DATA_ROOT+'/frames/'+session_id):
            # pass if already extrated
            continue

        print session_id
        os.makedirs(cfg.DATA_ROOT+'/frames/'+session_id)
        session_folder = session_template.format(cfg.DATA_ROOT,
                                                    session_id[:4],
                                                    session_id[4:6],
                                                    session_id[6:8],
                                                    session_id)
        video_filename = glob.glob(session_folder + "camera/center/*mp4")[0]

        command = ["ffmpeg",
                   '-i', video_filename,
                   '-vf', 'fps='+str(sample_rate),
                   cfg.DATA_ROOT+'/frames/'+session_id+'/frame_%04d.jpg']
        os.system(' '.join(command))

if __name__ == '__main__':
    main()
