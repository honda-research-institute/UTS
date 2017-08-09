import os
import sys
import glob

sys.path.append("../configs")
from base_config import BaseConfig


def main():

    cfg = BaseConfig().parse()

    day_list = glob.glob(cfg.DATA_ROOT+'/*data_collection')
    day_list = sorted(day_list)
    
    session_list = []
    test_session = []
    for day in day_list:
        l = glob.glob(day + '/*ITS')
        l = sorted(l)

        for session in l:
            if os.path.isdir(session+'/camera') and os.path.isdir(session+'/general'):
                session_list.append(os.path.basename(session).strip('_ITS'))

                if os.path.isdir(session+'/annotation'):
                    test_session.append(os.path.basename(session).strip('_ITS'))

    with open(cfg.DATA_ROOT+'/session_list.txt', 'w') as fout:
        for s in session_list:
            fout.write(s+'\n')
    with open(cfg.DATA_ROOT+'/test_session.txt', 'w') as fout:
        for s in test_session:
            fout.write(s+'\n')

if __name__ == '__main__':
    main()
