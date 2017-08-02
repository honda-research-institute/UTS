import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

from configs.lstm_config import LSTMConfig
import utils.data_io as data_io


def main():

    # load default configuration
    cfg = LSTMConfig()
    print cfg.name

    if cfg.seq2seq:
        print "Seq2seq Model ..."
        from models.seq2seq_lstm import Seq2seqLSTM
    
        data = data_io.load_data(cfg, cfg.session_list[0])
        if cfg.modality == 'both':
            X = data[0]; Y = data[1]
        else:
            X = data; Y = data

        if cfg.predict:
            model = Seq2seqLSTM(batch_size=cfg.batch_size,
                                n_steps=cfg.n_steps,
                                n_input=X.shape[1],
                                n_hidden=cfg.n_hidden,
                                n_output=Y.shape[1],
                                reconstruct=cfg.reconstruct,
                                n_predict=cfg.n_predict)
        else:
            model = Seq2seqLSTM(batch_size=cfg.batch_size,
                                n_steps=cfg.n_steps,
                                n_input=X.shape[1],
                                n_hidden=cfg.n_hidden,
                                reconstruct=cfg.reconstruct)

        if cfg.is_Train:
            # load data
            print ("Loading data ...")
            data, vid = data_io.load_data_list(cfg)
            if cfg.modality == 'both':
                X = data[0]; Y = data[1]
            else:
                X = data; Y = data
    
            # Training
            model.init_train(n_epochs=cfg.n_epochs,
                             learning_rate=cfg.learning_rate,
                             result_path=os.path.join(cfg.result_root, cfg.name))
    
            model.train(X, vid, Y)

        else:
            # Extracting features

            for session_id in cfg.session_list:
                data = data_io.load_data(cfg, session_id)
                feat = model.extract_feat(data)
            
                data_io.save_data(feat, cfg, session_id)
            

if __name__ == "__main__":

    main()
