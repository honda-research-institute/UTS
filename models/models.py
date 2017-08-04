
def create_model(cfg):

    model = None
    print(cfg.model)

    if cfg.model == 'seq2seq_recon':
        assert(cfg.modality_X == cfg.modality_Y)
        assert(cfg.iter_mode == 'recon')
        from .seq2seq_reconstruct import Seq2seqRecon
        model = Seq2seqRecon(cfg)
    else:
        raise ValueError("Model %s not recognized" % cfg.model)

    model.build_network()
    return model
