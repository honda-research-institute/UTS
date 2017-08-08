
def create_model(cfg):

    model = None
    print(cfg.model)

    if cfg.model == 'seq2seq_basic':
        from .seq2seq_basic import Seq2seqBasic
        model = Seq2seqBasic(cfg)
    else:
        raise ValueError("Model %s not recognized" % cfg.model)

    model.build_network()
    return model
