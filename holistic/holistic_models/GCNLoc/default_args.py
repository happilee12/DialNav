from argparse import Namespace

def get_default_args(basepath):
    """Get default arguments for ScaleVLN model"""
    return Namespace(
        panofeat_dir=f'{basepath}/dataset/node_feats/',
        embedding_dir=f'{basepath}/modules/loc/GCN/src/data/word_embeddings/',
        connect_dir=f'{basepath}/dataset/connectivity/',
        batch_size=None,
        max_nodes=345, max_nodes_test=345, 
        pano_embed_size=2048, rnn_embed_size=300, rnn_hidden_size=1024,
        gcn=True, attention=False, 
        train=False, 
        model_save=False,
        geodistance_file=f'{basepath}/dataset/localization/geodistance_nodes.json', 
        evaluate=False,
        bidirectional=True,
        num_gcn_layers=3,
        bert_enc=True,
        lr=0.0001,
    ) 