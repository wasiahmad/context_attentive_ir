from __future__ import print_function

ESM = {
    'arch': {
    },
    'data': {
    }
}

DSSM = {
    'arch': {
        'nhid': 300,
        'nout': 128
    },
    'data': {
        'use_char_ngram': 3,
        'src_vocab_size': 30000,
        'embedding_file': ''
    }
}

CDSSM = {
    'arch': {
        'nhid': 300,
        'nout': 128
    },
    'data': {
        'use_char_ngram': 3,
        'src_vocab_size': 30000,
        'embedding_file': ''
    }
}

DUET = {
    'arch': {
        'nfilters': 300,
        'local_filter_size': 1,
        'dist_filter_size': 3,
        'pool_size': 5,
    },
    'data': {
        'src_vocab_size': None,
        'force_pad': True,
        'fix_embeddings': True
    }
}

ARCI = {
    'arch': {
        'filters_1d': [256, 128],
        'kernel_size_1d': [3, 3],
        'maxpool_size_1d': [2, 2],
    },
    'data': {
        'src_vocab_size': None,
        'force_pad': True,
        'fix_embeddings': True,
    }
}

ARCII = {
    'arch': {
        'filters_1d': 128,
        'kernel_size_1d': 3,
        'filters_2d': [256, 128],
        'kernel_size_2d': [[3, 3], [3, 3]],
        'maxpool_size_2d': [[2, 2], [2, 2]],
    },
    'data': {
        'src_vocab_size': None,
        'force_pad': True,
        'fix_embeddings': True,
        'max_doc_len': 100,
        'max_query_len': 10
    }
}

DRMM = {
    'arch': {
        'nbins': 5
    },
    'data': {
        'src_vocab_size': None,
        'fix_embeddings': True
    }
}

MATCH_TENSOR = {
    'arch': {
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 1,
        'dropout_rnn': 0.2,
        'featsize': 40,
        'nhid_query': 30,
        'nhid_doc': 140,
        'nchannels': 50,
        'nfilters': 6,
        'match_filter_size': 20
    },
    'data': {
        'src_vocab_size': None,
        'fix_embeddings': True
    }
}

SEQ2SEQ = {
    'arch': {
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 2,
        'nhid': 512,
        'dropout_rnn': 0.2,
        'attn_type': 'general'
    },
    'data': {
        'tgt_vocab_size': 30000,
        'fix_embeddings': True
    }
}

HREDQS = {
    'arch': {
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 1,
        'nhid': 512,
        'dropout_rnn': 0.2,
        'nhid_session': 1024
    },
    'data': {
        'tgt_vocab_size': 30000,
        'fix_embeddings': True
    }
}

ACG = {
    'arch': {
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 1,
        'nhid': 512,
        'dropout_rnn': 0.2,
        'attn_type': 'general',
        'copy_attn': True,
        'reuse_copy_attn': True,
        'force_copy': False
    },
    'data': {
        'tgt_vocab_size': 10000,
        'fix_embeddings': True
    }
}

MNSRF = {
    'arch': {
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 1,
        'nhid_query': 512,
        'nhid_document': 512,
        'nhid_session': 1024,
        'dropout_rnn': 0.2,
        'regularize_coeff': 0.1,
        'alpha': 0.5
    },
    'data': {
        'tgt_vocab_size': 30000,
        'fix_embeddings': True
    }
}

M_MATCH_TENSOR = {
    'arch': {
        'featsize': 40,
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 1,
        'nhid_query': 30,
        'nhid_document': 140,
        'nhid_session': 300,
        'dropout_rnn': 0.2,
        'nchannels': 50,
        'nfilters': 6,
        'match_filter_size': 20,
        'regularize_coeff': 0.1,
        'alpha': 0.5
    },
    'data': {
        'max_doc_len': 100,
        'max_query_len': 10,
        'tgt_vocab_size': 30000,
        'fix_embeddings': True
    }
}

CARS = {
    'arch': {
        'rnn_type': 'LSTM',
        'bidirection': True,
        'nlayers': 1,
        'nhid_query': 256,
        'nhid_document': 256,
        'nhid_click': 512,
        'nhid_session_query': 512,
        'nhid_session_document': 512,
        'nhid_decoder': 512,
        'query_session_off': False,
        'doc_session_off': False,
        'dropout_rnn': 0.2,
        'attn_type': 'general',
        'mlp_nhid': 150,
        'pool_type': 'attn',
        'regularize_coeff': 0.1,
        'alpha': 0.1,
        'lambda1': 0.01,
        'lambda2': 0.0001,
        'turn_ranker_off': False,
        'turn_recommender_off': False
    },
    'data': {
        'tgt_vocab_size': 30000,
        'fix_embeddings': True
    }
}

MODEL_ARCHITECTURE = {
    'DSSM': DSSM,
    'CDSSM': CDSSM,
    'ESM': ESM,
    'DUET': DUET,
    'ARCI': ARCI,
    'ARCII': ARCII,
    'DRMM': DRMM,
    'MATCH_TENSOR': MATCH_TENSOR,
    'SEQ2SEQ': SEQ2SEQ,
    'HREDQS': HREDQS,
    'ACG': ACG,
    'MNSRF': MNSRF,
    'M_MATCH_TENSOR': M_MATCH_TENSOR,
    'CARS': CARS
}


def get_model_specific_params(model_name, field):
    return MODEL_ARCHITECTURE[model_name.upper()][field]
