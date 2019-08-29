# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/config.py
""" Implementation of all available options """
from __future__ import print_function

import argparse
import logging
from .hyparam import get_model_specific_params

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_OPTIONS = {
    'model_type', 'emsize', 'use_word', 'use_char_ngram',
    'copy_attn', 'resue_copy_attn', 'force_copy'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings', 'optimizer', 'learning_rate', 'momentum',
    'weight_decay', 'rnn_padding', 'dropout_rnn', 'dropout',
    'dropout_emb', 'cuda', 'grad_clipping', 'lr_decay'
}

DATA_OPTIONS = {
    'max_doc_len', 'max_query_len', 'num_candidates', 'force_pad'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Data options
    data = parser.add_argument_group('Data parameters')
    data.add_argument('--max_doc_len', type=int, default=200,
                      help='Maximum allowed length for the documents')
    data.add_argument('--max_query_len', type=int, default=10,
                      help='Maximum allowed length for the queries')
    data.add_argument('--num_candidates', type=int, default=10,
                      help='Number of candidates per query')

    # Model architecture
    model = parser.add_argument_group('Neural QA Reader Architecture')
    model.add_argument('--use_word', type='bool', default=True,
                       help='Use word embeddings as a part of the input representations.')
    model.add_argument('--use_char_ngram', type=int, default=0,
                       help='Use char ngram for the input representations.')
    model.add_argument('--emsize', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--rnn_type', type=str, default='LSTM',
                       help='RNN type: LSTM, GRU')
    model.add_argument('--bidirection', type='bool', default=True,
                       help='use bidirectional recurrent unit')
    model.add_argument('--nlayers', type=int, default=1,
                       help='Number of encoding layers')

    seq2seq = parser.add_argument_group('Seq2seq Model Specific Params')
    seq2seq.add_argument('--attn_type', type=str, default='general',
                         help='Attention type for the seq2seq [dot, general, mlp]')
    seq2seq.add_argument('--coverage_attn', type='bool', default=False,
                         help='Use coverage attention')
    seq2seq.add_argument('--copy_attn', type='bool', default=False,
                         help='Use copy attention')
    seq2seq.add_argument('--force_copy', type='bool', default=False,
                         help='Apply force copying')
    seq2seq.add_argument('--reuse_copy_attn', type='bool', default=False,
                         help='Reuse encoder attention')

    # Optimization details
    optim = parser.add_argument_group('Neural QA Reader Optimization')
    optim.add_argument('--dropout_emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout_rnn', type=float, default=0.2,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout for NN layers')
    optim.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for the optimizer')
    optim.add_argument('--lr_decay', type=float, default=0.95,
                       help='Decay ratio for learning rate')
    optim.add_argument('--grad_clipping', type=float, default=10,
                       help='Gradient clipping')
    optim.add_argument('--early_stop', type=int, default=5,
                       help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix_embeddings', type='bool', default=False,
                       help='Keep word embeddings fixed (use pretrained)')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_OPTIONS, MODEL_OPTIMIZER, DATA_OPTIONS

    model = args.model_type.upper()
    required_args = MODEL_OPTIONS | MODEL_OPTIMIZER | DATA_OPTIONS

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    # using a fixed set of hyper-parameters that are model specific
    for k, v in get_model_specific_params(model, field='arch').items():
        arg_values[k] = v
    return argparse.Namespace(**arg_values)


def update_model_args(args):
    model = args.model_type.upper()
    old_args = vars(args)
    for k, v in get_model_specific_params(model, field='data').items():
        old_args[k] = v
    return argparse.Namespace(**old_args)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))

    return argparse.Namespace(**old_args)
