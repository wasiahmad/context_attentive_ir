# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

import neuroir.config as config
from tqdm import tqdm
from neuroir.utils.timer import AverageMeter, Timer
import neuroir.inputters.ranker.vector as vector
from neuroir.inputters.ranker import data, utils

from neuroir.models.ranker import Ranker
from neuroir.eval.ltorank import MAP, MRR, precision_at_k

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Ranker parameters
    ranker = parser.add_argument_group('Ranker')
    ranker.add_argument('--model_type', type=str, default='dssm',
                        choices=['dssm', 'cdssm', 'duet', 'esm',
                                 'arci', 'arcii', 'drmm', 'match_tensor'],
                        help='Name of the ranking model')

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', type=str, choices=['msmarco'],
                       default='msmarco', help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/msmarco/',
                       help='Directory of training/validation data')
    files.add_argument('--train_file', type=str,
                       default='train.json',
                       help='Preprocessed train file')
    files.add_argument('--dev_file', type=str,
                       default='dev.json',
                       help='Preprocessed dev file')
    files.add_argument('--test_file', type=str,
                       default='test.json',
                       help='Preprocessed dev file')
    files.add_argument('--embed_dir', type=str, default='/data/glove/',
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding_file', type=str, default='',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='All text will be lower-cased')
    preprocess.add_argument('--restrict_vocab', type='bool', default=False,
                            help='Only use pre-trained words in embedding_file')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')
    preprocess.add_argument('--force_pad', type='bool', default=False,
                            help='Force padding while processing the data')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='map',
                         help='The evaluation metric used for model selection')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not args.only_test and not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    args.test_file = os.path.join(args.data_dir, args.test_file)
    if not os.path.isfile(args.test_file):
        raise IOError('No such file: %s' % args.test_file)

    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            # if first line is of form count/dim.
            line = f.readline().rstrip().split(' ')
            dim = int(line[1]) if len(line) == 2 \
                else len(line) - 1
        args.emsize = dim
    elif not args.emsize:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    src_dict = utils.build_word_and_char_dict(args, train_exs + dev_exs,
                                              dict_size=args.src_vocab_size)
    logger.info('Num words in vocabulary = %d' % len(src_dict))

    # Initialize model
    model = Ranker(config.get_model_args(args), src_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(src_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ce_loss = AverageMeter()
    epoch_time = Timer()
    model.optimizer.param_groups[0]['lr'] = \
        model.optimizer.param_groups[0]['lr'] * args.lr_decay

    pbar = tqdm(data_loader)
    pbar.set_description("%s" % 'Epoch = %d [ce_loss = x.xx]' % global_stats['epoch'])

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        net_loss = model.update(ex)
        ce_loss.update(net_loss.item(), bsz)

        log_info = 'Epoch = %d [ce_loss = %.2f]' % \
                   (global_stats['epoch'], ce_loss.avg)

        pbar.set_description("%s" % log_info)
        torch.cuda.empty_cache()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats=None):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    map = AverageMeter()
    mrr = AverageMeter()
    prec_1 = AverageMeter()
    prec_3 = AverageMeter()
    prec_5 = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for ex in pbar:
            ids, batch_size = ex['ids'], ex['batch_size']
            scores = model.predict(ex)
            predictions = np.argsort(-scores.cpu().numpy())  # sort in descending order
            labels = ex['label'].numpy()

            map.update(MAP(predictions, labels))
            mrr.update(MRR(predictions, labels))
            prec_1.update(precision_at_k(predictions, labels, 1))
            prec_3.update(precision_at_k(predictions, labels, 3))
            prec_5.update(precision_at_k(predictions, labels, 5))

            if global_stats is None:
                pbar.set_description('[testing ... ]')
            else:
                pbar.set_description("%s" % 'Epoch = %d [validating... ]' % global_stats['epoch'])

            examples += batch_size

    result = dict()
    result['map'] = map.avg
    result['mrr'] = mrr.avg
    result['prec@1'] = prec_1.avg
    result['prec@3'] = prec_3.avg
    result['prec@5'] = prec_5.avg

    if global_stats is None:
        logger.info('test results: MAP = %.2f | MRR = %.2f | Prec@1 = %.2f | ' %
                    (result['map'], result['mrr'], result['prec@1']) +
                    'Prec@3 = %.2f | Prec@5 = %.2f | examples = %d | ' %
                    (result['prec@3'], result['prec@5'], examples) +
                    'time elapsed = %.2f (s)' %
                    (eval_time.time()))
    else:
        logger.info('valid official: Epoch = %d | MAP = %.2f | ' %
                    (global_stats['epoch'], result['map']) +
                    'MRR = %.2f | Prec@1 = %.2f | Prec@3 = %.2f | ' %
                    (result['mrr'], result['prec@1'], result['prec@3']) +
                    'Prec@5 = %.2f | examples = %d | valid time = %.2f (s)' %
                    (result['prec@5'], examples, eval_time.time()))

    return result


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    args = config.update_model_args(args)
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')

    logger.info('Reading data...')
    if not args.only_test:
        train_exs = utils.load_data(args, args.train_file,
                                    max_examples=args.max_examples,
                                    dataset_name=args.dataset_name)
        logger.info('Num train examples = %d' % len(train_exs))

        dev_exs = utils.load_data(args, args.dev_file,
                                  max_examples=args.max_examples,
                                  dataset_name=args.dataset_name)
        logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.only_test:
        if args.pretrained:
            model = Ranker.load(args.pretrained, args)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = Ranker.load(args.model_file, args)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = Ranker.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = Ranker.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(args, train_exs, dev_exs)

            # Set up optimizer
            model.init_optimizer()

            logger.info('Total trainable parameters # %d' % model.count_parameters())
            table = model.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    if not args.only_test:
        train_dataset = data.RankerDataset(train_exs, model, shuffle=True)
        if args.sort_by_len:
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

        dev_dataset = data.RankerDataset(dev_exs, model, shuffle=False)
        if args.sort_by_len:
            dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                                  args.test_batch_size,
                                                  shuffle=False)
        else:
            dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    if args.model_type.upper() != 'ESM':  # no training for ESM
        if not args.only_test:
            logger.info('-' * 100)
            logger.info('Starting training...')
            stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
            for epoch in range(start_epoch, args.num_epochs):
                stats['epoch'] = epoch

                # Train
                train(args, train_loader, model, stats)
                result = validate_official(args, dev_loader, model, stats)

                # Save best valid
                if result[args.valid_metric] > stats['best_valid']:
                    logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                                (args.valid_metric, result[args.valid_metric],
                                 stats['epoch'], model.updates))
                    model.save(args.model_file)
                    stats['best_valid'] = result[args.valid_metric]
                    stats['no_improvement'] = 0
                else:
                    stats['no_improvement'] += 1
                    if stats['no_improvement'] >= args.early_stop:
                        break

            # reset the model using best parameters
            model = Ranker.load(args.model_file)

    # --------------------------------------------------------------------------
    logger.info('-' * 100)
    logger.info('Starting evaluation...')
    # --------------------------------------------------------------------------
    test_exs = utils.load_data(args, args.test_file,
                               max_examples=args.max_examples,
                               dataset_name=args.dataset_name)
    logger.info('Num test examples = %d' % len(test_exs))

    test_dataset = data.RankerDataset(test_exs, model, shuffle=False)
    if args.sort_by_len:
        test_sampler = data.SortedBatchSampler(test_dataset.lengths(),
                                               args.test_batch_size,
                                               shuffle=False)
    else:
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )
    validate_official(args, test_loader, model)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Neural IR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
