import copy
import logging

import torch
import torch.optim as optim
import torch.nn.functional as f

from tqdm import tqdm
from prettytable import PrettyTable

from neuroir.config import override_model_args
from neuroir.recommender.seq2seq import Seq2seq
from neuroir.recommender.hredqs import HredQS
from neuroir.utils.misc import count_file_lines, tens2sen
from neuroir.utils.copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align

logger = logging.getLogger(__name__)


class Recommender(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, tgt_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.tgt_dict = tgt_dict
        self.args.tgt_vocab_size = len(tgt_dict)
        self.type = args.model_type.upper()
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if self.type in ['SEQ2SEQ', 'ACG']:
            self.network = Seq2seq(self.args)
        elif self.type == 'HREDQS':
            self.network = HredQS(self.args)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.network.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        emb_layer = self.network.embedder.word_embeddings
        words = {w for w in words if w in self.src_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts, embedding = {}, {}
        with open(embedding_file) as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)

            duplicates = set()
            for line in tqdm(f, total=count_file_lines(embedding_file)):
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == emb_layer.word_vec_size + 1)
                w = self.src_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[w] = vec
                    else:
                        duplicates.add(w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[w].add_(vec)

            if len(duplicates) > 0:
                logging.warning(
                    'WARN: Duplicate embedding found for %s' % ', '.join(duplicates)
                )

        for w, c in vec_counts.items():
            embedding[w].div_(c)

        emb_layer.init_word_vectors(self.src_dict, embedding, self.args.fix_embeddings)
        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            for p in self.network.embedder.word_embeddings.parameters():
                p.requires_grad = False

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters, self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(parameters, self.args.learning_rate,
                                            weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        source_map, alignment = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda \
                else source_map

            alignment = align(ex['alignment'])
            alignment = alignment.cuda(non_blocking=True) if self.use_cuda \
                else alignment

        # all tensors are of shape: # bsz x sess_len-1 x max_len
        source_words = ex['source_words']
        target_words = ex['target_words']
        target_seq = ex['target_seq']
        # all tensors are of shape: # bsz x sess_len-1
        source_lens = ex['source_lens']
        target_lens = ex['target_lens']

        if self.type in ['SEQ2SEQ', 'ACG']:
            # sess_len-1 = 1
            source_words = source_words.squeeze(1)
            target_words = target_words.squeeze(1)
            target_seq = target_seq.squeeze(1)
            source_lens = source_lens.squeeze(1)
            target_lens = target_lens.squeeze(1)

        if self.use_cuda:
            source_words = source_words.cuda(non_blocking=True)
            source_lens = source_lens.cuda(non_blocking=True)
            target_words = target_words.cuda(non_blocking=True)
            target_lens = target_lens.cuda(non_blocking=True)
            target_seq = target_seq.cuda(non_blocking=True)

        # Run forward
        loss = self.network(source_rep=source_words,
                            source_len=source_lens,
                            target_rep=target_words,
                            target_len=target_lens,
                            target_seq=target_seq,
                            source_map=source_map,
                            alignment=alignment)
        if self.parallel:
            loss = loss.mean()

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        return loss

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda \
                else source_map

            if ex['alignment'][0][0] is not None:
                alignment = align(ex['alignment'])
                alignment = alignment.cuda(non_blocking=True) if self.use_cuda \
                    else alignment

            blank, fill = collapse_copy_scores(self.tgt_dict, ex['src_vocab'])

        # all tensors are of shape: # bsz x sess_len-1 x max_len
        source_words = ex['source_words']
        # all tensors are of shape: # bsz x sess_len-1
        source_lens = ex['source_lens']

        if self.type in ['SEQ2SEQ', 'ACG']:
            # sess_len-1 = 1
            source_words = source_words.squeeze(1)
            source_lens = source_lens.squeeze(1)

        if self.use_cuda:
            source_words = source_words.cuda(non_blocking=True)
            source_lens = source_lens.cuda(non_blocking=True)

        # Run forward
        decoder_fn = self.network.module.decode if self.parallel \
            else self.network.decode
        decoder_out = decoder_fn(source_rep=source_words,
                                 source_len=source_lens,
                                 max_len=self.args.max_query_len,
                                 src_dict=self.src_dict,
                                 tgt_dict=self.tgt_dict,
                                 src_map=source_map,
                                 alignment=alignment,
                                 blank=blank,
                                 fill=fill,
                                 source_vocabs=ex['src_vocab'])

        outputs = dict()
        outputs['ex_ids'] = None
        outputs['predictions'] = None
        outputs['targets'] = None
        outputs['src_sequences'] = None

        if self.type in ['SEQ2SEQ', 'ACG']:
            outputs['ex_ids'] = ex['ids']
            predictions = tens2sen(decoder_out['predictions'],
                                   self.tgt_dict,
                                   ex['src_vocab'])
            # ex['target_tokens'] is a 3d list
            outputs['targets'] = [[' '.join(query[1:-1]) for query in batch_item]
                                  for batch_item in ex['target_tokens']]
            outputs['src_sequences'] = [[' '.join(qtokens[1:-1]) for qtokens in session]
                                        for session in ex['source_tokens']]
            if "attentions" in decoder_out:
                for i in range(len(predictions)):
                    predictions[i] = replace_unknown(predictions[i],
                                                     decoder_out['attentions'][i],
                                                     ex['source_tokens'][i][0])
            outputs['predictions'] = predictions
        else:
            session_len = ex['session_len']
            outputs['ex_ids'] = [_id + str(i) for i in range(session_len)
                                 for _id in ex['ids']]
            outputs['predictions'] = []
            outputs['targets'] = []
            outputs['src_sequences'] = []
            for sidx in range(session_len):
                predictions = tens2sen(decoder_out['predictions'][:, sidx, :],
                                       self.tgt_dict,
                                       None)
                outputs['predictions'].extend(predictions)
                for bidx in range(ex['batch_size']):
                    tokens = ex['target_tokens'][bidx][sidx]
                    outputs['targets'].append([' '.join(tokens[1:-1])])
                    source_tokens = ex['source_tokens'][bidx][0:sidx + 1]
                    source_tokens = [' '.join(source[1:-1]) for source in source_tokens]
                    outputs['src_sequences'].append(' '.join(source_tokens))

        return outputs

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'src_dict': self.src_dict,
            'tgt_dict': self.tgt_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'src_dict': self.src_dict,
            'tgt_dict': self.tgt_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        tgt_dict = saved_params['tgt_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return Recommender(args, src_dict, tgt_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        tgt_dict = saved_params['tgt_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = Recommender(args, src_dict, tgt_dict, state_dict)
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
