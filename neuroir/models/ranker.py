import copy
import logging

import torch
import torch.optim as optim
import torch.nn.functional as f

from tqdm import tqdm
from prettytable import PrettyTable

from neuroir.config import override_model_args
from neuroir.rankers.dssm import DSSM
from neuroir.rankers.cdssm import CDSSM
from neuroir.rankers.duet import DUET
from neuroir.rankers.esm import ESM
from neuroir.rankers.arci import ARCI
from neuroir.rankers.arcii import ARCII
from neuroir.rankers.drmm import DRMM
from neuroir.rankers.mtensor import MatchTensor
from neuroir.utils.misc import count_file_lines

logger = logging.getLogger(__name__)


class Ranker(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if args.model_type.upper() == 'DSSM':
            self.network = DSSM(self.args)
            self.criterion = self.compute_loss
        elif args.model_type.upper() == 'CDSSM':
            self.network = CDSSM(self.args)
            self.criterion = self.compute_loss
        elif args.model_type.upper() == 'ESM':
            self.network = ESM(self.args)
        elif args.model_type.upper() == 'DUET':
            self.network = DUET(self.args)
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif args.model_type.upper() == 'ARCI':
            self.network = ARCI(self.args)
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif args.model_type.upper() == 'ARCII':
            self.network = ARCII(self.args)
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif args.model_type.upper() == 'DRMM':
            self.network = DRMM(self.args)
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif args.model_type.upper() == 'MATCH_TENSOR':
            self.network = MatchTensor(self.args)
            self.criterion = torch.nn.BCEWithLogitsLoss()
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

    @staticmethod
    def compute_loss(predictions, target):
        """
        Compute negative log-likelihood loss for a batch of predictions.
        :param predictions: 2d tensor [batch_size x num_rel_docs_per_query]
        :param target: 2d tensor [batch_size x num_rel_docs_per_query]
        :return: average negative log-likelihood loss over the input mini-batch [autograd Variable]
        """
        predictions = f.log_softmax(predictions, dim=-1)
        loss = -(predictions * target).sum(1)
        return loss.mean()

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
        emb_layer = self.network.word_embeddings
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
            for p in self.network.word_embeddings.parameters():
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

        documents = ex['doc_rep']
        queries = ex['que_rep']
        que_len = ex['que_len']
        doc_len = ex['doc_len']
        labels = ex['label'].float()
        if self.use_cuda:
            documents = documents.cuda(non_blocking=True)
            queries = queries.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            que_len = que_len.cuda(non_blocking=True)
            doc_len = doc_len.cuda(non_blocking=True)

        # Run forward
        scores = self.network(queries, que_len, documents, doc_len)
        loss = self.criterion(scores, labels)
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

        documents = ex['doc_rep']
        queries = ex['que_rep']
        que_len = ex['que_len']
        doc_len = ex['doc_len']
        if self.use_cuda:
            documents = documents.cuda(non_blocking=True)
            queries = queries.cuda(non_blocking=True)
            que_len = que_len.cuda(non_blocking=True)
            doc_len = doc_len.cuda(non_blocking=True)

        # Run forward
        scores = self.network(queries, que_len, documents, doc_len)
        scores = f.softmax(scores, dim=-1)

        return scores

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
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return Ranker(args, src_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = Ranker(args, src_dict, state_dict)
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
