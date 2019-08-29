import numpy
import torch
import torch.nn as nn
import torch.nn.functional as f

from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


class DRMM(nn.Module):
    """Implementation of the deep relevance matching model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(DRMM, self).__init__()

        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)
        self.emb_drop = nn.Dropout(p=args.dropout_emb)

        self.nbins = args.nbins
        self.bins = [-1.0, -0.5, 0, 0.5, 1.0, 1.0]

        self.gating_network = GatingNetwork(args.emsize)
        self.ffnn = nn.Sequential(nn.Linear(self.nbins, 1), nn.Linear(1, 1))
        self.output = nn.Linear(1, 1)

    def forward(self, batch_queries, query_len, batch_docs, doc_len):
        """
        Forward function of the match tensor model. Return average loss for a batch of sessions.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param query_len: 1d numpy array [batch_size]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :param doc_len: 2d numpy array [batch_size x num_clicks_per_query]
        :return: score representing click probability [batch_size x num_clicks_per_query]
        """
        assert batch_queries.shape[0] == batch_docs.shape[0]
        batch_size = batch_queries.shape[0]
        qlen = batch_queries.shape[1]
        num_docs, dlen = batch_docs.shape[1], batch_docs.shape[2]
        use_cuda = batch_queries.is_cuda

        # embed query
        embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
        # batch_size x max_query_len x emsize
        embedded_queries = self.emb_drop(embedded_queries)

        # batch_size x num_rel_docs x max_q_len
        term_weights = self.gating_network(embedded_queries).unsqueeze(1).expand(
            batch_size, num_docs, qlen)

        # embed documents
        doc_rep = batch_docs.view(batch_size * num_docs, dlen)
        embedded_docs = self.word_embeddings(doc_rep.unsqueeze(2))
        # batch_size * num_rel_docs x max_doc_len x emsize
        embedded_docs = self.emb_drop(embedded_docs)

        # batch_size x num_rel_docs x max_query_len x emsize
        embedded_queries = torch.stack([embedded_queries] * num_docs, dim=1)
        # batch_size * num_rel_docs x max_query_len x emsize
        embedded_queries = embedded_queries.contiguous().view(batch_size * num_docs, qlen, -1)

        # batch_size * num_rel_docs x max_query_len x max_doc_len x emsize
        embedded_queries = torch.stack([embedded_queries] * dlen, dim=2)
        # batch_size * num_rel_docs x max_query_len x max_doc_len x emsize
        embedded_docs = torch.stack([embedded_docs] * qlen, dim=1)

        cos_sim = f.cosine_similarity(embedded_queries, embedded_docs, 3)

        hist = numpy.apply_along_axis(
            lambda x: numpy.histogram(x, bins=self.bins), 2, cos_sim.detach().cpu().numpy())
        histogram_feats = torch.from_numpy(
            numpy.array([[axis2 for axis2 in axis1] for axis1 in hist[:, :, 0]])
        ).float()

        if use_cuda:
            histogram_feats = histogram_feats.cuda()

        ffnn_out = self.ffnn(histogram_feats).squeeze(2)
        ffnn_out = ffnn_out.view(batch_size, num_docs, -1).contiguous()
        weighted_ffnn_out = ffnn_out * term_weights
        score = self.output(torch.sum(weighted_ffnn_out, 2, keepdim=True)).squeeze(1)
        return score.view(batch_size, num_docs)


class GatingNetwork(nn.Module):
    """Term gating network"""

    def __init__(self, emsize):
        """"Constructor of the class"""
        super(GatingNetwork, self).__init__()
        self.weight = nn.Linear(emsize, 1)

    def forward(self, term_embeddings):
        """"Defines the forward computation of the gating network layer."""
        dot_out = self.weight(term_embeddings).squeeze(2)
        return f.softmax(dot_out, 1)
