import torch
import torch.nn as nn
import torch.nn.functional as f
from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


class ESM(nn.Module):
    """Implementation of the embedding space model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(ESM, self).__init__()

        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)

    def forward(self, batch_queries, query_len, batch_docs, doc_len):
        """
        Forward function of the dssm model. Return average loss for a batch of queries.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param query_len: 1d numpy array [batch_size]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :param doc_len: 2d numpy array [batch_size x num_clicks_per_query]
        :return: softmax score representing click probability [batch_size x num_rel_docs_per_query]
        """
        assert batch_queries.shape[0] == batch_docs.shape[0]
        batch_size = batch_queries.shape[0]
        qlen = batch_queries.shape[1]
        num_docs, dlen = batch_docs.shape[1], batch_docs.shape[2]

        # embed query
        embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
        embedded_queries = embedded_queries.mean(1)  # averaging

        # embed document
        doc_rep = batch_docs.view(batch_size * num_docs, dlen).unsqueeze(2)
        embedded_docs = self.word_embeddings(doc_rep)
        embedded_docs = embedded_docs.mean(1)  # averaging
        doc_rep = embedded_docs.view(batch_size, num_docs, -1)

        query_rep = embedded_queries.unsqueeze(1).expand(*doc_rep.size())
        scores = f.cosine_similarity(query_rep, doc_rep, dim=2)
        return scores
