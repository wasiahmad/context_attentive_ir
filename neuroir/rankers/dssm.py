import torch
import torch.nn as nn
import torch.nn.functional as f
from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


class DSSM(nn.Module):
    """Implementation of the deep semantic similarity model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(DSSM, self).__init__()

        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)
        self.emb_drop = nn.Dropout(p=args.dropout_emb)

        self.query_mlp = nn.Sequential(
            nn.Linear(args.emsize, args.nhid),
            nn.Tanh(),
            nn.Linear(args.nhid, args.nout),
            nn.Tanh()
        )
        self.doc_mlp = nn.Sequential(
            nn.Linear(args.emsize, args.nhid),
            nn.Tanh(),
            nn.Linear(args.nhid, args.nout),
            nn.Tanh()
        )

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
        embedded_queries = self.emb_drop(embedded_queries)
        embedded_queries = embedded_queries.max(1)[0]  # max-pooling

        # embed document
        doc_rep = batch_docs.view(batch_size * num_docs, dlen).unsqueeze(2)
        embedded_docs = self.word_embeddings(doc_rep)
        embedded_docs = self.emb_drop(embedded_docs)
        embedded_docs = embedded_docs.max(1)[0]  # max-pooling
        embedded_docs = embedded_docs.view(batch_size, num_docs, -1)

        query_rep = self.query_mlp(embedded_queries)
        doc_rep = self.doc_mlp(embedded_docs)
        query_rep = query_rep.unsqueeze(1).expand(*doc_rep.size())
        scores = f.cosine_similarity(query_rep, doc_rep, dim=2)
        return scores
