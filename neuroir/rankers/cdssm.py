import torch
import torch.nn as nn
import torch.nn.functional as f
from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


class CDSSM(nn.Module):
    """Implementation of the convolutional deep semantic similarity model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(CDSSM, self).__init__()

        self.window = 3
        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)
        self.emb_drop = nn.Dropout(p=args.dropout_emb)

        K = self.window * args.emsize
        L = args.nhid
        KERNEL_SIZE = 3
        O = args.nout

        self.query_conv = nn.Conv1d(K, L, KERNEL_SIZE)
        self.query_sem = nn.Linear(L, O)

        self.doc_conv = nn.Conv1d(K, L, KERNEL_SIZE)
        self.doc_sem = nn.Linear(L, O)

    def _interleave_tensor(self, inp):
        dim = inp.shape[1]
        assert dim >= self.window
        constituents = []
        offset = dim - self.window + 1
        for i in range(self.window):
            constituents.append(inp[:, i:offset + i, :])
        out = torch.cat(constituents, dim=-1)
        return out

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

        # query encoding
        embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
        embedded_queries = self.emb_drop(embedded_queries)  # b,s,h
        embedded_queries = self._interleave_tensor(embedded_queries)  # b,s-2,3h
        query_rep = self.query_conv(embedded_queries.transpose(1, 2)).transpose(1, 2)
        query_rep = f.tanh(self.query_sem(f.tanh(query_rep)))
        latent_query_rep = query_rep.max(1)[0]  # max-pooling

        # document encoding
        doc_rep = batch_docs.view(batch_size * num_docs, dlen).unsqueeze(2)
        embedded_docs = self.word_embeddings(doc_rep)
        embedded_docs = self.emb_drop(embedded_docs)  # b,s,h
        embedded_docs = self._interleave_tensor(embedded_docs)  # b,s-2,3h
        doc_rep = self.doc_conv(embedded_docs.transpose(1, 2)).transpose(1, 2)
        doc_rep = f.tanh(self.doc_sem(f.tanh(doc_rep)))
        latent_doc_rep = doc_rep.max(1)[0]  # max-pooling
        latent_doc_rep = latent_doc_rep.view(batch_size, num_docs, -1)

        # compute loss
        latent_query_rep = latent_query_rep.unsqueeze(1).expand(*latent_doc_rep.size())
        scores = f.cosine_similarity(latent_query_rep, latent_doc_rep, dim=2)
        return scores
