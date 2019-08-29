import torch
import torch.nn as nn
from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


class ARCI(nn.Module):
    """Implementation of the convolutional matching model (ARC-I)."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(ARCI, self).__init__()

        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)
        self.emb_drop = nn.Dropout(p=args.dropout_emb)

        num_conv1d_layers = len(args.filters_1d)
        assert num_conv1d_layers == len(args.kernel_size_1d)
        assert num_conv1d_layers == len(args.maxpool_size_1d)

        query_feats = args.max_query_len
        doc_feats = args.max_doc_len

        query_conv1d_layers = []
        doc_conv1d_layers = []
        for i in range(num_conv1d_layers):
            inpsize = args.emsize if i == 0 else args.filters_1d[i - 1]
            pad = args.kernel_size_1d[i] // 2
            layer = nn.Sequential(
                nn.Conv1d(inpsize, args.filters_1d[i], args.kernel_size_1d[i],
                          padding=pad),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(args.maxpool_size_1d[i])
            )
            query_conv1d_layers.append(layer)
            layer = nn.Sequential(
                nn.Conv1d(inpsize, args.filters_1d[i], args.kernel_size_1d[i],
                          padding=pad),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(args.maxpool_size_1d[i])
            )
            doc_conv1d_layers.append(layer)

            doc_feats = doc_feats // args.maxpool_size_1d[i]
            query_feats = query_feats // args.maxpool_size_1d[i]
            assert query_feats != 0 and doc_feats != 0

        self.query_conv1d_layers = nn.ModuleList(query_conv1d_layers)
        self.doc_conv1d_layers = nn.ModuleList(doc_conv1d_layers)

        inpsize = (args.filters_1d[-1] * query_feats) + \
                  (args.filters_1d[-1] * doc_feats)
        self.mlp = nn.Sequential(
            nn.Linear(inpsize, inpsize // 2),
            nn.Linear(inpsize // 2, 1)
        )

    def forward(self, batch_queries, query_len, batch_docs, doc_len):
        """
        Forward function of the match tensor model. Return average loss for a batch of sessions.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param query_len: 1d numpy array [batch_size]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :param doc_len: 2d numpy array [batch_size x num_clicks_per_query]
        :return: click probabilities [batch_size x num_rel_docs_per_query]
        """
        assert batch_queries.shape[0] == batch_docs.shape[0]
        batch_size = batch_queries.shape[0]
        qlen = batch_queries.shape[1]
        num_docs, dlen = batch_docs.shape[1], batch_docs.shape[2]

        # embed query
        embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
        # batch_size x max_q_len x emsize
        embedded_queries = self.emb_drop(embedded_queries)

        inp_rep = embedded_queries.transpose(1, 2)
        for layer in self.query_conv1d_layers:
            inp_rep = layer(inp_rep)
        # batch_size x ?
        conv_queries = inp_rep.flatten(1)

        # batch_size x num_rel_docs x ?
        conv_queries = conv_queries.unsqueeze(1).expand(
            batch_size, num_docs, conv_queries.size(1))
        # batch_size * num_rel_docs x ?
        conv_queries = conv_queries.contiguous().view(batch_size * num_docs, -1)

        # embed documents
        doc_rep = batch_docs.view(batch_size * num_docs, dlen)
        embedded_docs = self.word_embeddings(doc_rep.unsqueeze(2))
        # batch_size * num_rel_docs x max_doc_len x emsize
        embedded_docs = self.emb_drop(embedded_docs)

        inp_rep = embedded_docs.transpose(1, 2)
        for layer in self.doc_conv1d_layers:
            inp_rep = layer(inp_rep)
        # batch_size * num_rel_docs x ?
        conv_docs = inp_rep.flatten(1)

        com_rep = torch.cat((conv_queries, conv_docs), 1)
        score = self.mlp(com_rep).squeeze(1)
        return score.view(batch_size, num_docs)
