import torch
import torch.nn as nn

from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


class ARCII(nn.Module):
    """Implementation of the convolutional matching model (ARC-II)."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(ARCII, self).__init__()

        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)
        self.emb_drop = nn.Dropout(p=args.dropout_emb)
        self.conv_query = nn.Conv1d(args.emsize,
                                    args.filters_1d,
                                    args.kernel_size_1d,
                                    padding=args.kernel_size_1d // 2)
        self.conv_doc = nn.Conv1d(args.emsize,
                                  args.filters_1d,
                                  args.kernel_size_1d,
                                  padding=args.kernel_size_1d // 2)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        num_conv2d_layers = len(args.kernel_size_2d)
        assert num_conv2d_layers == len(args.maxpool_size_2d)

        doc_feats = args.max_doc_len // 2
        query_feats = args.max_query_len // 2

        conv2d_layers = []
        for i in range(num_conv2d_layers):
            inpsize = args.filters_1d if i == 0 else args.filters_2d[i - 1]
            layer = nn.Sequential(
                nn.Conv2d(inpsize, args.filters_2d[i], args.kernel_size_2d[i],
                          padding=(args.kernel_size_2d[i][0] // 2, args.kernel_size_2d[i][1] // 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((args.maxpool_size_2d[i][0], args.maxpool_size_2d[i][1]))
            )
            conv2d_layers.append(layer)

            doc_feats = doc_feats // args.maxpool_size_2d[i][0]
            query_feats = query_feats // args.maxpool_size_2d[i][1]
            assert query_feats != 0 and doc_feats != 0

        self.conv2d_layers = nn.ModuleList(conv2d_layers)

        inpsize = args.filters_2d[-1] * query_feats * doc_feats
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
        :return: score representing click probability [batch_size x num_clicks_per_query]
        """
        assert batch_queries.shape[0] == batch_docs.shape[0]
        batch_size = batch_queries.shape[0]
        qlen = batch_queries.shape[1]
        num_docs, dlen = batch_docs.shape[1], batch_docs.shape[2]

        # embed query
        embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
        # batch_size x max_q_len x emsize
        embedded_queries = self.emb_drop(embedded_queries)
        # batch_size x nfilters x max_q_len
        embedded_queries = self.conv_query(embedded_queries.transpose(1, 2))

        # batch_size x num_rel_docs x nfilters x max_q_len
        embedded_queries = embedded_queries.unsqueeze(1).expand(batch_size, num_docs,
                                                                embedded_queries.size(1),
                                                                embedded_queries.size(2))
        # batch_size * num_rel_docs x nfilters x max_q_len
        embedded_queries = embedded_queries.contiguous().view(batch_size * num_docs,
                                                              embedded_queries.size(2),
                                                              embedded_queries.size(3))

        # embed documents
        doc_rep = batch_docs.view(batch_size * num_docs, dlen)
        embedded_docs = self.word_embeddings(doc_rep.unsqueeze(2))
        # batch_size * num_rel_docs x max_doc_len x emsize
        embedded_docs = self.emb_drop(embedded_docs)
        # batch_size * num_rel_docs x nfilters x max_doc_len
        embedded_docs = self.conv_doc(embedded_docs.transpose(1, 2))

        # batch_size * num_rel_docs x nfilters x max_doc_len x max_q_len
        embedded_queries = torch.stack([embedded_queries] * dlen, dim=2)
        # batch_size * num_rel_docs x nfilters x max_doc_len x max_q_len
        embedded_docs = torch.stack([embedded_docs] * qlen, dim=3)

        # batch_size * num_rel_docs x nfilters x max_doc_len x max_q_len
        comb_rep = embedded_queries + embedded_docs
        # batch_size * num_rel_docs x nfilters x max_doc_len/2 x max_q_len/2
        comb_rep = self.maxpool1(comb_rep)

        for layer in self.conv2d_layers:
            comb_rep = layer(comb_rep)
        comb_rep = comb_rep.flatten(1)

        score = self.mlp(comb_rep).squeeze(1)
        return score.view(batch_size, num_docs)
