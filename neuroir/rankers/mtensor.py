import torch
import torch.nn as nn

from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings
from neuroir.encoders import RNNEncoder


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()
        self.encoder = RNNEncoder(args.rnn_type,
                                  input_size,
                                  args.bidirection,
                                  args.nlayers,
                                  args.nhid,
                                  args.dropout_rnn)

    def forward(self, input, input_len):
        hidden, M = self.encoder(input, input_len)  # B x Seq-len x h
        return hidden, M


class MatchTensor(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(MatchTensor, self).__init__()

        self.word_embeddings = Embeddings(args.emsize,
                                          args.src_vocab_size,
                                          PAD)
        self.emb_drop = nn.Dropout(p=args.dropout_emb)

        self.linear_projection = nn.Linear(args.emsize, args.featsize)

        self.query_encoder = RNNEncoder(args.rnn_type,
                                        args.featsize,
                                        args.bidirection,
                                        args.nlayers,
                                        args.nhid_query,
                                        args.dropout_rnn)
        self.document_encoder = RNNEncoder(args.rnn_type,
                                           args.featsize,
                                           args.bidirection,
                                           args.nlayers,
                                           args.nhid_doc,
                                           args.dropout_rnn)

        self.query_projection = nn.Linear(args.nhid_query, args.nchannels)
        self.document_projection = nn.Linear(args.nhid_doc, args.nchannels)

        self.exact_match_channel = ExactMatchChannel()
        self.conv1 = nn.Conv2d(args.nchannels + 1, args.nfilters, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(args.nchannels + 1, args.nfilters, (3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(args.nchannels + 1, args.nfilters, (3, 7), padding=(1, 3))
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(args.nfilters * 3, args.match_filter_size, (1, 1))
        self.output = nn.Linear(args.match_filter_size, 1)

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

        # step1: apply embedding lookup
        embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
        # batch_size x max_q_len x emsize
        embedded_queries = self.emb_drop(embedded_queries)
        #
        doc_rep = batch_docs.view(batch_size * num_docs, dlen)
        embedded_docs = self.word_embeddings(doc_rep.unsqueeze(2))
        # batch_size * num_rel_docs x max_doc_len x emsize
        embedded_docs = self.emb_drop(embedded_docs)

        # step2: apply linear projection on embedded queries and documents
        # batch_size x max_q_len x featsize
        embedded_queries = self.linear_projection(embedded_queries)
        # batch_size * num_rel_docs x max_doc_len x featsize
        embedded_docs = self.linear_projection(embedded_docs)

        # step3: pass the encoded query and doc through an RNN
        _, encoded_queries = self.query_encoder(embedded_queries, query_len)
        _, encoded_docs = self.document_encoder(embedded_docs, doc_len.reshape(-1))

        # step4: apply linear projection on query hidden states

        # batch_size x max_q_len x nchannels
        projected_queries = self.query_projection(encoded_queries)
        projected_queries = torch.stack([projected_queries] * num_docs, dim=1)
        # batch_size * num_rel_docs x max_q_len x nchannels
        projected_queries = projected_queries.contiguous().view(batch_size * num_docs, qlen, -1)

        # batch_size * num_rel_docs x max_q_len x max_doc_len x nchannels
        projected_queries = torch.stack([projected_queries] * dlen, dim=2)

        # batch_size * num_rel_docs x max_doc_len x nchannels
        projected_docs = self.document_projection(encoded_docs)
        # batch_size * num_rel_docs x max_q_len x max_doc_len x nchannels
        projected_docs = torch.stack([projected_docs] * qlen, dim=1)

        # step5: 2d product between projected query and doc vectors
        # batch_size * num_rel_docs x max_q_len x max_doc_len x nchannels
        query_document_product = projected_queries * projected_docs

        # step6: append exact match channel
        exact_match = self.exact_match_channel(batch_queries, batch_docs).unsqueeze(3)
        query_document_product = torch.cat((query_document_product, exact_match), 3)
        query_document_product = query_document_product.transpose(2, 3).transpose(1, 2)

        # step7: run the convolutional operation, max-pooling and linear projection
        convoluted_feat1 = self.conv1(query_document_product)
        convoluted_feat2 = self.conv2(query_document_product)
        convoluted_feat3 = self.conv3(query_document_product)
        convoluted_feat = self.relu(torch.cat((convoluted_feat1, convoluted_feat2, convoluted_feat3), 1))
        convoluted_feat = self.conv(convoluted_feat).transpose(1, 2).transpose(2, 3)

        max_pooled_feat = torch.max(convoluted_feat, 2)[0]
        max_pooled_feat = torch.max(max_pooled_feat, 1)[0]
        scores = self.output(max_pooled_feat).squeeze(-1)
        return scores.view(batch_size, num_docs)


class ExactMatchChannel(nn.Module):
    """Exact match channel layer for the match tensor"""

    def __init__(self):
        """"Constructor of the class"""
        super(ExactMatchChannel, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        # Initializing the value of alpha
        torch.nn.init.uniform_(self.alpha)

    def forward(self, batch_query, batch_docs):
        """"Computes the exact match channel"""
        query_tensor = batch_query.unsqueeze(1).expand(batch_query.size(0),
                                                       batch_docs.size(1),
                                                       batch_query.size(1))
        query_tensor = query_tensor.contiguous().view(-1, query_tensor.size(2))
        doc_tensor = batch_docs.view(-1, batch_docs.size(2))

        query_tensor = query_tensor.unsqueeze(2).expand(*query_tensor.size(), batch_docs.size(2))
        doc_tensor = doc_tensor.unsqueeze(1).expand(doc_tensor.size(0),
                                                    batch_query.size(1),
                                                    doc_tensor.size(1))

        exact_match = (query_tensor == doc_tensor).float()
        return exact_match * self.alpha.expand(exact_match.size())
