import torch
import torch.nn as nn
import torch.nn.functional as f
from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings


# verified from https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb
class DUET(nn.Module):
    """Learning to Match using Local and Distributed Representations of Text for Web Search."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(DUET, self).__init__()

        self.use_word = args.use_word
        if self.use_word:
            self.word_embeddings = Embeddings(args.emsize,
                                              args.src_vocab_size,
                                              PAD)
            self.emb_drop = nn.Dropout(p=args.dropout_emb)
        else:
            raise TypeError('Non-word inputs are not supported!')

        self.local_model = LocalModel(args)
        self.distributed_model = DistributedModel(args)

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

        local_score = self.local_model(batch_queries, batch_docs)
        # ----------Embed the questions and paragraphs from word---------- #
        if self.use_word:
            # batch_size x max_query_len x emsize
            embedded_queries = self.word_embeddings(batch_queries.unsqueeze(2))
            embedded_queries = self.emb_drop(embedded_queries)
            # batch_size x num_rel_docs x max_doc_len x emsize
            doc_rep = batch_docs.view(batch_size * num_docs, dlen).unsqueeze(2)
            embedded_docs = self.word_embeddings(doc_rep)
            embedded_docs = self.emb_drop(embedded_docs)
            embedded_docs = embedded_docs.view(batch_size, num_docs, dlen, -1)
        else:
            embedded_queries = batch_queries
            embedded_docs = batch_docs
        # --------------------------------------------------------------- #
        distributed_score = self.distributed_model(embedded_queries, embedded_docs)
        total_score = local_score + distributed_score
        return total_score


class LocalModel(nn.Module):
    """Implementation of the local model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(LocalModel, self).__init__()

        self.conv1d = nn.Conv1d(args.max_doc_len,
                                args.nfilters,
                                args.local_filter_size)
        self.drop = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.max_query_len, 1)
        self.fc2 = nn.Linear(args.nfilters, args.nfilters)
        self.fc3 = nn.Linear(args.nfilters, 1)

    def forward(self, batch_queries, batch_clicks):
        """
        Forward function of the local model.
        Parameters
        --------------------
            batch_queries   -- 2d tensor (batch_size, max_q_len)
            batch_clicks    -- 3d tensor (batch_size, num_rel_docs, max_doc_len)
        Returns
        --------------------
            score           -- 2d tensor (batch_size, num_rel_docs) local relevance score
        """
        batch_size, num_candidates = batch_clicks.size(0), batch_clicks.size(1)
        max_query_len = batch_queries.size(1)
        max_doc_len = batch_clicks.size(2)

        # batch_size x num_rel_docs x max_q_len
        extended_queries = batch_queries.unsqueeze(1).expand(batch_size,
                                                             num_candidates,
                                                             max_query_len)
        # batch_size x num_rel_docs x max_doc_len x max_q_len
        query_rep = extended_queries.unsqueeze(2).expand(batch_size,
                                                         num_candidates,
                                                         max_doc_len,
                                                         max_query_len)

        # batch_size x num_rel_docs x max_doc_len x max_q_len
        doc_rep = batch_clicks.unsqueeze(3).expand(batch_size,
                                                   num_candidates,
                                                   max_doc_len,
                                                   max_query_len)

        # ----------Create binary matrix based on unigram overlapping---------- #
        diff_matrix = doc_rep - query_rep
        bin_matrix = diff_matrix.clone()
        bin_matrix[diff_matrix == 0] = 1
        bin_matrix[diff_matrix != 0] = 0
        # (batch_size * num_rel_docs) x max_doc_len x max_q_len
        bin_matrix = bin_matrix.view(-1, max_doc_len, max_query_len).contiguous()
        # (batch_size * num_rel_docs) x nfilters x max_q_len
        conv_unigram = f.tanh(self.conv1d(bin_matrix.float()))

        mapped_feature1 = f.tanh(self.fc1(conv_unigram)).squeeze(2)
        mapped_feature2 = self.drop(f.tanh(self.fc2(mapped_feature1)))
        score = f.tanh(self.fc3(mapped_feature2)).view(batch_size, num_candidates)
        return score


class DistributedModel(nn.Module):
    """Implementation of the distributed model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(DistributedModel, self).__init__()

        self.conv_q = nn.Conv1d(args.emsize,
                                args.nfilters,
                                args.dist_filter_size)
        self.conv_d1 = nn.Conv1d(args.emsize,
                                 args.nfilters,
                                 args.dist_filter_size)
        self.conv_d2 = nn.Conv1d(args.nfilters,
                                 args.nfilters,
                                 1)

        self.pool_size = args.pool_size
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.nfilters, args.nfilters)
        self.fc2 = nn.Linear(args.max_doc_len - args.pool_size - 1, 1)
        self.fc3 = nn.Linear(args.nfilters, args.nfilters)
        self.fc4 = nn.Linear(args.nfilters, 1)

    def forward(self, embedded_q, embedded_d):
        """
        Forward function of neural ranker.
        Parameters
        --------------------
            embedded_q      -- 3d tensor (batch_size, max_q_len, emsize)
            embedded_d      -- 4d tensor (batch_size, num_rel_docs, max_doc_len, emsize)
        Returns
        --------------------
            score           -- 2d tensor (batch_size, num_rel_docs) distributed relevance score
        """
        batch_size = embedded_d.size(0)
        num_candidates = embedded_d.size(1)
        max_doc_len = embedded_d.size(2)
        # (batch_size * num_rel_docs) x max_doc_len x emsize
        embedded_d = embedded_d.view(batch_size * num_candidates, max_doc_len, -1)

        # ----------Apply convolution on question and paragraph embeddings---------- #
        # batch_size x num_filters x (max_q_len - filter_size + 1)
        conv_q = f.tanh(self.conv_q(embedded_q.transpose(1, 2)))
        # (batch_size * num_rel_docs) x num_filters x (max_doc_len - filter_size + 1)
        conv_p = f.tanh(self.conv_d1(embedded_d.transpose(1, 2)))

        # ----------Apply max-pooling on convolved question and document features---------- #
        # batch_size x num_filters
        max_pooled_q = f.max_pool1d(conv_q, conv_q.size(-1)).squeeze(2)
        # (batch_size * num_rel_docs) x num_filters x (max_doc_len - filter_size - pool_size + 2)
        max_pooled_d = f.max_pool1d(conv_p, self.pool_size, 1)

        # ----------Apply LT on query and convolution on paragraph representation---------- #
        # batch_size x num_filters
        query_rep = f.tanh(self.fc1(max_pooled_q))
        # (batch_size * num_rel_docs) x num_filters x (max_doc_len - filter_size - pool_size + 2)
        doc_rep = f.tanh(self.conv_d2(max_pooled_d))

        # ----------Apply hadamard (element-wise) product on question and document representation---------- #
        # (batch_size * num_rel_docs) x (max_doc_len - filter_size - pool_size + 2) x num_filters
        transposed_p = doc_rep.transpose(1, 2)
        # batch_size x (max_doc_len - filter_size - pool_size + 2) x num_filters
        transposed_q = query_rep.unsqueeze(1).expand(query_rep.size(0), *transposed_p.size()[1:])
        # batch_size x num_rel_docs x (max_doc_len - filter_size - pool_size + 2) x num_filters
        expanded_q = transposed_q.unsqueeze(1).expand(transposed_q.size(0), num_candidates, *transposed_q.size()[1:])
        # (batch_size * num_rel_docs) x (max_doc_len - filter_size - pool_size + 2) x num_filters
        mod_q = expanded_q.contiguous().view(-1, *expanded_q.size()[2:])
        # (batch_size * num_rel_docs) x (max_doc_len - filter_size - pool_size + 2) x num_filters
        hadamard = mod_q * transposed_p
        # (batch_size * num_rel_docs) x num_filters x (max_doc_len - filter_size - pool_size + 2)
        hadamard = hadamard.transpose(1, 2)

        # ----------Apply rest of the operation---------- #
        # (batch_size * num_rel_docs * num_filters)
        mapped_features1 = f.tanh(self.fc2(hadamard.contiguous().view(-1, hadamard.size(-1)).squeeze()))
        # (batch_size * num_rel_docs) x num_filters
        mapped_features1 = mapped_features1.view(*hadamard.size()[:-1])
        # (batch_size * num_rel_docs) x num_filters
        mapped_features2 = self.dropout(f.tanh(self.fc3(mapped_features1)))
        # (batch_size * num_rel_docs)
        score = f.tanh(self.fc4(mapped_features2).squeeze())
        # batch_size x num_rel_docs
        score = score.view(batch_size, num_candidates)
        return score
