import torch
import torch.nn as nn
import torch.nn.functional as f

from collections import OrderedDict
from .layers import Embedder, Encoder, Decoder
from neuroir.inputters import constants


class M_MATCH_TENSOR(nn.Module):
    def __init__(self, args):
        super(M_MATCH_TENSOR, self).__init__()

        self.embedder = Embedder(emsize=args.emsize,
                                 src_vocab_size=args.src_vocab_size,
                                 dropout_emb=args.dropout_emb)

        self.linear_projection = nn.Linear(self.embedder.output_size,
                                           args.featsize)

        self.query_encoder = Encoder(rnn_type=args.rnn_type,
                                     input_size=args.featsize,
                                     bidirection=args.bidirection,
                                     nlayers=args.nlayers,
                                     nhid=args.nhid_query,
                                     dropout_rnn=args.dropout_rnn)

        self.document_encoder = Encoder(rnn_type=args.rnn_type,
                                        input_size=args.featsize,
                                        bidirection=args.bidirection,
                                        nlayers=args.nlayers,
                                        nhid=args.nhid_document,
                                        dropout_rnn=args.dropout_rnn)

        self.query_projection = nn.Linear(args.nhid_query, args.nchannels)
        self.document_projection = nn.Linear(args.nhid_document, args.nchannels)

        self.exact_match_channel = ExactMatchChannel()
        self.conv1 = nn.Conv2d(args.nchannels + 1, args.nfilters, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(args.nchannels + 1, args.nfilters, (3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(args.nchannels + 1, args.nfilters, (3, 7), padding=(1, 3))
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(args.nfilters * 3, args.match_filter_size, (1, 1))
        self.output = nn.Linear(args.match_filter_size, 1)

        # session query encoder is unidirectional
        self.nhid_session = args.nhid_session
        self.session_query_encoder = Encoder(rnn_type=args.rnn_type,
                                             input_size=args.nchannels,
                                             bidirection=False,
                                             nlayers=args.nlayers,
                                             nhid=args.nhid_session,
                                             dropout_rnn=args.dropout_rnn)

        self.decoder = Decoder(rnn_type=args.rnn_type,
                               input_size=self.embedder.output_size,
                               bidirection=False,
                               nlayers=args.nlayers,
                               nhid=args.nhid_session,  # check hidsize
                               attn_type='none',
                               dropout_rnn=args.dropout_rnn,
                               copy_attn=False,
                               reuse_copy_attn=False)

        self.dropout = nn.Dropout(args.dropout)
        self.generator = nn.Linear(args.nhid_session, args.tgt_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.regularize_coeff = args.regularize_coeff

    def encode(self,
               source_rep,
               source_len):

        batch_size = source_rep.size(0)
        session_len = source_rep.size(1)

        source_rep = source_rep.view(batch_size * session_len, -1).contiguous()
        source_len = source_len.view(-1).contiguous()

        # -------- Query Encoding --------
        # (batch_size * session_len) x max_src_len x emsize
        source_word_rep = self.embedder(source_rep)
        # (batch_size * session_len) x max_src_len x featsize
        source_word_rep = self.linear_projection(source_word_rep)
        # (batch_size * session_len) x max_src_len x nhid_query
        _, encoded_queries = self.query_encoder(source_word_rep, source_len)
        encoded_queries = self.dropout(encoded_queries)
        # (batch_size * session_len) x max_src_len x nchannels
        projected_queries = self.query_projection(encoded_queries)

        # -------- Session Encoding and Ranking --------
        # (batch_size * session_len) x nchannels
        memory_bank = self.apply_pooling(projected_queries, pool_type='max')
        memory_bank = memory_bank.view(batch_size, session_len, -1).contiguous()

        # session level encoding
        hidden = None
        hidden_states, cell_states = [], []
        session_bank = []
        # loop over all the queries in session
        for qidx in range(session_len):
            # batch_size x nhid_query
            i_input = memory_bank[:, qidx, :]
            # hidden: (layers*directions) x batch x dim.
            hidden, session_rep = self.session_query_encoder(i_input.unsqueeze(1),
                                                             None,
                                                             init_states=hidden)
            session_bank.append(session_rep.squeeze(1))
            if isinstance(hidden, tuple):  # LSTM
                hidden_states.append(hidden[0])
                cell_states.append(hidden[1])
            else:  # GRU
                hidden_states.append(hidden[0])

        # batch_size x session_len x nhid_session
        session_bank = torch.stack(session_bank, dim=1)
        # states: (layers*directions) x (batch*session_len-1) x dim.
        if len(cell_states) != 0:
            hidden_states = torch.cat(hidden_states[:-1], dim=1)
            cell_states = torch.cat(cell_states[:-1], dim=1)
            states = (hidden_states, cell_states)
        else:
            states = torch.cat(hidden_states[:-1], dim=1)

        return projected_queries, session_bank, states

    def rank_document(self,
                      source_rep,
                      projected_queries,
                      session_bank,
                      document_rep,
                      document_len):

        batch_size = document_rep.size(0)
        session_len = document_rep.size(1)
        num_candidates = document_rep.size(2)
        max_doc_len = document_rep.size(3)
        max_query_len = projected_queries.size(1)

        source_rep = source_rep.view(batch_size * session_len, -1).contiguous()
        document_word_rep = document_rep.view(batch_size * session_len * num_candidates,
                                              -1).contiguous()
        document_len = document_len.view(-1).contiguous()

        # -------- Document Encoding --------
        document_word_rep = self.embedder(document_word_rep)
        # batch_size * session_len * num_candidates x max_doc_len x featsize
        document_word_rep = self.linear_projection(document_word_rep)
        # batch_size * session_len * num_candidates x max_doc_len x nhid_document
        _, encoded_docs = self.document_encoder(document_word_rep, document_len)
        encoded_queries = self.dropout(encoded_docs)
        # batch_size * session_len * num_candidates x max_doc_len x nchannels
        projected_docs = self.document_projection(encoded_queries)
        # (batch_size * session_len * num_candidates) x max_query_len x max_doc_len x nchannels
        projected_docs = torch.stack([projected_docs] * max_query_len, dim=1)
        # --------
        projected_queries = projected_queries.view(batch_size,
                                                   session_len,
                                                   max_query_len,
                                                   -1).contiguous()
        projected_queries = torch.stack([projected_queries] * num_candidates, dim=2)
        # (batch_size * session_len * num_candidates) x max_query_len x nhid_query
        projected_queries = projected_queries.view(batch_size * session_len * num_candidates,
                                                   max_query_len, -1).contiguous()
        # (batch_size * session_len * num_candidates) x max_query_len x max_doc_len x nchannels
        projected_queries = torch.stack([projected_queries] * max_doc_len, dim=2)
        # (batch_size * session_len * num_candidates) x max_query_len x max_doc_len x nchannels
        query_document_product = projected_queries * projected_docs

        # append exact match channel
        batch_docs_rep = document_rep.view(batch_size * session_len,
                                           num_candidates,
                                           -1).contiguous()
        exact_match = self.exact_match_channel(source_rep, batch_docs_rep).unsqueeze(3)
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
        scores = scores.view(batch_size, session_len, num_candidates).contiguous()
        return scores

    def forward(self,
                source_rep,
                source_len,
                target_rep,
                target_len,
                target_seq,
                document_rep,
                document_len,
                document_label):
        """
        Input:
            - source_rep: ``(batch_size, session_len, max_src_len)``
            - source_len: ``(batch_size, session_len)``
            - target_rep: ``(batch_size, session_len-1, max_tgt_len)``
            - target_len: ``(batch_size, session_len-1)``
            - target_seq: ``(batch_size, session_len-1, max_tgt_len)``
            - document_rep: ``(batch_size, session_len, num_candidates, max_doc_len)``
            - document_len: ``(batch_size, session_len, num_candidates)``
            - document_label: ``(batch_size, session_len, num_candidates)``
        Output:
            - loss: average loss over the batch elements
        """
        batch_size = source_rep.size(0)
        session_len = source_rep.size(1)
        num_candidates = document_rep.size(2)

        projected_queries, session_bank, states = self.encode(source_rep,
                                                              source_len)
        click_scores = self.rank_document(source_rep,
                                          projected_queries,
                                          session_bank,
                                          document_rep,
                                          document_len)
        click_loss = f.binary_cross_entropy_with_logits(click_scores, document_label)

        # ------- Decoding -------
        target_rep = target_rep.view(batch_size * (session_len - 1), -1).contiguous()
        target_seq = target_seq.view(batch_size * (session_len - 1), -1).contiguous()
        target_len = target_len.view(-1).contiguous()

        # (batch*session_len) x max_src_len x emsize
        target_word_rep = self.embedder(target_rep)

        init_decoder_state = self.decoder.init_decoder(states)
        decoder_outputs, _ = self.decoder(target_word_rep,
                                          None,
                                          None,
                                          init_decoder_state)

        target = target_seq[:, 1:].contiguous()
        scores = self.generator(decoder_outputs)  # `(batch*session_len) x max_tgt_len x vocab_size`
        scores = scores[:, :-1, :].contiguous()  # `(batch*session_len) x max_tgt_len - 1 x vocab_size`
        logll = self.log_softmax(scores)
        ml_loss = f.nll_loss(logll.view(-1, logll.size(2)),
                             target.view(-1),
                             reduce=False)

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(constants.PAD).float())
        decoding_loss = ml_loss.sum(1).mean()

        if self.regularize_coeff > 0:
            regularized_loss = logll.exp().mul(logll).sum(2) * self.regularize_coeff
            decoding_loss += regularized_loss.sum(1).mean()

        return {
            'ranking_loss': click_loss,
            'suggestion_loss': decoding_loss
        }

    @staticmethod
    def apply_pooling(encodings, pool_type):
        if pool_type == 'max':
            pooled_encodings = encodings.max(1)[0]
        elif pool_type == 'mean':
            pooled_encodings = encodings.mean(1)
        else:
            raise NotImplementedError

        return pooled_encodings

    def __tens2sent(self, t, tgt_dict, src_vocabs):
        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            elif src_vocabs:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
            else:
                raise NotImplementedError
        return words

    def decode(self,
               states,
               max_len,
               src_dict,
               tgt_dict,
               batch_size,
               session_len,
               use_cuda,
               **kwargs):

        # states: (layers*directions) x (batch*session_len-1) x dim.
        init_decoder_state = self.decoder.init_decoder(states)

        tgt = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt = tgt.cuda()
        tgt = tgt.expand(batch_size * session_len).unsqueeze(1)  # B x 1

        dec_preds = []
        for idx in range(max_len):
            # (batch*session_len) x 1 x emsize
            target_word_rep = self.embedder(tgt)

            decoder_outputs, _ = self.decoder(target_word_rep,
                                              None,
                                              None,
                                              init_decoder_state)

            prediction = self.generator(decoder_outputs.squeeze(1))
            prediction = f.softmax(prediction, dim=1)

            # (batch*session_len) x 1
            tgt = torch.max(prediction, dim=1, keepdim=True)[1]
            dec_preds.append(tgt.squeeze(1).clone())

            words = self.__tens2sent(tgt, tgt_dict, None)
            words = [src_dict[w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt = words.unsqueeze(1)

        # (batch*session_len) x max_len
        dec_preds = torch.stack(dec_preds, dim=1)
        dec_preds = dec_preds.view(batch_size, session_len, max_len).contiguous()

        return {
            'predictions': dec_preds
        }


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
