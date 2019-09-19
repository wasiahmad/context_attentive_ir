import torch
import torch.nn as nn
import torch.nn.functional as f

from collections import OrderedDict
from .layers import Embedder, Encoder, Decoder
from neuroir.inputters import constants


class MNSRF(nn.Module):
    def __init__(self, args):
        super(MNSRF, self).__init__()

        self.embedder = Embedder(emsize=args.emsize,
                                 src_vocab_size=args.src_vocab_size,
                                 dropout_emb=args.dropout_emb)

        self.query_encoder = Encoder(rnn_type=args.rnn_type,
                                     input_size=self.embedder.output_size,
                                     bidirection=args.bidirection,
                                     nlayers=args.nlayers,
                                     nhid=args.nhid_query,
                                     dropout_rnn=args.dropout_rnn)

        self.document_encoder = Encoder(rnn_type=args.rnn_type,
                                        input_size=self.embedder.output_size,
                                        bidirection=args.bidirection,
                                        nlayers=args.nlayers,
                                        nhid=args.nhid_document,
                                        dropout_rnn=args.dropout_rnn)

        # session query encoder is unidirectional
        self.nhid_session = args.nhid_session
        self.session_query_encoder = Encoder(rnn_type=args.rnn_type,
                                             input_size=args.nhid_query,
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

        self.projection = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(args.nhid_query + args.nhid_session,
                                 args.nhid_document)),
            ('tanh', nn.Tanh())
        ]))

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
        # (batch_size * session_len) x max_src_len x nhid_query
        _, memory_bank = self.query_encoder(source_word_rep, source_len)
        memory_bank = self.dropout(memory_bank)
        # apply max-pooling
        memory_bank = self.apply_pooling(memory_bank, pool_type='max')
        # batch_size x session_len x nhid_query
        memory_bank = memory_bank.view(batch_size, session_len, -1).contiguous()

        # -------- Session Encoding and Ranking --------

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

        return memory_bank, session_bank, states

    def rank_document(self,
                      source_rep,
                      memory_bank,
                      session_bank,
                      document_rep,
                      document_len):

        batch_size = memory_bank.size(0)
        session_len = memory_bank.size(1)
        num_candidates = document_rep.size(2)

        document_rep = document_rep.view(batch_size * session_len * num_candidates, -1).contiguous()
        document_len = document_len.view(-1).contiguous()

        # -------- Document Encoding --------
        document_word_rep = self.embedder(document_rep)
        _, encoded_docs = self.document_encoder(document_word_rep, document_len)
        encoded_docs = self.apply_pooling(encoded_docs, pool_type='max')
        # batch_size x session_length x num_candidates x nhid_doc
        encoded_docs = encoded_docs.view(batch_size, session_len, num_candidates, -1).contiguous()
        # --------

        ranking_scores = []
        init_session_rep = torch.zeros(batch_size, self.nhid_session)
        if memory_bank.is_cuda:
            init_session_rep = init_session_rep.cuda()
        # loop over all the queries in session
        for qidx in range(session_len):
            # batch_size x nhid_query
            i_input = memory_bank[:, qidx, :]
            # batch_size x (nhid_query + nhid_session)
            if qidx == 0:
                combined_rep = torch.cat((i_input, init_session_rep), 1)
            else:
                combined_rep = torch.cat((i_input, session_bank[:, qidx, :]), 1)
            # batch_size x nhid_document
            combined_rep = self.projection(combined_rep)
            # batch_size x num_candidates x nhid_doc
            combined_rep = combined_rep.unsqueeze(1). \
                expand(batch_size, num_candidates, combined_rep.size(1))
            # batch_size x num_candidates
            click_score = torch.sum(torch.mul(combined_rep, encoded_docs[:, qidx, :, :]), 2)
            ranking_scores.append(click_score)

        # batch_size x session_len x num_candidates
        ranking_scores = torch.stack(ranking_scores, dim=1)
        return ranking_scores

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

        memory_bank, session_bank, states = self.encode(source_rep,
                                                        source_len)
        click_scores = self.rank_document(source_rep,
                                          memory_bank,
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
