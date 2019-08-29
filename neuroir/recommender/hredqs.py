import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import Embedder, Encoder, Decoder
from neuroir.inputters import constants


class HredQS(nn.Module):
    def __init__(self, args):
        super(HredQS, self).__init__()

        self.embedder = Embedder(emsize=args.emsize,
                                 src_vocab_size=args.src_vocab_size,
                                 dropout_emb=args.dropout_emb)

        self.encoder = Encoder(rnn_type=args.rnn_type,
                               input_size=self.embedder.output_size,
                               bidirection=args.bidirection,
                               nlayers=args.nlayers,
                               nhid=args.nhid,
                               dropout_rnn=args.dropout_rnn)

        # session encoder is unidirectional
        self.session_encoder = Encoder(rnn_type=args.rnn_type,
                                       input_size=args.nhid,
                                       bidirection=False,
                                       nlayers=args.nlayers,
                                       nhid=args.nhid_session,
                                       dropout_rnn=args.dropout_rnn)

        self.decoder = Decoder(rnn_type=args.rnn_type,
                               input_size=self.embedder.output_size,
                               bidirection=args.bidirection,
                               nlayers=args.nlayers,
                               nhid=args.nhid_session,  # check hidsize
                               attn_type='none',
                               dropout_rnn=args.dropout_rnn,
                               copy_attn=False,
                               reuse_copy_attn=False)

        self.dropout = nn.Dropout(args.dropout)
        self.generator = nn.Linear(args.nhid_session, args.tgt_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode(self,
               source_rep,
               source_len,
               batch_size,
               session_len):

        # batch_size x max_src_len x emsize
        source_word_rep = self.embedder(source_rep)

        # memory_bank: B x P x h; hidden: l*num_directions x B x h
        _, memory_bank = self.encoder(source_word_rep, source_len)
        memory_bank = self.dropout(memory_bank)

        # apply max-pooling
        memory_bank = self.apply_pooling(memory_bank, pool_type='max')
        # batch_size x session_len x nhid
        memory_bank = memory_bank.view(batch_size, session_len, -1).contiguous()

        # session level encoding
        hidden = None
        hidden_states, cell_states = [], []
        for sidx in range(memory_bank.size(1)):
            i_input = memory_bank[:, sidx, :].unsqueeze(1)
            # hidden: (layers*directions) x batch x dim.
            hidden, session_bank = self.session_encoder(i_input,
                                                        None,
                                                        init_states=hidden)
            if isinstance(hidden, tuple):  # LSTM
                hidden_states.append(hidden[0])
                cell_states.append(hidden[1])
            else:  # GRU
                hidden_states.append(hidden[0])

        # (layers*directions) x (batch*session_len) x dim.
        if len(cell_states) != 0:
            hidden_states = torch.cat(hidden_states, dim=1)
            cell_states = torch.cat(cell_states, dim=1)
            states = (hidden_states, cell_states)
        else:
            states = torch.cat(hidden_states, dim=1)

        return states

    def forward(self,
                source_rep,
                source_len,
                target_rep,
                target_len,
                target_seq,
                source_map,
                alignment):
        """
        Input:
            - source_rep: ``(batch_size, session_len-1, max_src_len)``
            - source_len: ``(batch_size, session_len-1)``
            - target_rep: ``(batch_size, session_len-1, max_tgt_len)``
            - target_len: ``(batch_size, session_len-1)``
            - target_seq: ``(batch_size, session_len-1, max_tgt_len)``
        Output:
            - loss: average loss over the batch elements
        """
        batch_size = source_rep.size(0)
        session_len = source_rep.size(1)

        source_rep = source_rep.view(batch_size * session_len, -1).contiguous()
        target_rep = target_rep.view(batch_size * session_len, -1).contiguous()
        target_seq = target_seq.view(batch_size * session_len, -1).contiguous()
        source_len = source_len.view(-1).contiguous()
        target_len = target_len.view(-1).contiguous()

        states = self.encode(source_rep,
                             source_len,
                             batch_size,
                             session_len)

        # ------- Decoding -------

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
        ml_loss = ml_loss.sum(1).mean()
        return ml_loss

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
               source_rep,
               source_len,
               max_len,
               src_dict,
               tgt_dict,
               src_map,
               alignment,
               blank,
               fill,
               source_vocabs):

        batch_size = source_rep.size(0)
        session_len = source_rep.size(1)
        use_cuda = source_rep.is_cuda

        source_rep = source_rep.view(batch_size * session_len, -1).contiguous()
        source_len = source_len.view(-1).contiguous()

        states = self.encode(source_rep,
                             source_len,
                             batch_size,
                             session_len)

        # ------- Decoding -------

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
