# Includes a simplified implementation of https://arxiv.org/abs/1708.03418

import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import Embedder, Encoder, Decoder

from neuroir.inputters import constants
from neuroir.modules.copy_generator import CopyGenerator, CopyGeneratorCriterion


class Seq2seq(nn.Module):
    def __init__(self, args):
        super(Seq2seq, self).__init__()

        self.embedder = Embedder(emsize=args.emsize,
                                 src_vocab_size=args.src_vocab_size,
                                 dropout_emb=args.dropout_emb)

        self.encoder = Encoder(rnn_type=args.rnn_type,
                               input_size=self.embedder.output_size,
                               bidirection=args.bidirection,
                               nlayers=args.nlayers,
                               nhid=args.nhid,
                               dropout_rnn=args.dropout_rnn)

        self.decoder = Decoder(rnn_type=args.rnn_type,
                               input_size=self.embedder.output_size,
                               bidirection=args.bidirection,
                               nlayers=args.nlayers,
                               nhid=args.nhid,
                               attn_type=args.attn_type,
                               dropout_rnn=args.dropout_rnn,
                               copy_attn=args.copy_attn,
                               reuse_copy_attn=args.reuse_copy_attn)

        self.dropout = nn.Dropout(args.dropout)
        self.generator = nn.Linear(args.nhid, args.tgt_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.copy_attn = args.copy_attn
        if self.copy_attn:
            self.copy_generator = CopyGenerator(args.nhid,
                                                self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=args.tgt_vocab_size,
                                                    force_copy=args.force_copy)

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
            - source_rep: ``(batch_size, max_src_len)``
            - source_len: ``(batch_size)``
            - target_rep: ``(batch_size, max_tgt_len)``
            - target_len: ``(batch_size)``
            - target_seq: ``(batch_size, max_tgt_len)``
        Output:
            - loss: tensor with a single value
        """

        # batch_size x max_src_len x emsize
        source_word_rep = self.embedder(source_rep)

        # memory_bank: B x P x h; hidden: l*num_directions x B x h
        hidden, memory_bank = self.encoder(source_word_rep, source_len)
        memory_bank = self.dropout(memory_bank)

        # batch_size x max_src_len x emsize
        target_word_rep = self.embedder(target_rep)

        init_decoder_state = self.decoder.init_decoder(hidden)
        decoder_outputs, attns = self.decoder(target_word_rep,
                                              memory_bank,
                                              source_len,
                                              init_decoder_state)

        target = target_seq[:, 1:].contiguous()
        if self.copy_attn:
            scores = self.copy_generator(decoder_outputs,
                                         attns["copy"],
                                         source_map)
            scores = scores[:, :-1, :].contiguous()
            ml_loss = self.criterion(scores,
                                     alignment[:, 1:].contiguous(),
                                     target)
        else:
            scores = self.generator(decoder_outputs)  # `batch x max_tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x max_tgt_len - 1 x vocab_size`
            logll = self.log_softmax(scores)
            ml_loss = f.nll_loss(logll.view(-1, logll.size(2)),
                                 target.view(-1),
                                 reduce=False)

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(constants.PAD).float())
        ml_loss = ml_loss.sum(1).mean()
        return ml_loss

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
        use_cuda = source_rep.is_cuda

        # batch_size x max_src_len x emsize
        source_word_rep = self.embedder(source_rep)

        # memory_bank: B x P x h; hidden: l*num_directions x B x h
        hidden, memory_bank = self.encoder(source_word_rep, source_len)
        memory_bank = self.dropout(memory_bank)

        init_decoder_state = self.decoder.init_decoder(hidden)

        tgt = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt = tgt.cuda()
        tgt = tgt.expand(batch_size).unsqueeze(1)  # B x 1

        dec_preds, attentions = [], []
        for idx in range(max_len):
            target_word_rep = self.embedder(tgt)

            # decoder_outputs = batch_size x 1 x tgt_dict_size
            decoder_outputs, attns = self.decoder(target_word_rep,
                                                  memory_bank,
                                                  source_len,
                                                  init_decoder_state)

            if self.copy_attn:
                prediction = self.copy_generator(decoder_outputs,
                                                 attns["copy"],
                                                 src_map)
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if blank[b]:
                        blank_b = torch.LongTensor(blank[b])
                        fill_b = torch.LongTensor(fill[b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = f.softmax(prediction, dim=1)

            tgt = torch.max(prediction, dim=1, keepdim=True)[1]
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                attentions.append(attns["std"].squeeze(1))

            words = self.__tens2sent(tgt, tgt_dict, source_vocabs)
            words = [src_dict[w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt = words.unsqueeze(1)

        # batch_size x max_len
        dec_preds = torch.stack(dec_preds, dim=1)
        # batch_size x max_len x source_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'attentions': attentions
        }
