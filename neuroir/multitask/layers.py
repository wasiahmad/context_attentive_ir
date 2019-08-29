import torch
import torch.nn as nn

from neuroir.inputters import BOS, PAD
from neuroir.modules.embeddings import Embeddings
from neuroir.encoders.rnn_encoder import RNNEncoder
from neuroir.decoders.rnn_decoder import RNNDecoder


class Embedder(nn.Module):
    def __init__(self,
                 emsize,
                 src_vocab_size,
                 dropout_emb):
        super(Embedder, self).__init__()

        self.word_embeddings = Embeddings(emsize,
                                          src_vocab_size,
                                          PAD)
        self.output_size = emsize
        self.dropout = nn.Dropout(dropout_emb)

    def forward(self,
                sequence):
        word_rep = self.word_embeddings(sequence.unsqueeze(2))  # B x P x d
        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 bidirection,
                 nlayers,
                 nhid,
                 dropout_rnn):
        super(Encoder, self).__init__()

        self.encoder = RNNEncoder(rnn_type,
                                  input_size,
                                  bidirection,
                                  nlayers,
                                  nhid,
                                  dropout_rnn)

    def forward(self,
                input,
                input_len,
                init_states=None):
        hidden, M = self.encoder(input,
                                 input_len,
                                 init_states)  # B x Seq-len x h
        return hidden, M


class Decoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 bidirection,
                 nlayers,
                 nhid,
                 attn_type,
                 dropout_rnn,
                 copy_attn,
                 reuse_copy_attn):
        super(Decoder, self).__init__()

        attn_type = None if attn_type == 'none' else attn_type
        self.decoder = RNNDecoder(rnn_type,
                                  input_size,
                                  bidirection,
                                  nlayers,
                                  nhid,
                                  attn_type=attn_type,
                                  dropout=dropout_rnn,
                                  copy_attn=copy_attn,
                                  reuse_copy_attn=reuse_copy_attn)

    def init_decoder(self, hidden):
        return self.decoder.init_decoder_state(hidden)

    def forward(self,
                tgt,
                memory_bank,
                memory_len,
                state):
        decoder_outputs, _, attns = self.decoder(tgt,
                                                 memory_bank,
                                                 state,
                                                 memory_lengths=memory_len)
        return decoder_outputs, attns
