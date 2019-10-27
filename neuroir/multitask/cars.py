import numpy
import torch
import torch.nn as nn
import torch.nn.functional as f

from collections import OrderedDict
from .layers import Embedder, Encoder, Decoder
from neuroir.inputters import constants
from neuroir.modules import Maxout
from neuroir.utils.misc import sequence_mask


class CARS(nn.Module):
    def __init__(self, args):
        super(CARS, self).__init__()

        self.embedder = Embedder(emsize=args.emsize,
                                 src_vocab_size=args.src_vocab_size,
                                 dropout_emb=args.dropout_emb)

        self.query_encoder = Encoder(rnn_type=args.rnn_type,
                                     input_size=self.embedder.output_size,
                                     bidirection=args.bidirection,
                                     nlayers=args.nlayers,
                                     nhid=args.nhid_query,
                                     dropout_rnn=args.dropout_rnn)

        # the following executes if at least the ranker is turned ON
        # OR, the document session encoding is enabled
        if not (args.turn_ranker_off and args.doc_session_off):
            self.document_encoder = Encoder(rnn_type=args.rnn_type,
                                            input_size=self.embedder.output_size,
                                            bidirection=args.bidirection,
                                            nlayers=args.nlayers,
                                            nhid=args.nhid_document,
                                            dropout_rnn=args.dropout_rnn)

        # NOTE: Pooling layers to form a vector from contextualized vectors
        if args.pool_type == 'attn':
            # weighted pooling parameters to form vectors from RNN states
            self.q_attn = nn.Sequential(
                nn.Linear(args.nhid_query, args.nhid_query),
                nn.Tanh(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.nhid_query, 1)
            )

            if not (args.turn_ranker_off and args.doc_session_off):
                # weighted pooling parameters to form vectors from RNN states
                self.d_attn = nn.Sequential(
                    nn.Linear(args.nhid_document, args.nhid_document),
                    nn.Tanh(),
                    nn.Dropout(p=args.dropout),
                    nn.Linear(args.nhid_document, 1)
                )

        session_rep_size = 0
        if not args.query_session_off:
            # query-level session encoder is a unidirectional RNN
            self.nhid_session_query = args.nhid_session_query
            self.session_query_encoder = Encoder(rnn_type=args.rnn_type,
                                                 input_size=args.nhid_query,
                                                 bidirection=False,
                                                 nlayers=args.nlayers,
                                                 nhid=args.nhid_session_query,
                                                 dropout_rnn=args.dropout_rnn)

            # apply weighted pooling over the session states
            self.session_query_attn = nn.Linear(args.nhid_session_query,
                                                args.nhid_query)
            self.session_query_inner_attn = nn.Sequential(
                nn.Linear(args.nhid_session_query, args.nhid_session_query),
                nn.Tanh(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.nhid_session_query, 1)
            )
            session_rep_size += args.nhid_session_query

        # if either ranker or the doc_session is on (ex., for suggestion task)
        if not args.doc_session_off:
            # click attn layer to combine all clicked documents
            self.click_attn = nn.Sequential(
                nn.Linear(args.nhid_document, args.nhid_document),
                nn.Tanh(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.nhid_document, 1)
            )

            # doc-level session encoder is a unidirectional RNN
            self.nhid_session_document = args.nhid_session_document
            self.session_doc_encoder = Encoder(rnn_type=args.rnn_type,
                                               input_size=args.nhid_document,
                                               bidirection=False,
                                               nlayers=args.nlayers,
                                               nhid=args.nhid_session_document,
                                               dropout_rnn=args.dropout_rnn)
            # weighted pooling parameters to form vectors from RNN states
            self.session_doc_attn = nn.Linear(args.nhid_session_document,
                                              args.nhid_document)
            self.session_doc_inner_attn = nn.Sequential(
                nn.Linear(args.nhid_session_document, args.nhid_session_document),
                nn.Tanh(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.nhid_session_document, 1)
            )
            session_rep_size += args.nhid_session_document

        # transform query+doc session representation to match document representation size for ranking
        if session_rep_size > 0:
            self.shared_session_projector = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=args.dropout)),
                ('linear', nn.Linear(session_rep_size, args.nhid_document, bias=False))
            ]))

        if not args.turn_ranker_off:
            # transform query representation to match document representation size for ranking
            self.q_projection = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=args.dropout)),
                ('linear', nn.Linear(args.nhid_query, args.nhid_document))
            ]))
            if session_rep_size > 0:
                self.private_session_projector1 = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(p=args.dropout)),
                    ('linear', nn.Linear(session_rep_size, args.nhid_document, bias=False))
                ]))

            # maxout network to compute relevance score for each candidate document
            self.ranknet = Maxout(input_dim=args.nhid_document * 4,
                                  num_layers=3,
                                  output_dims=[256, 128, 1],
                                  pool_sizes=[2, 2, 2])

        if not args.turn_recommender_off:
            if session_rep_size > 0:
                self.private_session_projector2 = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(p=args.dropout)),
                    ('linear', nn.Linear(session_rep_size, args.nhid_document, bias=False))
                ]))
            else:
                # Both session-level RNNs are off, raise error
                raise ValueError('Both session-level RNNs cannot be off!')

            # NOTE: We need to transform the session hidden states or encoder states to
            # initialize the decoder in the Seq2seq architecture.
            self.transform_hid = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=args.dropout)),
                ('linear', nn.Linear(session_rep_size, args.nhid_decoder))
            ]))
            if args.rnn_type == 'LSTM':
                self.transform_cell = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(p=args.dropout)),
                    ('linear', nn.Linear(session_rep_size, args.nhid_decoder))
                ]))

            self.decoder = Decoder(rnn_type=args.rnn_type,
                                   input_size=self.embedder.output_size,
                                   bidirection=False,
                                   nlayers=1,
                                   nhid=args.nhid_decoder,  # check hidsize
                                   attn_type=args.attn_type,
                                   dropout_rnn=args.dropout_rnn,
                                   copy_attn=False,
                                   reuse_copy_attn=False)
            self.dec_attn = nn.Linear(args.nhid_query, args.nhid_decoder,
                                      bias=False)

            input_size = args.nhid_decoder

            predictor_input_size = self.embedder.output_size
            if not (args.turn_ranker_off and args.doc_session_off):
                predictor_input_size = args.nhid_document

            self.token_prob_predictor1 = nn.Linear(input_size,
                                                   predictor_input_size,
                                                   bias=False)
            self.token_prob_predictor2 = nn.Linear(predictor_input_size,
                                                   args.tgt_vocab_size,
                                                   bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.regularize_coeff = args.regularize_coeff

        # book-keeping
        self.no_ranker = args.turn_ranker_off
        self.no_recommender = args.turn_recommender_off
        self.no_query_session_encoding = args.query_session_off
        self.no_document_session_encoding = args.doc_session_off
        self.pool_type = args.pool_type
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

    def encode(self, queries, query_length):
        """
        Encode the queries into matrices/vectors.
        Parameters
        --------------------
            queries         -- 3d tensor (batch_size,session_length,max_query_length)
            query_length    -- 2d tensor (batch_size,session_length)
        Returns
        --------------------
            pooled_rep      -- 3d tensor (batch_size,session_length,nhid_query * num_directions)
            encoded_rep     -- 3d tensor (batch_size * session_length,max_query_length,nhid_query * num_directions)
            embedded_rep    -- 2d tensor (batch_size * session_length,emb_size)
        """
        batch_size = queries.size(0)
        session_len = queries.size(1)
        max_query_len = queries.size(2)

        source_rep = queries.view(batch_size * session_len, -1).contiguous()
        source_len = query_length.view(-1).contiguous()

        # (batch_size * session_len) x max_src_len x emsize
        source_word_rep = self.embedder(source_rep)
        # (batch_size * session_len) x max_src_len x nhid_query
        hidden, encoded_queries = self.query_encoder(source_word_rep, source_len)
        encoded_queries = self.dropout(encoded_queries)

        query_mask = sequence_mask(source_len, max_len=max_query_len)
        # pooled_queries: (batch_size * session_length) x nhid_query
        pooled_queries = self.apply_pooling(encoded_queries, self.pool_type, dtype='query', mask=query_mask)
        # batch_size x session_len x nhid_query
        pooled_queries = pooled_queries.view(batch_size, session_len, -1).contiguous()

        return pooled_queries, encoded_queries, hidden

    def encode_document(self, docs, docs_length):
        """
        Encode the documents into vectors.
        Parameters
        --------------------
            docs           -- 4d tensor (batch_size,session_length,num_candidates,max_doc_length)
            docs_length    -- 3d tensor (batch_size,session_length,num_candidates)
        Returns
        --------------------
            encoded_docs   -- 4d tensor (batch_size,session_length,num_candidates,nhid_document)
        """
        batch_size = docs.size(0)
        session_len = docs.size(1)
        num_candidates = docs.size(2)
        max_doc_len = docs.size(3)

        source_rep = docs.view(batch_size * session_len * num_candidates, -1).contiguous()
        source_len = docs_length.view(-1).contiguous()

        # (batch_size * session_len * num_candidates) x max_src_len x emsize
        source_word_rep = self.embedder(source_rep)
        # (batch_size * session_len * num_candidates) x max_src_len x nhid_document
        hidden, encoded_docs = self.document_encoder(source_word_rep, source_len)
        encoded_docs = self.dropout(encoded_docs)

        document_mask = sequence_mask(source_len, max_len=max_doc_len)
        # pooled_docs: (batch_size * session_len * num_candidates) x nhid_document
        pooled_docs = self.apply_pooling(encoded_docs,
                                         self.pool_type,
                                         dtype='document',
                                         mask=document_mask)
        # encoded_docs: batch_size x session_length x num_candidates x nhid_document
        pooled_docs = pooled_docs.view(batch_size, session_len, num_candidates, -1).contiguous()
        return pooled_docs

    def encode_clicks(self, docs, doc_labels):
        """
        Encode all the clicked documents of queries into vectors.
        Parameters
        --------------------
            docs            -- 4d tensor (batch_size,session_length,num_candidates,nhid_document)
            doc_labels      -- 3d tensor (batch_size,session_length,num_candidates)
        Returns
        --------------------
            encoded_clicks  -- 3d tensor (batch_size,session_length,nhid_document)
        """
        batch_size = docs.size(0)
        session_len = docs.size(1)
        num_candidates = docs.size(2)
        use_cuda = docs.is_cuda

        sorted_index = doc_labels.sort(2, descending=True)[1]
        sorted_docs = [torch.index_select(docs[i, j], 0, sorted_index[i, j])
                       for i in range(batch_size)
                       for j in range(session_len)]
        # batch_size*session_len x num_candidates x nhid_document
        sorted_docs = torch.stack(sorted_docs, 0)

        click_length = numpy.count_nonzero(doc_labels.view(batch_size * session_len,
                                                           -1).cpu().numpy(), axis=1)
        click_length = torch.from_numpy(click_length)
        # (batch_size*session_len) x max_n_click
        click_length = sequence_mask(click_length)  # B*s_len x max_n_click

        click_mask = torch.ones(*sorted_docs.size()[:-1]).byte()
        click_mask[:, :click_length.size(1)] = click_length
        if use_cuda:
            click_mask = click_mask.cuda()

        att_weights = self.click_attn(sorted_docs.view(-1, sorted_docs.size(2))).squeeze(1)
        att_weights = att_weights.view(*sorted_docs.size()[:-1])
        att_weights.masked_fill_(1 - click_mask, -float('inf'))
        att_weights = f.softmax(att_weights, 1)
        encoded_clicks = torch.bmm(sorted_docs.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)

        # encoded_clicks: batch_size x session_length x (nhid_doc * num_directions)
        encoded_clicks = encoded_clicks.contiguous().view(batch_size, session_len, -1)
        return encoded_clicks

    def encode_session(self, encoded_queries, encoded_docs, encoded_clicks):
        """
        Encode all the queries and clicked documents into session vectors.
        Parameters
        --------------------
            encoded_queries     -- 3d tensor (batch_size,session_length,nhid_query)
            encoded_docs        -- 4d tensor (batch_size,session_length,num_candidates,nhid_document)
            encoded_clicks      -- 3d tensor (batch_size,session_length,nhid_document)
            doc_labels          -- 3d tensor (batch_size,session_length,num_candidates)
        Returns
        --------------------
            click_loss          -- an autograd variable, representing average batch click loss
            hidden_states       -- tuple or single 3d tensor
        """

        batch_size = encoded_queries.size(0)
        session_len = encoded_queries.size(1)
        use_cuda = encoded_queries.is_cuda

        sess_q_hidden = None
        sess_d_hidden = None

        query_session_states = []
        if not self.no_query_session_encoding:
            sess_q_out = torch.zeros(batch_size, self.nhid_session_query)
            if use_cuda:
                sess_q_out = sess_q_out.cuda()
            query_session_states.append(sess_q_out)

        document_session_states = []
        if not self.no_document_session_encoding:
            sess_d_out = torch.zeros(batch_size, self.nhid_session_document)
            if use_cuda:
                sess_d_out = sess_d_out.cuda()
            document_session_states.append(sess_d_out)

        hidden_states, cell_states = [], []
        query_session_attns, doc_session_attns = [], []
        document_scores = []
        # loop over all the queries in a session
        for idx in range(session_len):
            if not self.no_ranker:
                if not self.no_query_session_encoding:
                    temp_q_hids = torch.stack(query_session_states, 1)
                    intermediate = self.session_query_attn(temp_q_hids)
                    att_weights = torch.bmm(intermediate, encoded_queries[:, idx, :].unsqueeze(2))
                    att_weights = att_weights.squeeze(2)
                    att_weights = f.softmax(att_weights, 1)
                    sess_q_out = torch.bmm(temp_q_hids.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
                else:
                    sess_q_out = None

                if not self.no_document_session_encoding:
                    temp_d_hids = torch.stack(document_session_states, 1)
                    intermediate = self.session_doc_attn(temp_d_hids)
                    att_weights = torch.bmm(intermediate, encoded_queries[:, idx, :].unsqueeze(2))
                    att_weights = att_weights.squeeze(2)
                    att_weights = f.softmax(att_weights, 1)
                    sess_d_out = torch.bmm(temp_d_hids.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
                else:
                    sess_d_out = None

                # batch_size x num_candidates
                score = self.rank(encoded_queries[:, idx, :],
                                  sess_q_out,
                                  sess_d_out,
                                  encoded_docs[:, idx, :, :])
                document_scores.append(score)

            hid, cell = None, None
            if not self.no_query_session_encoding:
                # update session-level query encoder state using query representations
                sess_q_hidden, sess_q_out = self.session_query_encoder(encoded_queries[:, idx, :].unsqueeze(1),
                                                                       None,
                                                                       sess_q_hidden)
                sess_q_out = self.dropout(sess_q_out)
                sess_q_out = sess_q_out.squeeze(1)
                query_session_states.append(sess_q_out)

                temp_q_hids = torch.stack(query_session_states[1:], 1)
                att_weights = self.session_query_inner_attn(temp_q_hids).squeeze(2)
                att_weights = f.softmax(att_weights, 1)
                inner_q_out = torch.bmm(temp_q_hids.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
                query_session_attns.append(inner_q_out)

                # layers x batch_size x nhid_session_query
                if isinstance(sess_q_hidden, tuple):  # LSTM
                    hid = sess_q_hidden[0]
                    cell = sess_q_hidden[1]
                else:  # GRU
                    hid = sess_q_hidden

            if not self.no_document_session_encoding:
                # update session-level document encoder state using click representations
                sess_d_hidden, sess_d_out = self.session_doc_encoder(encoded_clicks[:, idx, :].unsqueeze(1),
                                                                     None,
                                                                     sess_d_hidden)
                sess_d_out = self.dropout(sess_d_out)
                sess_d_out = sess_d_out.squeeze(1)
                document_session_states.append(sess_d_out)

                temp_d_hids = torch.stack(document_session_states[1:], 1)
                att_weights = self.session_doc_inner_attn(temp_d_hids).squeeze(2)
                att_weights = f.softmax(att_weights, 1)
                inner_d_out = torch.bmm(temp_d_hids.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
                doc_session_attns.append(inner_d_out)

                # layers x batch_size x (nhid_session_query + nhid_session_document)
                if isinstance(sess_d_hidden, tuple):
                    temp_hid = sess_d_hidden[0]
                    temp_cell = sess_d_hidden[1]
                    hid = temp_hid if hid is None else \
                        torch.cat((hid, temp_hid), dim=2)
                    cell = temp_cell if cell is None else \
                        torch.cat((cell, temp_cell), dim=2)
                else:
                    temp_hid = sess_d_hidden[0]
                    hid = temp_hid if hid is None else \
                        torch.cat((hid, temp_hid), dim=2)

            if hid is not None:
                hidden_states.append(hid)
            if cell is not None:
                cell_states.append(cell)

        # sess_q_attn: batch_size x sess_len x nhid_query_session
        # sess_d_attn: batch_size x sess_len x nhid_doc_session
        sess_q_attn = torch.stack(query_session_attns, 1) if query_session_attns else None
        sess_d_attn = torch.stack(doc_session_attns, 1) if doc_session_attns else None

        if sess_q_attn is not None and sess_d_attn is not None:
            assert sess_q_attn.size(1) == session_len
            assert sess_d_attn.size(1) == session_len

        if hidden_states:
            # layers x (batch_size*session_len-1) x (nhid_session_query + nhid_session_document)
            hidden_states = torch.cat(hidden_states[:-1], dim=1)
            # layers x (batch_size*session_len-1) x nhid_decoder
            hidden_states = self.transform_hid(hidden_states)
        else:
            hidden_states = None

        if cell_states:
            cell_states = torch.cat(cell_states[:-1], dim=1)
            cell_states = self.transform_cell(cell_states)
            states = (hidden_states, cell_states)
        else:
            states = hidden_states

        # batch_size x session_len x num_candidates
        if document_scores:
            document_scores = torch.stack(document_scores, dim=1)
        return (sess_q_attn, sess_d_attn), document_scores, states

    def rank(self, query_rep, q_sess_rep, d_sess_rep, encoded_docs):
        """
        Compute relevance score of candidate documents for each query.
        Parameters
        --------------------
            query_rep       -- 2d tensor (batch_size,nhid_query), current query representation
            q_sess_rep      -- 2d tensor (batch_size,nhid_session_query), in-session previous query representation
            d_sess_rep      -- 2d tensor (batch_size,nhid_session_document), in-session previous click representation
            encoded_docs    -- 3d tensor (batch_size,num_candidates,nhid_document), document representations
        Returns
        --------------------
            click_score1    -- 2d tensor (n, r), relevance score between current query and documents
            click_score2    -- 2d tensor (n, r), relevance between previous in-seesion queries and documents
            click_score3    -- 2d tensor (n, r), relevance between previous in-seesion clicks and documents
        """
        batch_size = encoded_docs.size(0)
        num_candidates = encoded_docs.size(1)
        nhid_document = encoded_docs.size(2)

        query_rep = self.q_projection(query_rep)
        query_rep = query_rep.unsqueeze(1).expand(batch_size,
                                                  num_candidates,
                                                  nhid_document)

        session_rep = []
        if not self.no_query_session_encoding:
            nhid_session_query = q_sess_rep.size(1)
            q_sess_rep = q_sess_rep.unsqueeze(1)
            q_sess_rep = q_sess_rep.expand(batch_size,
                                           num_candidates,
                                           nhid_session_query)
            session_rep.append(q_sess_rep)

        if not self.no_document_session_encoding:
            nhid_session_document = d_sess_rep.size(1)
            d_sess_rep = d_sess_rep.unsqueeze(1)
            d_sess_rep = d_sess_rep.expand(batch_size,
                                           num_candidates,
                                           nhid_session_document)
            session_rep.append(d_sess_rep)

        query_rep = query_rep.contiguous().view(batch_size * num_candidates,
                                                nhid_document)
        doc_rep = encoded_docs.contiguous().view(batch_size * num_candidates,
                                                 nhid_document)

        if session_rep:
            # batch_size x num_candidates x (nhid_session_query + nhid_session_document)
            session_rep = torch.cat(session_rep, dim=2)  # n,r,2d
            session_rep = self.shared_session_projector(session_rep) + \
                          self.private_session_projector1(session_rep)
            session_rep = session_rep.view(batch_size * num_candidates, nhid_document).contiguous()
            query_rep = query_rep + session_rep

        click_score = self.ranknet(torch.cat((query_rep,
                                              doc_rep,
                                              torch.abs(query_rep - doc_rep),
                                              torch.mul(query_rep, doc_rep)),
                                             1)).squeeze(1)
        click_score = click_score.view(batch_size, num_candidates)
        return click_score

    def rank_document(self,
                      pooled_rep,
                      document_rep,
                      document_len,
                      document_label):

        encoded_docs, encoded_clicks = None, None
        if not (self.no_ranker and self.no_document_session_encoding):
            # encoded_docs: batch_size x session_len x num_candidates x nhid_document
            encoded_docs = self.encode_document(document_rep, document_len)
            if not self.no_document_session_encoding:
                # encoded_clicks: batch_size x session_len x nhid_document
                encoded_clicks = self.encode_clicks(encoded_docs, document_label)

        session_attns, click_scores, hidden_states = self.encode_session(pooled_rep,
                                                                         encoded_docs,
                                                                         encoded_clicks)

        return click_scores, hidden_states, session_attns

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

        # pooled_rep: batch_size x session_len x nhid_query
        # encoded_queries: (batch_size * session_len) x max_query_length x nhid_query
        pooled_rep, encoded_queries, hidden = self.encode(source_rep, source_len)

        click_scores, hidden_states, session_attns = self.rank_document(pooled_rep,
                                                                        document_rep,
                                                                        document_len,
                                                                        document_label)

        # both query-session and document-session encoders cannot be turned off
        assert all(state is not None for state in hidden_states)

        cat_session_rep = []
        if session_attns[0] is not None:
            # batch_size x session_len x nhid_query_session
            cat_session_rep.append(session_attns[0])
        if session_attns[1] is not None:
            # batch_size x session_len x nhid_doc_session
            cat_session_rep.append(session_attns[1])
        if cat_session_rep:
            # batch_size x session_len x nhid_session
            cat_session_rep = torch.cat(cat_session_rep, dim=2)
            # cat_session_rep: (batch_size * session_len - 1) x session_hid_size
            cat_session_rep = cat_session_rep[:, :-1, :].contiguous()
            cat_session_rep = cat_session_rep.view(batch_size * (session_len - 1),
                                                   cat_session_rep.size(2))
        else:
            cat_session_rep = None

        result_dict = {
            'ranking_loss': None,
            'suggestion_loss': None
        }

        if not self.no_ranker:
            result_dict['ranking_loss'] = f.binary_cross_entropy_with_logits(click_scores, document_label)

        # ------- Decoding -------
        if not self.no_recommender:
            target_rep = target_rep.view(batch_size * (session_len - 1), -1).contiguous()
            target_seq = target_seq.view(batch_size * (session_len - 1), -1).contiguous()
            target_len = target_len.view(-1).contiguous()

            # (batch*session_len-1) x max_src_len x emsize
            target_word_rep = self.embedder(target_rep)
            memory_bank = encoded_queries.view(batch_size,
                                               session_len,
                                               encoded_queries.size(1),
                                               encoded_queries.size(2))
            memory_bank = memory_bank[:, :-1, :, :].contiguous()
            memory_bank = memory_bank.view(batch_size * (session_len - 1),
                                           memory_bank.size(2),
                                           memory_bank.size(3))
            # (batch*session_len-1) x max_src_len x nhid_decoder
            memory_bank = self.dec_attn(memory_bank)
            memory_len = source_len[:, :-1].contiguous().view(-1)

            init_decoder_state = self.decoder.init_decoder(hidden_states)
            decoder_outputs, _ = self.decoder(target_word_rep,
                                              memory_bank,
                                              memory_len,
                                              init_decoder_state)

            # `(batch*session_len) x max_tgt_len - 1 x vocab_size`
            decoder_outputs = decoder_outputs[:, :-1, :].contiguous()
            target = target_seq[:, 1:].contiguous()

            decoder_outputs = self.token_prob_predictor1(decoder_outputs)
            if cat_session_rep is not None:
                session_rep = self.shared_session_projector(cat_session_rep) + \
                              self.private_session_projector2(cat_session_rep)
                session_rep = torch.stack([session_rep] * decoder_outputs.size(1), dim=1)
                decoder_outputs = decoder_outputs + session_rep

            decoder_outputs = self.dropout(decoder_outputs)
            scores = self.token_prob_predictor2(decoder_outputs)
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

            result_dict['suggestion_loss'] = decoding_loss

        # if both ranker and recommender is active
        if not self.no_ranker and not self.no_recommender:
            if not (self.no_query_session_encoding or self.no_document_session_encoding):
                w1 = self.shared_session_projector.linear.weight.norm(2)
                w2 = self.private_session_projector1.linear.weight.norm(2) + \
                     self.private_session_projector2.linear.weight.norm(2)
                result_dict['regularization'] = self.lambda1 * w1 + self.lambda2 * w2
            else:
                result_dict['regularization'] = None

        return result_dict

    def apply_pooling(self, encodings, pool_type, dtype, mask=None):
        if pool_type == 'max':
            pooled_encodings = encodings.max(1)[0]
        elif pool_type == 'mean':
            pooled_encodings = encodings.mean(1)
        elif pool_type == 'attn':
            attn_weights = None
            if dtype == 'query':
                attn_weights = self.q_attn(encodings).squeeze(2)
            elif dtype == 'document':
                attn_weights = self.d_attn(encodings).squeeze(2)

            if mask is not None:
                attn_weights.masked_fill_(~mask, -float('inf'))
            attn_weights = f.softmax(attn_weights, 1)
            pooled_encodings = torch.bmm(encodings.transpose(1, 2),
                                         attn_weights.unsqueeze(2)).squeeze(2)
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
               encoded_source,
               source_len,
               session_attns):

        # both query-session and document-session encoders cannot be turned off
        assert all(state is not None for state in states)

        cat_session_rep = []
        if session_attns[0] is not None:
            # batch_size x session_len x nhid_query_session
            cat_session_rep.append(session_attns[0])
        if session_attns[1] is not None:
            # batch_size x session_len x nhid_doc_session
            cat_session_rep.append(session_attns[1])
        if cat_session_rep:
            # batch_size x session_len x nhid_session
            cat_session_rep = torch.cat(cat_session_rep, dim=2)
            # cat_session_rep: (batch_size * session_len - 1) x session_hid_size
            cat_session_rep = cat_session_rep[:, :-1, :].contiguous()
            cat_session_rep = cat_session_rep.view(batch_size * session_len,
                                                   cat_session_rep.size(2))
        else:
            cat_session_rep = None

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
            memory_bank = encoded_source.view(batch_size,
                                              session_len + 1,
                                              encoded_source.size(1),
                                              encoded_source.size(2))
            memory_bank = memory_bank[:, :-1, :, :].contiguous()
            memory_bank = memory_bank.view(batch_size * session_len,
                                           memory_bank.size(2),
                                           memory_bank.size(3))
            # (batch*session_len-1) x max_src_len x nhid_decoder
            memory_bank = self.dec_attn(memory_bank)
            memory_len = source_len[:, :-1].contiguous().view(-1)

            decoder_outputs, _ = self.decoder(target_word_rep,
                                              memory_bank,
                                              memory_len,
                                              init_decoder_state)

            decoder_outputs = self.token_prob_predictor1(decoder_outputs.squeeze(1))
            if cat_session_rep is not None:
                session_rep = self.shared_session_projector(cat_session_rep) + \
                              self.private_session_projector2(cat_session_rep)
                decoder_outputs = decoder_outputs + session_rep

            prediction = self.token_prob_predictor2(decoder_outputs)
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
