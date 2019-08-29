# Adapted from https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import torch
import random
import copy


def vectorize(session, model, shuffle=False):
    """Torchify a single example."""

    src_dict = model.src_dict
    tgt_dict = model.tgt_dict
    num_candidates = model.args.num_candidates

    session_len = len(session)
    max_source_len = max([len(query) for query in session.queries])
    max_target_len = max([len(query) for query in session.queries[1:]])
    max_document_len = max([len(doc) for query in session.queries for doc in query.documents])

    source_tokens = [query.tokens for query in session.queries]  # 2d list
    target_tokens = [query.tokens for query in session.queries[1:]]  # 2d list

    source_words = torch.LongTensor(session_len, max_source_len).zero_()
    source_lens = torch.LongTensor(session_len).zero_()
    target_words = torch.LongTensor(session_len - 1, max_target_len).zero_()
    target_lens = torch.LongTensor(session_len - 1).zero_()
    target_seq = torch.LongTensor(session_len - 1, max_target_len).zero_()  # use only to compute loss

    document_words = torch.LongTensor(session_len, num_candidates, max_document_len).zero_()
    document_lens = torch.LongTensor(session_len, num_candidates).zero_()
    document_labels = torch.LongTensor(session_len, num_candidates).zero_()

    for i in range(session_len):
        query = session.queries[i]
        query_len = len(query.tokens)
        source_lens[i] = query_len
        source_words[i, :query_len].copy_(torch.LongTensor(
            query.vectorize(word_dict=src_dict)))

        # candidate document ranking
        candidates = copy.deepcopy(query.documents)
        assert len(candidates) == num_candidates
        if shuffle:
            random.shuffle(candidates)
        for cidx in range(num_candidates):
            cand = candidates[cidx]
            document_lens[i, cidx] = len(cand.tokens)
            document_labels[i, cidx] = cand.label
            document_words[i, cidx, :len(cand.tokens)].copy_(torch.LongTensor(
                cand.vectorize(word_dict=src_dict)))

        if i != session_len - 1:
            # next query suggestion
            query = session.queries[i + 1]
            query_len = len(query.tokens)
            target_lens[i] = query_len
            target_words[i, :query_len].copy_(torch.LongTensor(
                query.vectorize(word_dict=src_dict)))
            target_seq[i, :query_len].copy_(torch.LongTensor(
                query.vectorize(word_dict=tgt_dict)))  # diff is which dict is used

    return {
        'id': session.id,
        'source_tokens': source_tokens,
        'source_words': source_words,
        'source_lens': source_lens,
        'target_tokens': target_tokens,
        'target_words': target_words,
        'target_lens': target_lens,
        'target_seq': target_seq,
        'max_source_len': max_source_len,
        'max_target_len': max_target_len,
        'session_len': session_len,
        'num_candidates': num_candidates,
        'document_words': document_words,  # 3d tensor
        'document_lens': document_lens,  # 2d tensor
        'document_labels': document_labels,  # 2d tensor
        'max_document_len': max_document_len
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    max_source_len = max([b['max_source_len'] for b in batch])
    max_target_len = max([b['max_target_len'] for b in batch])
    max_document_len = max([b['max_document_len'] for b in batch])
    session_len = batch[0]['session_len']
    num_candidates = batch[0]['num_candidates']

    # all the sessions must have the same length
    assert len(set([b['session_len'] for b in batch])) == 1

    ids = [ex['id'] for ex in batch]

    # --------- Prepare query tensors ---------
    source_lens = torch.LongTensor(batch_size,
                                   session_len).zero_()
    source_words = torch.LongTensor(batch_size,
                                    session_len,
                                    max_source_len).zero_()
    document_lens = torch.LongTensor(batch_size,
                                     session_len,
                                     num_candidates).zero_()
    document_words = torch.LongTensor(batch_size,
                                      session_len,
                                      num_candidates,
                                      max_document_len).zero_()
    document_labels = torch.FloatTensor(batch_size,
                                        session_len,
                                        num_candidates).zero_()
    target_lens = torch.LongTensor(batch_size,
                                   session_len - 1).zero_()
    target_words = torch.LongTensor(batch_size,
                                    session_len - 1,
                                    max_target_len).zero_()
    target_seq = torch.LongTensor(batch_size,
                                  session_len - 1,
                                  max_target_len).zero_()

    for bidx, session in enumerate(batch):
        source_lens[bidx] = session['source_lens']
        source_words[bidx, :, :session['max_source_len']].copy_(session['source_words'])

        document_lens[bidx] = session['document_lens']
        document_labels[bidx] = session['document_labels']
        document_words[bidx, :, :, :session['max_document_len']].copy_(session['document_words'])

        target_lens[bidx] = session['target_lens']
        target_words[bidx, :, :session['max_target_len']].copy_(session['target_words'])
        target_seq[bidx, :, :session['max_target_len']].copy_(session['target_seq'])

    return {
        'batch_size': batch_size,
        'ids': ids,
        'source_tokens': [item['source_tokens'] for item in batch],
        'source_words': source_words,
        'source_lens': source_lens,
        'target_tokens': [item['target_tokens'] for item in batch],
        'target_words': target_words,
        'target_lens': target_lens,
        'target_seq': target_seq,
        'session_len': session_len,
        'document_words': document_words,
        'document_lens': document_lens,
        'document_labels': document_labels
    }
