# Adapted from https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import torch


def vectorize(session, model):
    """Torchify a single example."""

    src_dict = model.src_dict
    tgt_dict = model.tgt_dict

    session_len = len(session)
    max_source_len = max([len(query) for query in session.queries[:-1]])
    max_target_len = max([len(query) for query in session.queries[1:]])
    source_tokens = [query.tokens for query in session.queries[:-1]]  # 2d list
    target_tokens = [query.tokens for query in session.queries[1:]]  # 2d list

    source_words = torch.LongTensor(session_len - 1, max_source_len).zero_()
    source_lens = torch.LongTensor(session_len - 1).zero_()
    target_words = torch.LongTensor(session_len - 1, max_target_len).zero_()
    target_lens = torch.LongTensor(session_len - 1).zero_()
    target_seq = torch.LongTensor(session_len - 1, max_target_len).zero_()  # use only to compute loss

    for i in range(session_len - 1):
        query = session.queries[i]
        query_len = len(query.tokens)
        source_lens[i] = query_len
        source_words[i, :query_len].copy_(torch.LongTensor(
            query.vectorize(word_dict=src_dict)))

        query = session.queries[i + 1]
        query_len = len(query.tokens)
        target_lens[i] = query_len
        target_words[i, :query_len].copy_(torch.LongTensor(
            query.vectorize(word_dict=src_dict)))
        target_seq[i, :query_len].copy_(torch.LongTensor(
            query.vectorize(word_dict=tgt_dict)))  # diff is which dict is used

    source_vocab = None
    if session_len == 2:
        source_vocab = session.queries[0].src_vocab

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
        'src_vocab': source_vocab
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    max_source_len = max([b['max_source_len'] for b in batch])
    max_target_len = max([b['max_target_len'] for b in batch])
    session_len = batch[0]['session_len']

    # all the sessions must have the same length
    assert len(set([b['session_len'] for b in batch])) == 1

    ids = [ex['id'] for ex in batch]

    # --------- Prepare query tensors ---------
    source_lens = torch.LongTensor(batch_size,
                                   session_len - 1).zero_()
    source_words = torch.LongTensor(batch_size,
                                    session_len - 1,
                                    max_source_len).zero_()
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
        target_lens[bidx] = session['target_lens']
        source_words[bidx, :, :session['max_source_len']].copy_(session['source_words'])
        target_words[bidx, :, :session['max_target_len']].copy_(session['target_words'])
        target_seq[bidx, :, :session['max_target_len']].copy_(session['target_seq'])

    # --------- Prepare other tensors ---------
    # prepare source vocabs, alignment [required for Copy Attention]
    source_maps = []
    alignments = []
    src_vocabs = []
    if session_len == 2:
        for idx in range(batch_size):
            target = batch[idx]['target_tokens'][0]
            context = batch[idx]['source_tokens'][0]
            vocab = batch[idx]['src_vocab']
            src_vocabs.append(vocab)

            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([vocab[w] for w in context])
            source_maps.append(src_map)

            mask = torch.LongTensor([vocab[w] for w in target])
            alignments.append(mask)

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
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'session_len': session_len - 1
    }
