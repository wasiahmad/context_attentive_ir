# Adapted from https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import random
import copy
import torch


def vectorize(ex, model, shuffle=False):
    """Torchify a single example."""
    src_dict = model.src_dict
    query, candidates = ex, copy.deepcopy(ex.documents)
    if shuffle:
        random.shuffle(candidates)

    # Index words
    Q_words = torch.LongTensor(query.vectorize(word_dict=src_dict))
    D_words = [torch.LongTensor(c.vectorize(word_dict=src_dict)) for c in candidates]
    max_doc_len = model.args.max_doc_len if model.args.force_pad \
        else max([len(c.tokens) for c in candidates])
    max_query_len = model.args.max_query_len \
        if model.args.force_pad else len(query.tokens)

    # label is only used to compute loss during training
    label = torch.LongTensor([c.label for c in candidates])

    return {
        'id': query.id,
        'query_tokens': query.tokens,
        'query_words': Q_words,
        'doc_tokens': [c.tokens for c in candidates],
        'doc_words': D_words,
        'label': label,
        'num_candidates': model.args.num_candidates,
        'max_doc_len': max_doc_len,
        'max_query_len': max_query_len
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    num_candidates = batch[0]['num_candidates']
    max_doc_len = max([b['max_doc_len'] for b in batch])
    max_que_len = max([b['max_query_len'] for b in batch])

    # --------- Prepare document tensors ---------

    batch_documents = [ex['doc_words'] for ex in batch]

    # Batch documents
    doc_len = torch.LongTensor(batch_size, num_candidates).zero_()
    doc_word = torch.LongTensor(batch_size,
                                num_candidates,
                                max_doc_len).zero_()

    for bidx, docs in enumerate(batch_documents):
        for didx, doc in enumerate(docs):
            doc_len[bidx, didx] = doc.size(0)
            doc_word[bidx, didx, :doc.size(0)].copy_(doc)

    # --------- Prepare query tensors ---------
    batch_queries = [ex['query_words'] for ex in batch]

    # Batch questions
    que_len = torch.LongTensor(batch_size).zero_()
    que_word = torch.LongTensor(batch_size,
                                max_que_len).zero_()

    for bidx, query in enumerate(batch_queries):
        que_len[bidx] = query.size(0)
        que_word[bidx, :query.size(0)].copy_(query)

    # --------- Prepare other tensors ---------
    ids = [ex['id'] for ex in batch]
    labels = [ex['label'] for ex in batch]
    label_tensor = torch.LongTensor(batch_size, num_candidates).zero_()
    for bidx, label in enumerate(labels):
        label_tensor[bidx, :].copy_(label)

    return {
        'batch_size': batch_size,
        'ids': ids,
        'doc_rep': doc_word,
        'doc_len': doc_len,
        'que_rep': que_word,
        'que_len': que_len,
        'label': label_tensor
    }
