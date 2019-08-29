import numpy


def MAP(predictions, target):
    """
    Compute mean average precision.
    :param predictions: 2d list [batch_size x num_candidate_paragraphs]
    :param target: 2d list [batch_size x num_candidate_paragraphs]
    :return: mean average precision [a float value]
    """
    assert predictions.shape == target.shape
    assert predictions.ndim == target.ndim == 2

    nrow, ncolumn = target.shape[0], target.shape[1]

    map = 0
    for i in range(nrow):
        average_precision, num_rel = 0, 0
        for j in range(ncolumn):
            if target[i, predictions[i, j]] == 1:
                num_rel += 1
                average_precision += num_rel / (j + 1)
        average_precision = average_precision / num_rel
        map += average_precision

    return map / nrow


def precision_at_k(predictions, target, k):
    """
    Compute precision at k.
    :param predictions: 2d list [batch_size x num_candidate_paragraphs]
    :param target: 2d list [batch_size x num_candidate_paragraphs]
    :return: precision@K [a float value]
    """
    assert predictions.shape == target.shape
    assert predictions.ndim == target.ndim == 2

    nrow, ncolumn = target.shape[0], target.shape[1]
    assert ncolumn >= k, 'Precision@K cannot be computed, invalid value of K.'

    p_at_k = 0
    for i in range(nrow):
        num_rel = numpy.count_nonzero(target[i, predictions[i, :k]])
        p_at_k += num_rel / k

    return p_at_k / nrow


def recall_at_k(predictions, target, k):
    """
    Compute recall at k.
    :param predictions: 2d list [batch_size x num_candidate_paragraphs]
    :param target: 2d list [batch_size x num_candidate_paragraphs]
    :return: precision@K [a float value]
    """
    assert predictions.shape == target.shape
    assert predictions.ndim == target.ndim == 2

    nrow, ncolumn = target.shape[0], target.shape[1]
    assert ncolumn >= k, 'Recall@K cannot be computed, invalid value of K.'

    r_at_k = 0
    for i in range(nrow):
        num_rel = numpy.count_nonzero(target[i, predictions[i, :k]])
        total_rel = numpy.count_nonzero(target[i])
        r_at_k += num_rel / total_rel

    return r_at_k / nrow


def NDCG_at_k(predictions, target, k):
    """
    Compute normalized discounted cumulative gain.
    :param predictions: 2d list [batch_size x num_candidate_paragraphs]
    :param target: 2d list [batch_size x num_candidate_paragraphs]
    :return: NDCG@k [a float value]
    """
    assert predictions.shape == target.shape
    assert predictions.ndim == target.ndim == 2

    nrow, ncolumn = target.shape[0], target.shape[1]
    assert ncolumn >= k, 'NDCG@K cannot be computed, invalid value of K.'

    NDCG = 0
    for i in range(nrow):
        DCG_ref = 0
        num_rel_docs = numpy.count_nonzero(target[i])
        for j in range(ncolumn):
            if j == k:
                break
            if target[i, predictions[i, j]] == 1:
                DCG_ref += 1 / numpy.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += 1 / numpy.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return NDCG / nrow


def MRR(predictions, target):
    """
    Compute mean reciprocal rank.
    :param predictions: 2d list [batch_size x num_candidate_paragraphs]
    :param target: 2d list [batch_size x num_candidate_paragraphs]
    :return: mean reciprocal rank [a float value]
    """
    assert predictions.shape == target.shape
    assert predictions.ndim == target.ndim == 2

    nrow, ncolumn = target.shape[0], target.shape[1]

    total_reciprocal_rank = 0
    for i in range(nrow):
        for j in range(ncolumn):
            if target[i, predictions[i, j]] == 1:
                total_reciprocal_rank += 1.0 / (j + 1)
                break

    return total_reciprocal_rank / nrow
