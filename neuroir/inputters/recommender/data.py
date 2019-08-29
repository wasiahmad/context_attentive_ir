# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize


# ------------------------------------------------------------------------------
# PyTorch dataset class for MSMARCO data.
# ------------------------------------------------------------------------------


class RecommenderDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model)

    def lengths(self):
        return [len(session) for session in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and query).
# ------------------------------------------------------------------------------

class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        clusters = dict()
        for i, num_queries in enumerate(self.lengths):
            if num_queries in clusters:
                clusters[num_queries].append(i)
            else:
                clusters[num_queries] = [i]

        batches = []
        for key, indices in clusters.items():
            if len(indices) % self.batch_size != 0:
                num_batch = len(indices) // self.batch_size
                indices = indices[:(num_batch * self.batch_size)]
            assert len(indices) % self.batch_size == 0
            batches.extend([indices[i:i + self.batch_size]
                            for i in range(0, len(indices), self.batch_size)])
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
