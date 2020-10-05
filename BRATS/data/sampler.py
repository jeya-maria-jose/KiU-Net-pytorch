import random
import torch

from torch.utils.data.sampler import Sampler

# Adapted from
# https://github.com/pytorch/pytorch/pull/3062/files
class RandomCycleIter(object):
    def __init__(self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            random.shuffle(self.data_list)
        return self.data_list[self.i]

    next = __next__  # Py2


def multi_data_generator(data_iters, index_data, n, size):
    i = 0
    while i < n:
        index = i % size
        d = index_data[index]
        yield d, next(data_iters[d])
        i += 1

class MSampler(object):
    def __init__(self, batch_sizes, sizes, num_samples=None, num_iters=None):
        self.batch_size = sum(batch_sizes)
        self.index_data = {}
        size, c = 0, -1
        for i in range(self.batch_size):
            if i == size:
                c    += 1
                size += batch_sizes[c]
            self.index_data[i] = c

        self.num_samples = num_samples or num_iters*self.batch_size or sum(sizes)
        self.data_iters = [RandomCycleIter(range(n)) for n in sizes]

    def __iter__(self):
        return multi_data_generator(
                self.data_iters, self.index_data,
                self.num_samples, self.batch_size)

    def __len__(self):
        return self.num_samples


def single_data_generator(data_iter, n):
    i = 0
    while i < n:
        yield next(data_iter)
        i += 1

class CycleSampler(Sampler):
    def __init__(self, size, num_samples=None, num_epochs=0):
        self.num_samples = num_samples or size*num_epochs
        self.data_iter = RandomCycleIter(range(size))

    def __iter__(self):
        return single_data_generator(self.data_iter, self.num_samples)

    def __len__(self):
        return self.num_samples


import numpy as np
class RandomSampler(object):
    def __init__(self, data_source, state=None, seed=None):
        self.data_source = data_source
        self.rng = np.random.RandomSatate(seed)

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)

    def get_state(self):
        return self.rng.get_state()

    def set_state(self, state):
        self.rng.set_state(state)

