import random
import timeit
import numpy as np
import torch

M = 8000000
#x = torch.randint(0, 100, (M, 3), dtype=torch.int16)
x = np.random.randint(0, 100, (M, 3), dtype=np.int16)


size = 20

def sample1():
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i])

def sample2():
    y = np.random.permutation(x)
    return torch.tensor(y[:size])

print(timeit.timeit(sample1, number=10)) # fast
print(timeit.timeit(sample2, number=10)) # never finish
