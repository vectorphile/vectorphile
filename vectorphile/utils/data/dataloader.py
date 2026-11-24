"""DataLoader — batches and shuffles a Dataset."""
from __future__ import annotations
import random as _random
from typing import Iterator, Optional
from vectorphile.utils.data.dataset import Dataset
from vectorphile.core.tensor import Tensor, stack
from vectorphile._core.ndarray import array as _nd_array


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 0, drop_last: bool = False, pin_memory: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            _random.shuffle(indices)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start:min(end, n)]
            batch = [self.dataset[i] for i in batch_idx]
            if isinstance(batch[0], tuple):
                yield tuple(_collate([item[i] for item in batch]) for i in range(len(batch[0])))
            else:
                yield _collate(batch)


def _collate(samples):
    if isinstance(samples[0], Tensor):
        return stack(samples, dim=0)
    if isinstance(samples[0], (int, float)):
        return Tensor(_nd_array([float(s) for s in samples], dtype='float32'))
    if isinstance(samples[0], list):
        flat = [v for s in samples for v in s]
        return Tensor(_nd_array(flat, dtype='float32').reshape((len(samples), len(samples[0]))))
    return samples
