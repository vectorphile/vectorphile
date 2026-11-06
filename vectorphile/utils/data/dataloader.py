"""DataLoader — batches and shuffles a Dataset."""
from __future__ import annotations
import numpy as np
from typing import Iterator, Optional
from vectorphile.utils.data.dataset import Dataset
from vectorphile.core.tensor import Tensor, stack


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
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start:end]
            batch = [self.dataset[int(i)] for i in batch_idx]
            if isinstance(batch[0], tuple):
                yield tuple(_collate([item[i] for item in batch]) for i in range(len(batch[0])))
            else:
                yield _collate(batch)


def _collate(samples):
    if isinstance(samples[0], Tensor):
        return stack(samples, dim=0)
    if isinstance(samples[0], np.ndarray):
        return Tensor(np.stack(samples, axis=0))
    if isinstance(samples[0], (int, float)):
        return Tensor(np.array(samples, dtype=np.float32))
    return samples
