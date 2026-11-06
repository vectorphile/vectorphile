"""Dataset abstractions."""
from __future__ import annotations
import numpy as np
from typing import Tuple
from vectorphile.core.tensor import Tensor


class Dataset:
    """Abstract base class for all datasets."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors — each sample is a tuple of matching tensors."""

    def __init__(self, *tensors: Tensor):
        assert all(len(tensors[0]) == len(t) for t in tensors), 'Tensors must have same first dimension'
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index) -> Tuple[Tensor, ...]:
        return tuple(t[index] for t in self.tensors)
