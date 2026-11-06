"""Index class for DataFrame/Series labels."""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional


class Index:
    def __init__(self, data=None, name=None):
        if data is None:
            self._data = np.array([])
        elif isinstance(data, Index):
            self._data = data._data.copy()
        else:
            self._data = np.array(data)
        self.name = name

    @property
    def values(self) -> np.ndarray:
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self):
        return len(self._data)

    def __contains__(self, item):
        return item in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def tolist(self) -> list:
        return self._data.tolist()

    def get_loc(self, key) -> int:
        matches = np.where(self._data == key)[0]
        if len(matches) == 0:
            raise KeyError(key)
        return int(matches[0])

    def isin(self, values) -> np.ndarray:
        return np.isin(self._data, values)

    def unique(self) -> 'Index':
        return Index(np.unique(self._data), name=self.name)

    def sort_values(self, ascending=True) -> 'Index':
        order = np.argsort(self._data)
        if not ascending:
            order = order[::-1]
        return Index(self._data[order], name=self.name)

    def append(self, other: 'Index') -> 'Index':
        return Index(np.concatenate([self._data, other._data]), name=self.name)

    def difference(self, other: 'Index') -> 'Index':
        return Index(np.setdiff1d(self._data, other._data), name=self.name)

    def intersection(self, other: 'Index') -> 'Index':
        return Index(np.intersect1d(self._data, other._data), name=self.name)

    def union(self, other: 'Index') -> 'Index':
        return Index(np.union1d(self._data, other._data), name=self.name)

    def __repr__(self):
        return f"Index({self._data.tolist()}, name={self.name!r})"

    def __str__(self):
        return self.__repr__()


class RangeIndex(Index):
    def __init__(self, start=0, stop=None, step=1, name=None):
        if stop is None:
            stop = start
            start = 0
        super().__init__(np.arange(start, stop, step), name=name)
        self.start, self.stop, self.step = start, stop, step

    def __repr__(self):
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"
