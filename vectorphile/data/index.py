"""Index class for DataFrame/Series labels — pure Python."""
from __future__ import annotations
from typing import Any, List, Optional


class Index:
    def __init__(self, data=None, name=None):
        if data is None:
            self._data = []
        elif isinstance(data, Index):
            self._data = data._data[:]
        else:
            self._data = list(data)
        self.name = name

    @property
    def values(self) -> list:
        return self._data

    @property
    def dtype(self):
        return type(self._data[0]).__name__ if self._data else 'object'

    def __len__(self): return len(self._data)
    def __contains__(self, item): return item in self._data
    def __getitem__(self, key): return self._data[key]
    def __iter__(self): return iter(self._data)

    def tolist(self) -> list: return self._data[:]

    def get_loc(self, key) -> int:
        try: return self._data.index(key)
        except ValueError: raise KeyError(key)

    def isin(self, values) -> List[bool]:
        s = set(values)
        return [v in s for v in self._data]

    def unique(self) -> 'Index':
        seen = {}
        return Index([seen.setdefault(v, v) for v in self._data if v not in seen], name=self.name)

    def sort_values(self, ascending=True) -> 'Index':
        return Index(sorted(self._data, reverse=not ascending), name=self.name)

    def append(self, other: 'Index') -> 'Index':
        return Index(self._data + other._data, name=self.name)

    def difference(self, other: 'Index') -> 'Index':
        s = set(other._data)
        return Index([v for v in self._data if v not in s], name=self.name)

    def intersection(self, other: 'Index') -> 'Index':
        s = set(other._data)
        seen = set()
        return Index([seen.add(v) or v for v in self._data if v in s and v not in seen], name=self.name)

    def union(self, other: 'Index') -> 'Index':
        seen = {}
        result = [seen.setdefault(v, v) for v in self._data + other._data if v not in seen]
        return Index(result, name=self.name)

    def __repr__(self): return f"Index({self._data!r}, name={self.name!r})"
    def __str__(self): return self.__repr__()


class RangeIndex(Index):
    def __init__(self, start=0, stop=None, step=1, name=None):
        if stop is None:
            stop = start; start = 0
        super().__init__(list(range(start, stop, step)), name=name)
        self.start, self.stop, self.step = start, stop, step

    def __repr__(self):
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"
