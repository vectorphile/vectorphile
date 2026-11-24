"""Numpy-compatible array wrapper for ML outputs — returns real numpy when available."""


def to_array(data):
    """Convert a nested list/flat list to numpy array if numpy is available, else return list."""
    try:
        import numpy as _np
        return _np.array(data)
    except ImportError:
        return data
