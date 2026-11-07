"""File I/O — MAT files, Parquet, HDF5, WAV, Arrow formats."""
from __future__ import annotations
import numpy as np
from pathlib import Path

# ── MAT files (scipy.io) ──────────────────────────────────────────────────────
try:
    from scipy.io import loadmat as _loadmat, savemat as _savemat
    def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
        return _loadmat(file_name, mdict=mdict, appendmat=appendmat, **kwargs)
    def savemat(file_name, mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row'):
        return _savemat(file_name, mdict, appendmat=appendmat, format=format,
                        long_field_names=long_field_names, do_compression=do_compression, oned_as=oned_as)
except ImportError:
    def loadmat(*args, **kwargs): raise ImportError("scipy required for loadmat")
    def savemat(*args, **kwargs): raise ImportError("scipy required for savemat")

# ── WAV files ─────────────────────────────────────────────────────────────────
try:
    from scipy.io import wavfile as _wavfile
    def wavfile_read(filename):
        rate, data = _wavfile.read(filename)
        return rate, data
    def wavfile_write(filename, rate, data):
        _wavfile.write(filename, rate, data)
except ImportError:
    def wavfile_read(*a, **k): raise ImportError("scipy required for WAV I/O")
    def wavfile_write(*a, **k): raise ImportError("scipy required for WAV I/O")

# ── Matrix Market format ──────────────────────────────────────────────────────
try:
    from scipy.io import mmread as _mmread, mmwrite as _mmwrite, mminfo as _mminfo
    mmread = _mmread
    mmwrite = _mmwrite
    mminfo = _mminfo
except ImportError:
    def mmread(*a, **k): raise ImportError("scipy required for mmread")
    def mmwrite(*a, **k): raise ImportError("scipy required for mmwrite")
    def mminfo(*a, **k): raise ImportError("scipy required for mminfo")

# ── Parquet / Arrow ──────────────────────────────────────────────────────────
try:
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import pyarrow.feather as _feather

    def read_parquet(source, columns=None, use_threads=True, **kwargs):
        table = _pq.read_table(source, columns=columns, use_threads=use_threads, **kwargs)
        return table.to_pydict()

    def write_parquet(table_dict, where, compression='snappy', **kwargs):
        if hasattr(table_dict, 'to_dict'):
            # DataFrame-like
            arrays = {k: _pa.array(v) for k, v in table_dict.items()}
        else:
            arrays = {k: _pa.array(v) for k, v in table_dict.items()}
        table = _pa.table(arrays)
        _pq.write_table(table, where, compression=compression, **kwargs)

    def read_feather(source, columns=None, use_threads=True, **kwargs):
        table = _feather.read_table(source, columns=columns, use_threads=use_threads, **kwargs)
        return table.to_pydict()

    def write_feather(table_dict, dest, compression='lz4', **kwargs):
        arrays = {k: _pa.array(v) for k, v in table_dict.items()}
        table = _pa.table(arrays)
        _feather.write_feather(table, dest, compression=compression, **kwargs)

    def read_csv_arrow(input_file, **kwargs):
        from pyarrow import csv as _csv
        table = _csv.read_csv(input_file, **kwargs)
        return table.to_pydict()

except ImportError:
    def read_parquet(*a, **k): raise ImportError("pyarrow required for parquet I/O. pip install pyarrow")
    def write_parquet(*a, **k): raise ImportError("pyarrow required for parquet I/O. pip install pyarrow")
    def read_feather(*a, **k): raise ImportError("pyarrow required for feather I/O. pip install pyarrow")
    def write_feather(*a, **k): raise ImportError("pyarrow required for feather I/O. pip install pyarrow")
    def read_csv_arrow(*a, **k): raise ImportError("pyarrow required. pip install pyarrow")

# ── HDF5 ─────────────────────────────────────────────────────────────────────
try:
    import h5py as _h5py

    def read_hdf(filename, key=None, **kwargs):
        with _h5py.File(filename, 'r') as f:
            if key:
                return np.array(f[key])
            return {k: np.array(f[k]) for k in f.keys()}

    def write_hdf(filename, data, key='data', mode='a', **kwargs):
        with _h5py.File(filename, mode) as f:
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in f:
                        del f[k]
                    f.create_dataset(k, data=np.array(v), **kwargs)
            else:
                if key in f:
                    del f[key]
                f.create_dataset(key, data=np.array(data), **kwargs)

except ImportError:
    def read_hdf(*a, **k): raise ImportError("h5py required for HDF5 I/O. pip install h5py")
    def write_hdf(*a, **k): raise ImportError("h5py required for HDF5 I/O. pip install h5py")

# ── NetCDF4 ───────────────────────────────────────────────────────────────────
try:
    import netCDF4 as _nc4

    def read_netcdf(filename, variables=None, **kwargs):
        ds = _nc4.Dataset(filename, 'r', **kwargs)
        if variables:
            result = {v: np.array(ds.variables[v][:]) for v in variables}
        else:
            result = {k: np.array(ds.variables[k][:]) for k in ds.variables.keys()}
        ds.close()
        return result

    def write_netcdf(filename, data, dimensions=None, **kwargs):
        ds = _nc4.Dataset(filename, 'w', **kwargs)
        if dimensions:
            for name, size in dimensions.items():
                ds.createDimension(name, size)
        for name, arr in data.items():
            arr = np.array(arr)
            dims = tuple(f'dim{i}' for i in range(arr.ndim))
            for d, s in zip(dims, arr.shape):
                if d not in ds.dimensions:
                    ds.createDimension(d, s)
            var = ds.createVariable(name, arr.dtype, dims)
            var[:] = arr
        ds.close()

except ImportError:
    def read_netcdf(*a, **k): raise ImportError("netCDF4 required. pip install netCDF4")
    def write_netcdf(*a, **k): raise ImportError("netCDF4 required. pip install netCDF4")

# ── Pure-NumPy formats ────────────────────────────────────────────────────────
def save(file, arr, allow_pickle=True, fix_imports=True):
    """Save a single array to .npy file."""
    np.save(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

def load(file, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """Load from .npy or .npz file."""
    return np.load(file, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

def savez(file, *args, **kwds):
    """Save multiple arrays to .npz file."""
    np.savez(file, *args, **kwds)

def savez_compressed(file, *args, **kwds):
    """Save multiple arrays to compressed .npz file."""
    np.savez_compressed(file, *args, **kwds)

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """Save array to text file."""
    np.savetxt(fname, X, fmt=fmt, delimiter=delimiter, newline=newline,
               header=header, footer=footer, comments=comments, encoding=encoding)

def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None):
    """Load array from text file."""
    return np.loadtxt(fname, dtype=dtype, comments=comments, delimiter=delimiter,
                      converters=converters, skiprows=skiprows, usecols=usecols,
                      unpack=unpack, ndmin=ndmin, encoding=encoding, max_rows=max_rows)

def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, converters=None, missing_values=None, filling_values=None,
               usecols=None, names=None, **kwargs):
    """Load data from text file, handling missing values."""
    return np.genfromtxt(fname, dtype=dtype, comments=comments, delimiter=delimiter,
                         skip_header=skip_header, skip_footer=skip_footer,
                         converters=converters, missing_values=missing_values,
                         filling_values=filling_values, usecols=usecols, names=names, **kwargs)
