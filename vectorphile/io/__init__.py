"""File I/O — MAT files, Parquet, HDF5, WAV, Arrow formats."""
from __future__ import annotations
import struct
import json
import pickle
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
        return _wavfile.read(filename)
    def wavfile_write(filename, rate, data):
        _wavfile.write(filename, rate, data)
except ImportError:
    def _wav_read_pure(filename):
        """Pure Python PCM WAV reader (16-bit or 8-bit, mono or stereo)."""
        import wave, array as _arr
        with wave.open(str(filename), 'rb') as wf:
            rate = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        fmt = {1: 'b', 2: 'h', 4: 'l'}.get(sw, 'h')
        samples = _arr.array(fmt, raw)
        samples = list(samples)
        if n_ch > 1:
            channels = [samples[i::n_ch] for i in range(n_ch)]
            return rate, [list(ch) for ch in channels]
        return rate, samples

    def _wav_write_pure(filename, rate, data):
        """Pure Python PCM WAV writer (16-bit)."""
        import wave, array as _arr
        if isinstance(data[0], (list, tuple)):
            n_ch = len(data)
            n_frames = len(data[0])
            interleaved = []
            for i in range(n_frames):
                for ch in data:
                    interleaved.append(int(max(-32768, min(32767, ch[i]))))
        else:
            n_ch = 1
            n_frames = len(data)
            interleaved = [int(max(-32768, min(32767, v))) for v in data]
        samples = _arr.array('h', interleaved)
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(2)
            wf.setframerate(int(rate))
            wf.writeframes(samples.tobytes())

    def wavfile_read(filename): return _wav_read_pure(filename)
    def wavfile_write(filename, rate, data): return _wav_write_pure(filename, rate, data)

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
                return list(f[key])
            return {k: list(f[k]) for k in f.keys()}

    def write_hdf(filename, data, key='data', mode='a', **kwargs):
        with _h5py.File(filename, mode) as f:
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in f: del f[k]
                    f.create_dataset(k, data=v, **kwargs)
            else:
                if key in f: del f[key]
                f.create_dataset(key, data=data, **kwargs)

except ImportError:
    def read_hdf(*a, **k): raise ImportError("h5py required for HDF5 I/O. pip install h5py")
    def write_hdf(*a, **k): raise ImportError("h5py required for HDF5 I/O. pip install h5py")

# ── NetCDF4 ───────────────────────────────────────────────────────────────────
try:
    import netCDF4 as _nc4

    def read_netcdf(filename, variables=None, **kwargs):
        ds = _nc4.Dataset(filename, 'r', **kwargs)
        if variables:
            result = {v: list(ds.variables[v][:]) for v in variables}
        else:
            result = {k: list(ds.variables[k][:]) for k in ds.variables.keys()}
        ds.close()
        return result

    def write_netcdf(filename, data, dimensions=None, **kwargs):
        ds = _nc4.Dataset(filename, 'w', **kwargs)
        if dimensions:
            for name, size in dimensions.items():
                ds.createDimension(name, size)
        for name, arr in data.items():
            if hasattr(arr, 'shape'):
                shape = arr.shape
                dims = tuple(f'dim{i}' for i in range(len(shape)))
            else:
                flat = list(arr) if not isinstance(arr, list) else arr
                shape = (len(flat),)
                dims = ('dim0',)
            for d, s in zip(dims, shape):
                if d not in ds.dimensions:
                    ds.createDimension(d, s)
            var = ds.createVariable(name, 'f4', dims)
            var[:] = arr
        ds.close()

except ImportError:
    def read_netcdf(*a, **k): raise ImportError("netCDF4 required. pip install netCDF4")
    def write_netcdf(*a, **k): raise ImportError("netCDF4 required. pip install netCDF4")

# ── Pure-Python array serialization (replaces numpy save/load) ────────────────

def _to_flat(arr) -> tuple:
    """Return (flat_list, shape) for any array-like."""
    if hasattr(arr, '_data') and hasattr(arr._data, 'to_flat_list'):
        return arr._data.to_flat_list(), arr._data.shape
    if hasattr(arr, 'to_flat_list'):
        return arr.to_flat_list(), getattr(arr, 'shape', (len(arr.to_flat_list()),))
    if isinstance(arr, list):
        flat = []
        def _r(v):
            if isinstance(v, list):
                for i in v: _r(i)
            else: flat.append(float(v))
        _r(arr); return flat, (len(arr),)
    return [float(arr)], (1,)


def save(file, arr, **kwargs):
    """Save a single array to a .vpa file (vectorphile format, JSON-based)."""
    flat, shape = _to_flat(arr)
    with open(str(file), 'w') as f:
        json.dump({'shape': list(shape), 'data': flat}, f)


def load(file, **kwargs) -> list:
    """Load from a .vpa file. Returns flat list."""
    path = str(file)
    if path.endswith('.json') or path.endswith('.vpa'):
        with open(path) as f:
            obj = json.load(f)
        return obj.get('data', obj)
    with open(path, 'rb') as f:
        return pickle.load(f)


def savez(file, *args, **kwds):
    """Save multiple arrays to a JSON archive (.vpz)."""
    arrays = {}
    for i, a in enumerate(args):
        flat, shape = _to_flat(a)
        arrays[f'arr_{i}'] = {'shape': list(shape), 'data': flat}
    for k, v in kwds.items():
        flat, shape = _to_flat(v)
        arrays[k] = {'shape': list(shape), 'data': flat}
    with open(str(file), 'w') as f:
        json.dump(arrays, f)


savez_compressed = savez  # same format


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """Save array to text file."""
    flat, shape = _to_flat(X)
    rows = shape[0] if len(shape) > 1 else 1
    cols = shape[1] if len(shape) > 1 else len(flat)
    lines = []
    if header:
        lines.append(comments + header)
    for r in range(rows):
        row_data = flat[r*cols:(r+1)*cols]
        lines.append(delimiter.join(fmt % v for v in row_data))
    if footer:
        lines.append(comments + footer)
    text = newline.join(lines) + newline
    mode = 'w' if encoding is None else 'w'
    with open(fname, mode, encoding=encoding or 'utf-8') as f:
        f.write(text)


def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None):
    """Load array from text file."""
    rows = []
    with open(fname, 'r', encoding='utf-8' if encoding == 'bytes' else encoding) as f:
        for i, line in enumerate(f):
            if i < skiprows: continue
            line = line.strip()
            if not line: continue
            if comments and line.startswith(comments): continue
            if max_rows is not None and len(rows) >= max_rows: break
            vals = line.split(delimiter) if delimiter else line.split()
            if usecols is not None:
                vals = [vals[c] for c in usecols]
            if converters:
                row = [converters.get(j, dtype)(v) for j, v in enumerate(vals)]
            else:
                row = [dtype(v) for v in vals]
            rows.append(row)
    if not rows: return []
    if len(rows[0]) == 1:
        return [r[0] for r in rows]
    return rows


def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, converters=None, missing_values=None, filling_values=None,
               usecols=None, names=None, **kwargs):
    """Load data from text file, handling missing values."""
    fill = filling_values if filling_values is not None else 0.0
    rows = []
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = lines[skip_header:len(lines)-skip_footer if skip_footer else None]
    for line in lines:
        line = line.strip()
        if not line: continue
        if comments and line.startswith(comments): continue
        vals = line.split(delimiter) if delimiter else line.split()
        if usecols is not None:
            vals = [vals[c] for c in usecols]
        row = []
        for j, v in enumerate(vals):
            v = v.strip()
            if missing_values and v in (missing_values if isinstance(missing_values,(list,tuple)) else [missing_values]):
                row.append(fill)
            else:
                try:
                    row.append(dtype(v))
                except (ValueError, TypeError):
                    row.append(fill)
        rows.append(row)
    return rows
