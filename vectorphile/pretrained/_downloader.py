"""Weight download, caching, and format parsing (safetensors + pytorch .bin)."""
from __future__ import annotations

import os
import sys
import json
import struct
import pickle
import shutil
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_CACHE_DIR = Path.home() / '.vectorphile' / 'weights'

_HF_BASE = 'https://huggingface.co'
_HF_API  = 'https://huggingface.co/api/models'


# ── Cache helpers ──────────────────────────────────────────────────────────────

def get_cache_dir(cache_dir: Optional[str] = None) -> Path:
    d = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_cache_dir(hf_id: str, cache_dir: Optional[str] = None) -> Path:
    safe = hf_id.replace('/', '__')
    d = get_cache_dir(cache_dir) / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cached_path(hf_id: str, filename: str, cache_dir: Optional[str] = None) -> Path:
    return _model_cache_dir(hf_id, cache_dir) / filename


def is_cached(hf_id: str, cache_dir: Optional[str] = None) -> bool:
    model_dir = _model_cache_dir(hf_id, cache_dir)
    has_config = (model_dir / 'config.json').exists()
    has_weights = any(
        f.suffix in ('.safetensors', '.bin')
        for f in model_dir.iterdir()
    ) if model_dir.exists() and model_dir.is_dir() else False
    return has_config and has_weights


# ── Token resolution ───────────────────────────────────────────────────────────

def _resolve_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token
    for var in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN'):
        t = os.environ.get(var)
        if t:
            return t
    return None


# ── Progress bar ───────────────────────────────────────────────────────────────

def _progress_bar(filename: str, downloaded: int, total: int, speed: float) -> None:
    if total <= 0:
        sys.stdout.write(f'\rDownloading {filename}: {downloaded / 1e6:.1f} MB')
        sys.stdout.flush()
        return
    pct = downloaded / total
    bar_w = 20
    filled = int(bar_w * pct)
    bar = '█' * filled + '░' * (bar_w - filled)
    sys.stdout.write(
        f'\rDownloading {filename}: {pct*100:5.1f}% |{bar}|'
        f' {downloaded/1e6:.1f}/{total/1e6:.1f} MB  {speed/1e6:.1f} MB/s'
    )
    sys.stdout.flush()


# ── File downloader ────────────────────────────────────────────────────────────

def _download_file(
    url: str,
    dest: Path,
    token: Optional[str] = None,
    show_progress: bool = True,
    chunk_size: int = 1 << 20,
) -> None:
    headers = {'User-Agent': 'vectorphile/0.3.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            t0 = time.time()
            with open(tmp, 'wb') as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress:
                        elapsed = max(time.time() - t0, 1e-9)
                        _progress_bar(dest.name, downloaded, total, downloaded / elapsed)
        if show_progress:
            print()
        shutil.move(str(tmp), str(dest))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ── HuggingFace API ────────────────────────────────────────────────────────────

def _hf_api_files(hf_id: str, token: Optional[str], revision: str) -> List[dict]:
    url = f'{_HF_API}/{hf_id}'
    headers = {'User-Agent': 'vectorphile/0.3.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get('siblings', [])
    except Exception:
        return []


def _pick_weight_files(file_list: List[dict]) -> List[str]:
    names = [f['rfilename'] for f in file_list]
    safetensors = sorted(n for n in names
                         if n.endswith('.safetensors') and 'tokenizer' not in n)
    if safetensors:
        return safetensors
    bins = sorted(n for n in names if n.endswith('.bin') and 'tokenizer' not in n)
    return bins


# ── Main download entry point ──────────────────────────────────────────────────

def download_hf_model(
    hf_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = 'main',
    show_progress: bool = True,
) -> Path:
    """Download model from HuggingFace Hub to local cache. Returns model dir."""
    token = _resolve_token(token)
    model_dir = _model_cache_dir(hf_id, cache_dir)

    ancillary = [
        'config.json', 'tokenizer_config.json', 'tokenizer.json',
        'vocab.json', 'merges.txt', 'special_tokens_map.json',
        'generation_config.json',
    ]

    file_list = _hf_api_files(hf_id, token, revision)
    all_names = {f['rfilename'] for f in file_list}
    weight_files = _pick_weight_files(file_list)

    # Check for shard index files
    for idx_name, key in [
        ('model.safetensors.index.json', None),
        ('pytorch_model.bin.index.json', None),
    ]:
        if idx_name in all_names:
            idx_path = model_dir / idx_name
            if not idx_path.exists():
                _download_file(
                    f'{_HF_BASE}/{hf_id}/resolve/{revision}/{idx_name}',
                    idx_path, token, show_progress=False,
                )
            with open(idx_path) as fh:
                idx_data = json.load(fh)
            weight_files = sorted(set(idx_data.get('weight_map', {}).values()))
            break

    # Ancillary files
    for fname in ancillary:
        if fname not in all_names:
            continue
        dest = model_dir / fname
        if dest.exists():
            continue
        try:
            _download_file(
                f'{_HF_BASE}/{hf_id}/resolve/{revision}/{fname}',
                dest, token, show_progress=False,
            )
        except Exception:
            pass

    # Weight shards
    for fname in weight_files:
        dest = model_dir / fname
        if dest.exists():
            continue
        _download_file(
            f'{_HF_BASE}/{hf_id}/resolve/{revision}/{fname}',
            dest, token, show_progress,
        )

    return model_dir


# ── Safetensors parser ─────────────────────────────────────────────────────────

_ST_DTYPE_MAP = {
    'F64':  np.float64,
    'F32':  np.float32,
    'F16':  np.float16,
    'BF16': None,          # special handling
    'I64':  np.int64,
    'I32':  np.int32,
    'I16':  np.int16,
    'I8':   np.int8,
    'U8':   np.uint8,
    'BOOL': np.bool_,
}


def load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Parse .safetensors file → {name: float32 ndarray}."""
    with open(path, 'rb') as f:
        raw = f.read()

    header_len = struct.unpack_from('<Q', raw, 0)[0]
    header = json.loads(raw[8: 8 + header_len])
    data_start = 8 + header_len

    out: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        dtype_str = meta['dtype']
        shape     = meta['shape']
        begin, end = meta['data_offsets']
        blob = raw[data_start + begin: data_start + end]

        if dtype_str == 'BF16':
            u16  = np.frombuffer(blob, dtype=np.uint16)
            f32  = (u16.astype(np.uint32) << 16).view(np.float32)
            arr  = f32.reshape(shape) if shape else f32
        else:
            np_dtype = _ST_DTYPE_MAP.get(dtype_str, np.float32)
            arr = np.frombuffer(blob, dtype=np_dtype)
            arr = arr.reshape(shape) if shape else arr

        out[name] = arr.astype(np.float32)

    return out


# ── PyTorch .bin loader ────────────────────────────────────────────────────────

class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that converts torch storage objects to numpy without torch."""

    _STORAGE = {
        'FloatStorage':    np.float32,
        'HalfStorage':     np.float16,
        'BFloat16Storage': np.float32,
        'DoubleStorage':   np.float64,
        'LongStorage':     np.int64,
        'IntStorage':      np.int32,
        'ShortStorage':    np.int16,
        'CharStorage':     np.int8,
        'ByteStorage':     np.uint8,
    }

    def find_class(self, module, name):
        if module in ('torch', 'torch.storage') and name in self._STORAGE:
            dt = self._STORAGE[name]
            class _S:
                _dtype = dt
                def __init__(self, *a): pass
                def __reduce__(self): return (_S, ())
            return _S
        if module == 'collections' and name == 'OrderedDict':
            return dict
        try:
            return super().find_class(module, name)
        except Exception:
            return lambda *a, **k: None


def load_pytorch_bin(path: Path) -> Dict[str, np.ndarray]:
    """Load pytorch_model.bin → {name: float32 ndarray}."""
    try:
        import torch
        state = torch.load(str(path), map_location='cpu', weights_only=True)
        return {k: v.float().numpy() for k, v in state.items() if hasattr(v, 'numpy')}
    except Exception:
        pass
    with open(path, 'rb') as f:
        try:
            state = _SafeUnpickler(f).load()
        except Exception as exc:
            raise RuntimeError(f'Cannot load {path}: {exc}') from exc
    result: Dict[str, np.ndarray] = {}
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                result[k] = v.astype(np.float32)
    return result


# ── Shard merger ───────────────────────────────────────────────────────────────

def load_weight_shards(model_dir: Path) -> Dict[str, np.ndarray]:
    """Merge all weight shards in model_dir into one dict."""
    weights: Dict[str, np.ndarray] = {}
    shards = sorted(model_dir.glob('*.safetensors'))
    if not shards:
        shards = sorted(model_dir.glob('*.bin'))
    if not shards:
        raise FileNotFoundError(f'No weight files in {model_dir}')
    for shard in shards:
        chunk = load_safetensors(shard) if shard.suffix == '.safetensors' \
                else load_pytorch_bin(shard)
        weights.update(chunk)
    return weights


# ── Cache info ─────────────────────────────────────────────────────────────────

def cache_info_raw(cache_dir: Optional[str] = None) -> dict:
    root = get_cache_dir(cache_dir)
    models = []
    total = 0
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        size = sum(f.stat().st_size for f in sub.rglob('*') if f.is_file())
        total += size
        models.append({'name': sub.name.replace('__', '/'), 'size_mb': round(size / 1e6, 1)})
    return {'models': models, 'total_size_mb': round(total / 1e6, 1)}
