"""vectorphile.pretrained — Pretrained model hub.

Downloads weights from HuggingFace Hub in safetensors or pytorch .bin format,
maps them to vectorphile model parameter names, and returns a ready-to-use
vectorphile Module with no external dependencies beyond urllib (stdlib).

Supported models
----------------
GPT-2:       gpt2, gpt2-medium, gpt2-large, gpt2-xl
BERT:        bert-base-uncased, bert-base-cased, bert-large-uncased
             distilbert-base-uncased, roberta-base, roberta-large
T5:          t5-small, t5-base, t5-large, t5-xl
ViT:         vit-base-patch16-224, vit-large-patch16-224, vit-base-patch32-224
LLaMA 2/3:   llama-2-7b, llama-2-13b, llama-3-8b  (requires HF token)
Mistral:     mistral-7b

Usage::

    from vectorphile.pretrained import load, list_models, download, cache_info

    # List every available model
    for m in list_models():
        print(m['name'], '-', m['description'])

    # Load GPT-2 (auto-downloads on first call, cached afterwards)
    model = load('gpt2')
    output = model.forward(input_ids)

    # Load with explicit cache directory
    model = load('bert-base-uncased', cache_dir='~/my_weights')

    # Load from already-downloaded local directory
    model = load('gpt2', local_path='/path/to/gpt2')

    # Gated model (LLaMA) — pass HF token or set HF_TOKEN env var
    model = load('llama-2-7b', token='hf_...')

    # Pre-download without loading into memory
    download('gpt2')

    # Inspect local cache
    print(cache_info())

    # Remove cached weights
    clear_cache('gpt2')           # one model
    clear_cache()                 # everything
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from vectorphile.pretrained._registry       import REGISTRY
from vectorphile.pretrained._downloader     import (
    download_hf_model, is_cached, get_cache_dir,
    load_weight_shards, cache_info_raw, _model_cache_dir,
)
from vectorphile.pretrained._config_loader  import load_hf_config, hf_config_to_vectorphile
from vectorphile.pretrained._mappers        import get_mapper

__all__ = [
    'load', 'list_models', 'download', 'cache_info', 'clear_cache',
]


# ── Model factory ──────────────────────────────────────────────────────────────

def _build_model(model_class: str, config: dict):
    """Instantiate a vectorphile model from class name and config dict."""
    mc = model_class.upper()

    if mc == 'GPT2':
        from vectorphile.nn.architectures.transformer import GPT2
        return GPT2(**config)

    if mc == 'BERT':
        from vectorphile.nn.architectures.transformer import BERT
        return BERT(**config)

    if mc == 'T5':
        from vectorphile.nn.architectures.transformer import T5
        return T5(**config)

    if mc == 'LLAMA':
        from vectorphile.nn.architectures.transformer import LLaMA
        return LLaMA(**config)

    if mc == 'MISTRAL':
        from vectorphile.nn.architectures.transformer import Mistral
        return Mistral(**config)

    if mc == 'VIT':
        from vectorphile.nn.architectures.vision_transformer import ViT
        return ViT(**config)

    raise ValueError(
        f'Unknown model class: {model_class!r}. '
        f'Supported: GPT2, BERT, T5, LLaMA, Mistral, ViT'
    )


# ── Main public API ────────────────────────────────────────────────────────────

def load(
    model_name: str,
    cache_dir: Optional[str] = None,
    local_path: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = 'main',
    show_progress: bool = True,
    device: str = 'cpu',
):
    """Load a pretrained model with weights.

    Parameters
    ----------
    model_name : str
        Short name from the registry (e.g. ``'gpt2'``, ``'bert-base-uncased'``)
        OR a HuggingFace repo ID (e.g. ``'openai-community/gpt2'``).
    cache_dir : str, optional
        Local directory for caching downloaded weights.
        Defaults to ``~/.vectorphile/weights``.
    local_path : str, optional
        Path to a directory that already contains the model files.
        When given, no download is attempted.
    token : str, optional
        HuggingFace access token for gated models.
        Can also be set via the ``HF_TOKEN`` environment variable.
    revision : str
        Git revision / branch on HuggingFace Hub (default ``'main'``).
    show_progress : bool
        Display download progress bars (default True).
    device : str
        ``'cpu'`` (default) or ``'cuda:0'`` etc.  Requires the vectorphile
        backend to be set to ``'torch'`` or ``'cuda'`` for GPU placement.

    Returns
    -------
    vectorphile Module
        Model in eval mode with pretrained weights loaded.

    Examples
    --------
    >>> model = load('gpt2')
    >>> model = load('bert-base-uncased')
    >>> model = load('t5-small')
    >>> model = load('meta-llama/Llama-2-7b-hf', token='hf_...')
    >>> model = load('gpt2', local_path='/my/weights/gpt2')
    """
    # 1. Registry lookup
    entry = REGISTRY.get(model_name)
    if entry is None:
        # Treat as a raw HF repo id
        entry = {
            'hf_id': model_name,
            'model_class': None,   # will be inferred from config.json
            'config': {},
            'mapper': None,
            'description': model_name,
            'requires_auth': False,
        }

    hf_id        = entry['hf_id']
    model_class  = entry.get('model_class')
    reg_config   = dict(entry.get('config', {}))
    mapper_name  = entry.get('mapper')
    requires_auth = entry.get('requires_auth', False)

    if requires_auth and token is None and not os.environ.get('HF_TOKEN'):
        raise PermissionError(
            f'{model_name!r} is a gated model. '
            'Pass token=\'hf_...\' or set the HF_TOKEN environment variable. '
            'You must also accept the model licence at https://huggingface.co/'
            + hf_id
        )

    # 2. Resolve model directory
    if local_path:
        model_dir = Path(local_path).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(f'local_path does not exist: {model_dir}')
    else:
        if not is_cached(hf_id, cache_dir):
            print(f'[pretrained] Downloading {model_name} from HuggingFace Hub...')
            model_dir = download_hf_model(
                hf_id, cache_dir, token, revision, show_progress
            )
        else:
            model_dir = _model_cache_dir(hf_id, cache_dir)
            print(f'[pretrained] Loading {model_name} from cache: {model_dir}')

    # 3. Read config and derive vectorphile model kwargs
    hf_cfg = load_hf_config(model_dir)

    # Infer model class from config if not in registry
    if model_class is None:
        arch = hf_cfg.get('architectures', [''])[0].lower()
        if 'gpt2' in arch or 'gpt_neo' in arch:
            model_class = 'GPT2'
        elif 'bert' in arch or 'roberta' in arch:
            model_class = 'BERT'
        elif 't5' in arch:
            model_class = 'T5'
        elif 'llama' in arch:
            model_class = 'LLaMA'
        elif 'mistral' in arch:
            model_class = 'Mistral'
        elif 'vit' in arch:
            model_class = 'ViT'
        else:
            raise ValueError(
                f'Cannot infer model class for {model_name!r}. '
                f'HF architectures field: {hf_cfg.get("architectures")}. '
                'Pass model_class explicitly or use a registered model name.'
            )

    # Merge: registry config takes precedence over HF config (for overrides)
    hf_derived = hf_config_to_vectorphile(hf_cfg, model_class)
    final_config = {**hf_derived, **{k: v for k, v in reg_config.items() if v is not None}}

    # Infer mapper if not in registry
    if mapper_name is None:
        mapper_name = f'map_{model_class.lower()}_weights'

    # 4. Build model
    print(f'[pretrained] Building {model_class} with config: {final_config}')
    model = _build_model(model_class, final_config)

    # 5. Load weights
    print(f'[pretrained] Loading weights from {model_dir}...')
    weights = load_weight_shards(model_dir)
    print(f'[pretrained] Loaded {len(weights)} weight tensors')

    # 6. Map weights into model
    mapper_fn = get_mapper(mapper_name)
    model = mapper_fn(model, weights)

    # 7. Eval mode + optional device placement
    model.eval()

    if device != 'cpu':
        try:
            import vectorphile.backend as _backend
            _backend.to_device(model, device)
        except Exception:
            pass  # silently stay on CPU if backend can't move

    return model


def list_models(search: Optional[str] = None) -> List[dict]:
    """Return info dicts for all registered models.

    Parameters
    ----------
    search : str, optional
        Case-insensitive filter applied to both model name and description.

    Returns
    -------
    list of dict
        Each dict has keys: ``name``, ``description``, ``size_mb``,
        ``model_class``, ``hf_id``, ``requires_auth``.
    """
    results = []
    seen_hf_ids = set()
    for name, entry in REGISTRY.items():
        hf_id = entry['hf_id']
        if hf_id in seen_hf_ids:
            continue
        seen_hf_ids.add(hf_id)
        info = {
            'name':          name,
            'description':   entry.get('description', ''),
            'size_mb':       entry.get('size_mb', None),
            'model_class':   entry.get('model_class', ''),
            'hf_id':         hf_id,
            'requires_auth': entry.get('requires_auth', False),
        }
        if search:
            q = search.lower()
            if q not in name.lower() and q not in info['description'].lower():
                continue
        results.append(info)
    return results


def download(
    model_name: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = 'main',
    show_progress: bool = True,
) -> Path:
    """Download model weights to local cache without loading into memory.

    Returns the path to the local model directory.
    """
    entry = REGISTRY.get(model_name, {'hf_id': model_name, 'requires_auth': False})
    hf_id = entry['hf_id']
    if entry.get('requires_auth') and token is None and not os.environ.get('HF_TOKEN'):
        raise PermissionError(
            f'{model_name!r} requires an HF token. '
            'Pass token=\'hf_...\' or set HF_TOKEN.'
        )
    return download_hf_model(hf_id, cache_dir, token, revision, show_progress)


def cache_info(cache_dir: Optional[str] = None) -> dict:
    """Return information about locally cached models.

    Returns
    -------
    dict
        ``{'models': [{'name': str, 'size_mb': float}, ...], 'total_size_mb': float}``
    """
    return cache_info_raw(cache_dir)


def clear_cache(
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """Remove cached weights.

    Parameters
    ----------
    model_name : str, optional
        Name or HF repo id to remove. If None, removes everything.
    cache_dir : str, optional
        Cache directory to operate on (default ``~/.vectorphile/weights``).
    """
    root = get_cache_dir(cache_dir)
    if model_name is None:
        shutil.rmtree(str(root), ignore_errors=True)
        print(f'[pretrained] Cleared entire cache at {root}')
        return
    hf_id = REGISTRY.get(model_name, {}).get('hf_id', model_name)
    model_dir = _model_cache_dir(hf_id, cache_dir)
    if model_dir.exists():
        shutil.rmtree(str(model_dir))
        print(f'[pretrained] Removed {model_dir}')
    else:
        print(f'[pretrained] Nothing cached for {model_name!r}')
