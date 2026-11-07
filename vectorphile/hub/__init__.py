"""Model Hub — save, load, and share vectorphile models."""
from __future__ import annotations
import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

_HUB_DIR = Path.home() / '.vectorphile' / 'hub'


def _ensure_hub():
    _HUB_DIR.mkdir(parents=True, exist_ok=True)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model: Any, name: str, version: str = '1.0.0',
               metadata: Optional[Dict] = None, path: Optional[str] = None) -> str:
    """Save any vectorphile model to disk.

    Returns the saved path.
    """
    _ensure_hub()
    save_dir = Path(path) if path else _HUB_DIR / name / version
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    meta = {
        'name': name,
        'version': version,
        'class': type(model).__name__,
        'module': type(model).__module__,
        **(metadata or {}),
    }
    # Save numpy arrays separately if model has params
    if hasattr(model, 'parameters'):
        params = {}
        for i, p in enumerate(model.parameters()):
            params[f'param_{i}'] = p._data if hasattr(p, '_data') else np.array(p)
        if params:
            np.savez_compressed(save_dir / 'params.npz', **params)
            meta['has_npz_params'] = True

    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    checksum = _file_hash(model_path)
    meta['checksum'] = checksum
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return str(model_path)


def load_model(name: str, version: str = '1.0.0', path: Optional[str] = None) -> Any:
    """Load a vectorphile model from disk."""
    load_dir = Path(path) if path else _HUB_DIR / name / version
    model_path = load_dir / 'model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. "
                                f"Save it first with vp.hub.save_model()")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_weights(model: Any, path: str) -> None:
    """Save only the weights/parameters of a model as .npz."""
    if not hasattr(model, 'parameters'):
        raise AttributeError("Model does not have a parameters() method")
    params = {}
    for i, p in enumerate(model.parameters()):
        params[f'param_{i}'] = p._data if hasattr(p, '_data') else np.array(p)
    np.savez_compressed(path, **params)


def load_weights(model: Any, path: str) -> None:
    """Load weights into an existing model (in-place)."""
    if not hasattr(model, 'parameters'):
        raise AttributeError("Model does not have a parameters() method")
    data = np.load(path)
    params = list(model.parameters())
    for i, p in enumerate(params):
        key = f'param_{i}'
        if key in data and hasattr(p, '_data'):
            p._data[:] = data[key]


# ── Registry / listing ────────────────────────────────────────────────────────

def list_local_models() -> list:
    """List all models saved in the local hub."""
    _ensure_hub()
    models = []
    for name_dir in _HUB_DIR.iterdir():
        if name_dir.is_dir():
            for ver_dir in name_dir.iterdir():
                meta_file = ver_dir / 'metadata.json'
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    models.append(meta)
    return models


def delete_model(name: str, version: str = '1.0.0') -> None:
    """Remove a model from the local hub."""
    import shutil
    target = _HUB_DIR / name / version
    if target.exists():
        shutil.rmtree(target)
        print(f"Deleted {name}@{version}")
    else:
        raise FileNotFoundError(f"Model {name}@{version} not found in local hub")


# ── Checkpoint system ─────────────────────────────────────────────────────────

class Checkpointer:
    """Manage training checkpoints — save best model, resume training."""

    def __init__(self, dirpath: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_top_k: int = 3, verbose: bool = True):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose
        self._scores: list = []
        self._best = None

    def __call__(self, model: Any, epoch: int, metrics: Dict) -> bool:
        """Call after each epoch. Returns True if this was the best checkpoint."""
        score = metrics.get(self.monitor)
        if score is None:
            return False
        is_better = (self._best is None or
                     (self.mode == 'min' and score < self._best) or
                     (self.mode == 'max' and score > self._best))
        path = self.dirpath / f'epoch_{epoch:04d}_{self.monitor}_{score:.4f}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        self._scores.append((score, path, epoch))
        if is_better:
            self._best = score
            best_path = self.dirpath / 'best.pkl'
            with open(best_path, 'wb') as f:
                pickle.dump(model, f, protocol=4)
            if self.verbose:
                print(f"[Checkpoint] Epoch {epoch}: {self.monitor}={score:.4f} (new best)")
        # Prune old checkpoints
        if self.mode == 'min':
            self._scores.sort(key=lambda x: x[0])
        else:
            self._scores.sort(key=lambda x: x[0], reverse=True)
        while len(self._scores) > self.save_top_k:
            _, old_path, _ = self._scores.pop()
            if old_path.exists() and old_path.name != 'best.pkl':
                old_path.unlink()
        return is_better

    def load_best(self) -> Any:
        best_path = self.dirpath / 'best.pkl'
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found. Run training first.")
        with open(best_path, 'rb') as f:
            return pickle.load(f)

    def load_latest(self) -> Any:
        if not self._scores:
            raise FileNotFoundError("No checkpoints recorded in this session.")
        latest_path = max(self.dirpath.glob('epoch_*.pkl'),
                          key=lambda p: p.stat().st_mtime)
        with open(latest_path, 'rb') as f:
            return pickle.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def model_card(model: Any, name: str = '', description: str = '',
               metrics: Optional[Dict] = None) -> str:
    """Generate a text model card for documentation."""
    lines = [
        f"# Model Card: {name or type(model).__name__}",
        f"\n**Class:** `{type(model).__module__}.{type(model).__name__}`",
    ]
    if description:
        lines.append(f"\n## Description\n{description}")
    if hasattr(model, 'parameters'):
        total = sum(p._data.size if hasattr(p, '_data') else np.array(p).size
                    for p in model.parameters())
        lines.append(f"\n**Parameters:** {total:,}")
    if metrics:
        lines.append("\n## Metrics")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
    return '\n'.join(lines)
