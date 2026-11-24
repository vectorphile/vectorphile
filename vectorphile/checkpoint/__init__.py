"""vectorphile.checkpoint — Model checkpoint saving and loading.

Provides PyTorch-compatible save/load for models, optimizers, and training state.
All serialization uses Python's built-in pickle + json — no third-party deps.

Usage::

    import vectorphile as vp
    from vectorphile.checkpoint import save, load, ModelCheckpoint

    # Save model
    vp.checkpoint.save(model, 'model.vp')

    # Load model
    state = vp.checkpoint.load('model.vp')
    model.load_state_dict(state['model'])

    # Callback-style auto-checkpointing
    cb = ModelCheckpoint('checkpoints/', monitor='val_loss', save_best_only=True)
"""
from __future__ import annotations
import os
import pickle
import json
import time
from typing import Any, Dict, Optional, Union


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------

def save(obj: Any, path: str, metadata: Optional[dict] = None) -> None:
    """Save any object (model, optimizer, dict) to a file using pickle.

    Args:
        obj: Any picklable Python object (nn.Module, optimizer, state_dict).
        path: Destination file path (conventionally *.vp or *.pkl).
        metadata: Optional dict of extra metadata saved alongside.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'obj': obj,
        'metadata': metadata or {},
        'timestamp': time.time(),
        'version': '0.5.0',
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> Dict[str, Any]:
    """Load a checkpoint saved with save().

    Returns a dict with keys:
        obj       — the saved object
        metadata  — extra metadata dict
        timestamp — save time (float)
        version   — library version string
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_state_dict(model, path: str, optimizer=None, epoch: int = 0,
                    extra: Optional[dict] = None) -> None:
    """Save model (and optionally optimizer) state dict."""
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'extra': extra or {},
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    save(state, path)


def load_state_dict(path: str) -> dict:
    """Load a state dict checkpoint. Returns the inner dict."""
    return load(path)['obj']


# ---------------------------------------------------------------------------
# ModelCheckpoint callback
# ---------------------------------------------------------------------------

class ModelCheckpoint:
    """Saves the best (or every N epochs) model checkpoint.

    Compatible with the vectorphile Trainer callbacks API.

    Usage::

        cb = ModelCheckpoint(
            dirpath='checkpoints/',
            monitor='val_loss',
            save_best_only=True,
            mode='min',          # 'min' or 'max'
            save_every_n=1,
            verbose=True,
        )
        trainer = Trainer(callbacks=[cb])
    """

    def __init__(self, dirpath: str = 'checkpoints/', monitor: str = 'val_loss',
                 save_best_only: bool = True, mode: str = 'min',
                 save_every_n: int = 1, verbose: bool = False,
                 filename: str = 'epoch{epoch}-{monitor:.4f}.vp'):
        self.dirpath = dirpath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_every_n = save_every_n
        self.verbose = verbose
        self.filename = filename
        self.best_score = None
        self.best_path = None
        self._is_better = (lambda a, b: a < b) if mode == 'min' else (lambda a, b: a > b)
        os.makedirs(dirpath, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None,
                     model=None, optimizer=None) -> bool:
        """Called after each epoch. Returns True if checkpoint was saved."""
        logs = logs or {}
        score = logs.get(self.monitor)

        if epoch % self.save_every_n != 0:
            return False

        should_save = True
        if self.save_best_only and score is not None:
            if self.best_score is None or self._is_better(score, self.best_score):
                self.best_score = score
            else:
                should_save = False

        if should_save and model is not None:
            monitor_val = score if score is not None else 0.0
            fname = self.filename.replace('{epoch}', str(epoch))
            fname = fname.replace('{monitor:.4f}', f'{monitor_val:.4f}')
            path = os.path.join(self.dirpath, fname)
            save_state_dict(model, path, optimizer=optimizer, epoch=epoch,
                            extra={'logs': logs})
            self.best_path = path
            if self.verbose:
                print(f'[ModelCheckpoint] Saved epoch {epoch} → {path}')
            return True
        return False

    def restore_best(self, model, optimizer=None) -> dict:
        """Restore weights from the best saved checkpoint."""
        if self.best_path is None:
            raise RuntimeError('No checkpoint saved yet')
        state = load_state_dict(self.best_path)
        model.load_state_dict(state['model'])
        if optimizer is not None and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        return state


# ---------------------------------------------------------------------------
# EarlyStopping callback
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a metric stops improving.

    Usage::

        es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        for epoch in range(100):
            ...
            if es.step({'val_loss': val_loss}):
                break
    """

    def __init__(self, monitor: str = 'val_loss', patience: int = 5,
                 mode: str = 'min', min_delta: float = 1e-4,
                 restore_best_weights: bool = False, verbose: bool = False):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self._is_better = (lambda a, b: a < b - min_delta) if mode == 'min' \
                          else (lambda a, b: a > b + min_delta)

    def step(self, logs: dict) -> bool:
        """Returns True if training should stop."""
        score = logs.get(self.monitor)
        if score is None:
            return False
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs')
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'[EarlyStopping] Stopping at best={self.best_score:.4f}')
                return True
        return False

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None, **kwargs) -> bool:
        return self.step(logs or {})


# ---------------------------------------------------------------------------
# LR scheduling callbacks
# ---------------------------------------------------------------------------

class LRSchedulerCallback:
    """Wraps any vectorphile.optim scheduler as a callback."""

    def __init__(self, scheduler, monitor: str = 'val_loss', verbose: bool = False):
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None, **kwargs):
        metric = (logs or {}).get(self.monitor)
        if hasattr(self.scheduler, 'step'):
            if metric is not None:
                try:
                    self.scheduler.step(metric)
                except TypeError:
                    self.scheduler.step()
            else:
                self.scheduler.step()
        if self.verbose:
            print(f'[LRScheduler] lr updated at epoch {epoch}')


__all__ = [
    'save', 'load', 'save_state_dict', 'load_state_dict',
    'ModelCheckpoint', 'EarlyStopping', 'LRSchedulerCallback',
]
