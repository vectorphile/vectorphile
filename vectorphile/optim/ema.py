"""Exponential Moving Average of model weights.

EMA maintains a shadow copy of parameters: shadow = decay * shadow + (1 - decay) * param
Widely used in diffusion models (DDPM, Stable Diffusion), YOLO, SimCLR.
"""
from __future__ import annotations
from typing import Optional


class EMA:
    """Maintains an exponential moving average of model parameters.

    Args:
        model       : the Module whose parameters to track
        decay       : smoothing coefficient (default 0.9999 — suitable for large models)
        warmup_steps: use a cosine-ramp warmup schedule instead of fixed decay
                      for the first `warmup_steps` updates so early noise doesn't
                      corrupt the shadow.  Set 0 to disable.
        device      : currently ignored (kept for API compat)

    Usage::
        ema = EMA(model, decay=0.9999)
        for step, (x, y) in enumerate(train_loader):
            loss = model(x) ...
            optimizer.step()
            ema.update(model)          # call every step AFTER optimizer.step()

        # for evaluation
        ema.apply_shadow(model)        # swap EMA weights into model
        val_loss = model(x_val) ...
        ema.restore(model)             # restore training weights
    """

    def __init__(self, model, decay: float = 0.9999,
                 warmup_steps: int = 0, device=None):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._shadow: dict = {}
        self._backup: dict = {}

        for name, p in model.named_parameters():
            if hasattr(p, '_data'):
                try:
                    self._shadow[name] = _copy_data(p._data)
                except Exception:
                    pass

    # ── internal ──────────────────────────────────────────────────────────────

    def _effective_decay(self) -> float:
        if self.warmup_steps > 0:
            # linear ramp from 0 to self.decay over warmup_steps
            t = min(self._step, self.warmup_steps)
            return self.decay * (t / max(1, self.warmup_steps))
        return self.decay

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, model) -> None:
        """Update EMA shadow parameters from current model weights."""
        self._step += 1
        d = self._effective_decay()
        for name, p in model.named_parameters():
            if name not in self._shadow or not hasattr(p, '_data'):
                continue
            self._shadow[name] = d * self._shadow[name] + (1.0 - d) * p._data

    def apply_shadow(self, model) -> None:
        """Swap EMA weights into model for evaluation.  Call restore() afterwards."""
        self._backup.clear()
        for name, p in model.named_parameters():
            if name not in self._shadow or not hasattr(p, '_data'):
                continue
            self._backup[name] = _copy_data(p._data)
            p._data = _copy_data(self._shadow[name])

    def restore(self, model) -> None:
        """Restore original training weights after apply_shadow()."""
        for name, p in model.named_parameters():
            if name in self._backup and hasattr(p, '_data'):
                p._data = self._backup[name]
        self._backup.clear()

    def state_dict(self) -> dict:
        return {'step': self._step, 'decay': self.decay,
                'shadow': {k: _copy_data(v) for k, v in self._shadow.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd['step']
        self.decay = sd['decay']
        self._shadow = {k: _copy_data(v) for k, v in sd['shadow'].items()}

    def __repr__(self) -> str:
        return (f'EMA(decay={self.decay}, warmup_steps={self.warmup_steps}, '
                f'step={self._step}, params={len(self._shadow)})')


def _copy_data(data):
    """Return a copy of an NDArray or numpy array."""
    try:
        return data.copy()
    except AttributeError:
        import copy
        return copy.deepcopy(data)
