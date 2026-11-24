"""Keras-compatible Model API for vectorphile.

Provides Model.compile() / .fit() / .predict() / .evaluate() — the same
high-level interface as tf.keras.Model or torch.nn.Module with Lightning.

Usage::

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(784, 10)
        def forward(self, x):
            return self.dense(x)

    model = MyModel()
    model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test)
    preds = model.predict(x_test)
"""
from __future__ import annotations
import math
import time
from typing import Any, Callable, Dict, List, Optional, Union

from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _to_tensor(x, dtype=None):
    if isinstance(x, Tensor):
        return x
    return Tensor(_np.asarray(x, dtype=dtype or 'float32'))


def _to_np(x):
    if isinstance(x, Tensor):
        return _np.asarray(x._data)
    return _np.asarray(x)


class History:
    """Training history object — holds epoch metrics."""
    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def _append(self, key, val):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(val)

    def __repr__(self):
        lines = ['History:']
        for k, v in self.history.items():
            lines.append(f'  {k}: {[round(x, 4) for x in v]}')
        return '\n'.join(lines)


class Model(Module):
    """Keras-compatible model base class.

    Subclass this and implement forward(). Then call compile() + fit().
    """

    def __init__(self):
        super().__init__()
        self._optimizer = None
        self._loss_fn = None
        self._metrics: List[str] = []
        self._compiled = False
        self._stop_training = False
        self._callbacks: List = []

    # ── compile ───────────────────────────────────────────────────────────────

    def compile(self,
                optimizer='adam',
                loss='mse',
                metrics: Optional[List[str]] = None,
                lr: float = 1e-3,
                **opt_kwargs):
        """Configure the model for training.

        Args:
            optimizer: 'adam', 'sgd', 'adamw', 'rmsprop', 'adagrad', or an
                       optimizer instance from vp.optim.
            loss:      'mse', 'mae', 'crossentropy', 'bce', 'huber', or a callable.
            metrics:   list of metric names, e.g. ['accuracy', 'mse'].
            lr:        learning rate (passed to optimizer if string).
        """
        import vectorphile.optim as _optim
        import vectorphile.nn as _nn
        # Build optimizer
        params = list(self.parameters())
        if isinstance(optimizer, str):
            opt_map = {
                'adam': _optim.Adam, 'adamw': _optim.AdamW, 'sgd': _optim.SGD,
                'rmsprop': _optim.RMSprop, 'adagrad': _optim.Adagrad,
                'adadelta': _optim.Adadelta, 'nadam': _optim.NAdam,
                'lion': _optim.Lion if hasattr(_optim, 'Lion') else _optim.Adam,
            }
            opt_cls = opt_map.get(optimizer.lower(), _optim.Adam)
            self._optimizer = opt_cls(params, lr=lr, **opt_kwargs)
        else:
            self._optimizer = optimizer
        # Build loss
        if isinstance(loss, str):
            loss_map = {
                'mse': _nn.MSELoss(), 'mae': _nn.MAELoss(),
                'crossentropy': _nn.CrossEntropyLoss(),
                'cross_entropy': _nn.CrossEntropyLoss(),
                'bce': _nn.BCELoss(), 'bce_logits': _nn.BCEWithLogitsLoss(),
                'huber': _nn.HuberLoss(), 'nll': _nn.NLLLoss(),
            }
            self._loss_fn = loss_map.get(loss.lower(), _nn.MSELoss())
        else:
            self._loss_fn = loss
        self._metrics = metrics or []
        self._compiled = True
        return self

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self,
            x,
            y=None,
            epochs: int = 1,
            batch_size: int = 32,
            validation_split: float = 0.0,
            validation_data=None,
            shuffle: bool = True,
            callbacks: Optional[List] = None,
            verbose: int = 1,
            class_weight: Optional[Dict] = None,
            sample_weight=None,
            initial_epoch: int = 0,
            steps_per_epoch: Optional[int] = None) -> History:
        """Train the model.

        Args:
            x: input data (Tensor, ndarray, or DataLoader)
            y: target data (None if x is a DataLoader yielding (x, y))
            epochs: number of training epochs
            batch_size: mini-batch size
            validation_split: fraction of training data for validation
            validation_data: (x_val, y_val) or DataLoader
            shuffle: whether to shuffle before each epoch
            callbacks: list of callback objects
            verbose: 0=silent, 1=progress, 2=one line per epoch
        """
        assert self._compiled, 'Call model.compile() before model.fit()'
        history = History()
        self._stop_training = False
        callbacks = list(callbacks or []) + list(self._callbacks)

        # ── Build dataset ────────────────────────────────────────────────────
        # Handle DataLoader vs raw arrays
        is_loader = hasattr(x, '__iter__') and not isinstance(x, (Tensor, _np.ndarray))
        if not is_loader:
            x_arr = _to_np(x)
            y_arr = _to_np(y) if y is not None else None
            n = len(x_arr)
            # Validation split
            if validation_split > 0 and validation_data is None:
                split = int(n * (1 - validation_split))
                x_arr, x_val_arr = x_arr[:split], x_arr[split:]
                y_arr, y_val_arr = (y_arr[:split], y_arr[split:]) if y_arr is not None else (None, None)
                validation_data = (x_val_arr, y_val_arr)
            n = len(x_arr)

        # ── Callbacks: on_train_begin ─────────────────────────────────────────
        for cb in callbacks:
            if hasattr(cb, 'on_train_begin'):
                cb.on_train_begin({'epochs': epochs})

        # ── Epoch loop ────────────────────────────────────────────────────────
        for epoch in range(initial_epoch, epochs):
            if self._stop_training:
                break
            self.train()
            t0 = time.time()
            for cb in callbacks:
                if hasattr(cb, 'on_epoch_begin'):
                    cb.on_epoch_begin(epoch, {})

            if is_loader:
                epoch_loss, epoch_metrics = self._run_loader(x, training=True, batch_size=batch_size)
            else:
                # Build mini-batches
                indices = _np.random.permutation(n) if shuffle else _np.arange(n)
                epoch_loss, epoch_metrics = self._run_batches(
                    x_arr, y_arr, indices, batch_size, training=True,
                    steps_per_epoch=steps_per_epoch)

            # ── Validation ───────────────────────────────────────────────────
            val_metrics = {}
            if validation_data is not None:
                self.eval()
                xv, yv = validation_data
                if hasattr(xv, '__iter__') and not isinstance(xv, (Tensor, _np.ndarray)):
                    val_loss, val_m = self._run_loader(xv, training=False, batch_size=batch_size)
                else:
                    xv_arr, yv_arr = _to_np(xv), _to_np(yv) if yv is not None else None
                    nv = len(xv_arr)
                    val_loss, val_m = self._run_batches(
                        xv_arr, yv_arr, _np.arange(nv), batch_size, training=False)
                val_metrics = {'val_loss': val_loss}
                for k, v in val_m.items():
                    val_metrics[f'val_{k}'] = v

            # ── Log ──────────────────────────────────────────────────────────
            dt = time.time() - t0
            logs = {'loss': epoch_loss, **epoch_metrics, **val_metrics}
            history._append('loss', epoch_loss)
            for k, v in epoch_metrics.items():
                history._append(k, v)
            for k, v in val_metrics.items():
                history._append(k, v)

            if verbose == 1:
                parts = [f'Epoch {epoch+1}/{epochs}', f'loss={epoch_loss:.4f}']
                for k, v in epoch_metrics.items():
                    parts.append(f'{k}={v:.4f}')
                for k, v in val_metrics.items():
                    parts.append(f'{k}={v:.4f}')
                parts.append(f'({dt:.1f}s)')
                print(' - '.join(parts))
            elif verbose == 2:
                print(f'Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}')

            for cb in callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    cb.on_epoch_end(epoch, logs)
                    if getattr(cb, 'stop_training', False):
                        self._stop_training = True

        for cb in callbacks:
            if hasattr(cb, 'on_train_end'):
                cb.on_train_end(history.history)
        return history

    def _run_batches(self, x_arr, y_arr, indices, batch_size, training=True, steps_per_epoch=None):
        n = len(indices)
        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        n_batches = 0
        step = 0
        for start in range(0, n, batch_size):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            idx = indices[start:start+batch_size]
            xb = _to_tensor(x_arr[idx])
            yb = _to_tensor(y_arr[idx]) if y_arr is not None else None
            loss_val, m = self._train_step(xb, yb) if training else self._test_step(xb, yb)
            total_loss += loss_val
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            n_batches += 1
            step += 1
        avg_loss = total_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
        return avg_loss, avg_metrics

    def _run_loader(self, loader, training=True, batch_size=32):
        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                xb, yb = batch[0], batch[1] if len(batch) > 1 else None
            else:
                xb, yb = batch, None
            xb = _to_tensor(xb)
            yb = _to_tensor(yb) if yb is not None else None
            loss_val, m = self._train_step(xb, yb) if training else self._test_step(xb, yb)
            total_loss += loss_val
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
        return avg_loss, avg_metrics

    def _train_step(self, x, y):
        self._optimizer.zero_grad()
        pred = self(x)
        loss = self._loss_fn(pred, y) if y is not None else pred
        loss.backward()
        self._optimizer.step()
        loss_val = float(_np.asarray(loss._data).mean())
        metrics = self._compute_metrics(pred, y)
        return loss_val, metrics

    def _test_step(self, x, y):
        pred = self(x)
        loss = self._loss_fn(pred, y) if y is not None else pred
        loss_val = float(_np.asarray(loss._data).mean())
        metrics = self._compute_metrics(pred, y)
        return loss_val, metrics

    def _compute_metrics(self, pred, y) -> Dict[str, float]:
        result = {}
        if y is None:
            return result
        for m in self._metrics:
            m_lower = m.lower()
            pd = _np.asarray(pred._data)
            yd = _np.asarray(y._data)
            if m_lower == 'accuracy':
                if pd.ndim > 1 and pd.shape[-1] > 1:
                    preds = pd.argmax(axis=-1)
                else:
                    preds = (pd >= 0.5).astype(int).ravel()
                result['accuracy'] = float((preds == yd.astype(int).ravel()).mean())
            elif m_lower in ('mse', 'mean_squared_error'):
                result['mse'] = float(((pd - yd) ** 2).mean())
            elif m_lower in ('mae', 'mean_absolute_error'):
                result['mae'] = float(_np.abs(pd - yd).mean())
        return result

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, x, batch_size: int = 32, verbose: int = 0) -> _np.ndarray:
        """Generate predictions for input samples."""
        self.eval()
        x_arr = _to_np(x) if isinstance(x, (Tensor, _np.ndarray)) else None
        is_loader = x_arr is None
        all_preds = []
        if is_loader:
            for batch in x:
                xb = _to_tensor(batch[0] if isinstance(batch, (tuple, list)) else batch)
                pred = self(xb)
                all_preds.append(_np.asarray(pred._data))
        else:
            n = len(x_arr)
            for start in range(0, n, batch_size):
                xb = _to_tensor(x_arr[start:start+batch_size])
                pred = self(xb)
                all_preds.append(_np.asarray(pred._data))
        return _np.concatenate(all_preds, axis=0)

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, x, y=None, batch_size: int = 32, verbose: int = 1,
                 return_dict: bool = False):
        """Evaluate the model on test data.

        Returns:
            Scalar loss if no metrics, else [loss, *metric_values].
            If return_dict=True, returns {'loss': ..., 'accuracy': ...}.
        """
        assert self._compiled, 'Call model.compile() before model.evaluate()'
        self.eval()
        is_loader = hasattr(x, '__iter__') and not isinstance(x, (Tensor, _np.ndarray))
        if is_loader:
            loss, metrics = self._run_loader(x, training=False, batch_size=batch_size)
        else:
            x_arr, y_arr = _to_np(x), _to_np(y) if y is not None else None
            n = len(x_arr)
            loss, metrics = self._run_batches(x_arr, y_arr, _np.arange(n), batch_size, training=False)
        if verbose:
            parts = [f'loss={loss:.4f}']
            for k, v in metrics.items():
                parts.append(f'{k}={v:.4f}')
            print(', '.join(parts))
        if return_dict:
            return {'loss': loss, **metrics}
        result = [loss] + list(metrics.values())
        return result[0] if len(result) == 1 else result

    # ── save / load ───────────────────────────────────────────────────────────

    def save_weights(self, path: str):
        """Save model weights to a file (.npz format)."""
        import os
        sd = self.state_dict()
        arrays = {k: _np.asarray(v) for k, v in sd.items()}
        _np.savez(path, **arrays)

    def load_weights(self, path: str, strict: bool = True):
        """Load model weights from a .npz file."""
        data = _np.load(path, allow_pickle=True)
        self.load_state_dict(dict(data), strict=strict)

    def summary(self, line_length: int = 80):
        """Print a table summary of the model."""
        print('_' * line_length)
        print(f'{"Layer":<30} {"Output Shape":<20} {"Param #":>10}')
        print('=' * line_length)
        total_params = 0
        for name, m in self.named_modules():
            if name == '':
                continue
            n_params = sum(p._data.size for p in m._parameters.values())
            total_params += n_params
            shape_str = ''
            print(f'{name:<30} {shape_str:<20} {n_params:>10,}')
        print('=' * line_length)
        print(f'Total params: {total_params:,}')
        trainable = sum(p._data.size for p in self.parameters() if p.requires_grad)
        print(f'Trainable params: {trainable:,}')
        print(f'Non-trainable params: {total_params - trainable:,}')
        print('_' * line_length)

    def count_params(self) -> int:
        return sum(p._data.size for p in self.parameters())

    # ── callbacks builder ────────────────────────────────────────────────────

    def add_callback(self, cb):
        self._callbacks.append(cb)
        return self


# ── Built-in Keras callbacks ─────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor='val_loss', patience=5, min_delta=0.0,
                 mode='auto', restore_best_weights=True, verbose=1):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode if mode != 'auto' else ('min' if 'loss' in monitor else 'max')
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best = None
        self.wait = 0
        self.stop_training = False
        self._best_weights = None

    def on_train_begin(self, logs=None):
        self.best = _np.inf if self.mode == 'min' else -_np.inf
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        improved = (self.mode == 'min' and current < self.best - self.min_delta) or \
                   (self.mode == 'max' and current > self.best + self.min_delta)
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f'EarlyStopping: patience {self.patience} reached at epoch {epoch+1}')


class ReduceLROnPlateau:
    """Reduce learning rate when metric stops improving."""

    def __init__(self, monitor='val_loss', factor=0.1, patience=10, min_lr=0,
                 mode='auto', min_delta=1e-4, cooldown=0, verbose=1):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode if mode != 'auto' else ('min' if 'loss' in monitor else 'max')
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.verbose = verbose
        self.best = None
        self.wait = 0
        self.cooldown_counter = 0
        self._optimizer = None

    def set_model(self, model):
        self._optimizer = model._optimizer

    def on_train_begin(self, logs=None):
        self.best = _np.inf if self.mode == 'min' else -_np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None or self._optimizer is None:
            return
        improved = (self.mode == 'min' and current < self.best - self.min_delta) or \
                   (self.mode == 'max' and current > self.best + self.min_delta)
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for pg in self._optimizer.param_groups:
                    new_lr = max(pg['lr'] * self.factor, self.min_lr)
                    if self.verbose and new_lr != pg['lr']:
                        print(f'ReduceLROnPlateau: lr {pg["lr"]:.6f} -> {new_lr:.6f}')
                    pg['lr'] = new_lr
                self.wait = 0


class ModelCheckpoint:
    """Save the model at some frequency."""

    def __init__(self, filepath: str, monitor='val_loss', save_best_only=True,
                 mode='auto', verbose=1):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode if mode != 'auto' else ('min' if 'loss' in monitor else 'max')
        self.verbose = verbose
        self.best = None
        self._model = None

    def set_model(self, model):
        self._model = model

    def on_train_begin(self, logs=None):
        self.best = _np.inf if self.mode == 'min' else -_np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None or self._model is None:
            return
        improved = (self.mode == 'min' and current < self.best) or \
                   (self.mode == 'max' and current > self.best)
        if improved or not self.save_best_only:
            self.best = current
            path = self.filepath.format(epoch=epoch+1, **{k: round(v, 4) for k, v in (logs or {}).items()})
            self._model.save_weights(path)
            if self.verbose:
                print(f'ModelCheckpoint: saved {path}')


class LambdaCallback:
    """Create simple callbacks from functions."""

    def __init__(self, on_epoch_begin=None, on_epoch_end=None,
                 on_batch_begin=None, on_batch_end=None,
                 on_train_begin=None, on_train_end=None):
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end

    def on_epoch_begin(self, epoch, logs=None):
        if self._on_epoch_begin:
            self._on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        if self._on_train_begin:
            self._on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._on_train_end:
            self._on_train_end(logs)


class CSVLogger:
    """Log metrics to a CSV file."""

    def __init__(self, filepath: str, separator=',', append=False):
        self.filepath = filepath
        self.separator = separator
        self.append = append
        self._file = None
        self._writer = None

    def on_train_begin(self, logs=None):
        import csv
        mode = 'a' if self.append else 'w'
        self._file = open(self.filepath, mode, newline='')
        self._header_written = False

    def on_epoch_end(self, epoch, logs=None):
        import csv
        if self._file is None:
            return
        logs = logs or {}
        if not self._header_written:
            writer = csv.DictWriter(self._file, fieldnames=['epoch'] + list(logs.keys()))
            writer.writeheader()
            self._header_written = True
        writer = csv.DictWriter(self._file, fieldnames=['epoch'] + list(logs.keys()))
        writer.writerow({'epoch': epoch, **{k: round(v, 6) for k, v in logs.items()}})
        self._file.flush()

    def on_train_end(self, logs=None):
        if self._file:
            self._file.close()


class TensorBoard:
    """TensorBoard-compatible logging (writes to a summary directory)."""

    def __init__(self, log_dir: str = './logs', histogram_freq: int = 0,
                 write_graph: bool = True, update_freq: str = 'epoch'):
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.update_freq = update_freq
        self._step = 0
        self._summaries: List[Dict] = []

    def on_epoch_end(self, epoch, logs=None):
        self._summaries.append({'step': epoch, 'metrics': dict(logs or {})})
        self._step = epoch

    def flush(self):
        """Write a simple JSON summary (replace with real TB writer if installed)."""
        import json, os
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self._summaries, f, indent=2)

    def on_train_end(self, logs=None):
        self.flush()


# ── Sequential Keras-style ────────────────────────────────────────────────────

class Sequential(Model):
    """Keras-style Sequential model — add layers with .add()."""

    def __init__(self, layers=None):
        super().__init__()
        self._layer_list = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        name = f'layer_{len(self._layer_list)}'
        self._modules[name] = layer
        self._layer_list.append(layer)
        return self

    def forward(self, x):
        for layer in self._layer_list:
            x = layer(x)
        return x

    def pop(self):
        if self._layer_list:
            last = self._layer_list.pop()
            keys = list(self._modules.keys())
            del self._modules[keys[-1]]
        return self
