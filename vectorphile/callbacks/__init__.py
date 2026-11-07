"""vectorphile.callbacks — Keras-style training callbacks."""
from __future__ import annotations
import os
import csv
import copy
import time
import struct
import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class Callback:
    """Base callback — override any method you need."""
    def set_model(self, model): self.model = model
    def set_optimizer(self, optimizer): self.optimizer = optimizer
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch: int, logs=None): pass
    def on_epoch_end(self, epoch: int, logs=None): pass
    def on_batch_begin(self, batch: int, logs=None): pass
    def on_batch_end(self, batch: int, logs=None): pass
    def on_validation_begin(self, logs=None): pass
    def on_validation_end(self, logs=None): pass


class CallbackList:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, cb: Callback): self.callbacks.append(cb)

    def set_model(self, model):
        for cb in self.callbacks: cb.set_model(model)

    def set_optimizer(self, opt):
        for cb in self.callbacks: cb.set_optimizer(opt)

    def on_train_begin(self, logs=None):
        for cb in self.callbacks: cb.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cb in self.callbacks: cb.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks: cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks: cb.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for cb in self.callbacks: cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for cb in self.callbacks: cb.on_batch_end(batch, logs)

    def on_validation_begin(self, logs=None):
        for cb in self.callbacks: cb.on_validation_begin(logs)

    def on_validation_end(self, logs=None):
        for cb in self.callbacks: cb.on_validation_end(logs)


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: metric name to watch
        patience: epochs with no improvement before stopping
        min_delta: minimum change to qualify as improvement
        mode: 'min' or 'max'
        restore_best_weights: restore model weights from best epoch on stop
        verbose: print message on stop
    """

    def __init__(self, monitor: str = 'val_loss', patience: int = 5,
                 min_delta: float = 0.0, mode: str = 'min',
                 restore_best_weights: bool = True, verbose: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self._best = None
        self._wait = 0
        self._best_weights = None
        self.stopped_epoch = 0
        self.stop_training = False

    def on_train_begin(self, logs=None):
        self._best = float('inf') if self.mode == 'min' else -float('inf')
        self._wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (current < self._best - self.min_delta) if self.mode == 'min' \
                   else (current > self._best + self.min_delta)
        if improved:
            self._best = current
            self._wait = 0
            if self.restore_best_weights and hasattr(self, 'model'):
                import pickle
                self._best_weights = pickle.dumps(self.model.state_dict())
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.verbose:
                    print(f'\nEarlyStopping: stopping at epoch {epoch+1} '
                          f'(best {self.monitor}={self._best:.4f})')
                if self.restore_best_weights and self._best_weights and hasattr(self, 'model'):
                    import pickle
                    self.model.load_state_dict(pickle.loads(self._best_weights))
                    if self.verbose:
                        print('Restored best weights.')


class ModelCheckpoint(Callback):
    """Save model weights when a monitored metric improves.

    Args:
        filepath: path to save weights (supports {epoch} and {val_loss} placeholders)
        monitor: metric to watch
        save_best_only: only save when metric improves
        mode: 'min' or 'max'
        verbose: print on save
    """

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, mode: str = 'min',
                 verbose: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self._best = float('inf') if mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        improved = True
        if self.save_best_only and current is not None:
            improved = (current < self._best) if self.mode == 'min' else (current > self._best)
        if improved:
            if current is not None:
                self._best = current
            path = self.filepath.format(epoch=epoch + 1, **{k: f'{v:.4f}' for k, v in (logs or {}).items()})
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self, 'model'):
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(self.model.state_dict(), f)
            if self.verbose:
                print(f'\nModelCheckpoint: saved to {path}'
                      + (f' ({self.monitor}={current:.4f})' if current is not None else ''))


class ReduceLROnPlateau(Callback):
    """Reduce LR when a metric stops improving.

    Args:
        monitor: metric to watch
        factor: LR multiplied by this on reduction (< 1)
        patience: epochs to wait before reducing
        min_lr: lower bound on LR
        mode: 'min' or 'max'
        verbose: print on reduction
    """

    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5,
                 patience: int = 3, min_lr: float = 1e-7,
                 mode: str = 'min', verbose: bool = True):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        self._best = float('inf') if mode == 'min' else -float('inf')
        self._wait = 0

    def on_train_begin(self, logs=None):
        self._best = float('inf') if self.mode == 'min' else -float('inf')
        self._wait = 0

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None or not hasattr(self, 'optimizer'):
            return
        improved = (current < self._best) if self.mode == 'min' else (current > self._best)
        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._wait = 0
                for pg in self.optimizer.param_groups:
                    old_lr = pg['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    pg['lr'] = new_lr
                    if self.verbose and old_lr != new_lr:
                        print(f'\nReduceLROnPlateau: lr {old_lr:.2e} → {new_lr:.2e}')


class LearningRateScheduler(Callback):
    """Apply a user-defined LR schedule each epoch.

    Args:
        schedule: callable(epoch) -> new_lr
    """

    def __init__(self, schedule: Callable[[int], float], verbose: bool = False):
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs=None):
        if not hasattr(self, 'optimizer'):
            return
        new_lr = self.schedule(epoch)
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr
        if self.verbose:
            print(f'LearningRateScheduler: epoch {epoch+1} lr={new_lr:.2e}')


class TensorBoardLogger(Callback):
    """Write scalar summaries in TensorBoard event file format (pure Python).

    Uses the TFRecord binary format for TF event files.
    """

    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self._writers: Dict[str, Any] = {}
        self._step = 0

    def _get_writer(self, tag: str):
        key = 'train' if 'val' not in tag else 'val'
        if key not in self._writers:
            path = self.log_dir / key
            path.mkdir(parents=True, exist_ok=True)
            fname = path / f'events.out.tfevents.{int(time.time())}'
            self._writers[key] = open(fname, 'wb')
        return self._writers[key]

    def _masked_crc32c(self, data: bytes) -> int:
        """Masked CRC32C as used by TFRecord."""
        import zlib
        crc = zlib.crc32(data) & 0xFFFFFFFF
        return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xFFFFFFFF

    def _write_scalar(self, tag: str, value: float, step: int):
        """Write a scalar summary TFRecord event."""
        # Build a minimal tensorflow.Event protobuf manually
        # Event: wall_time(1=double) + step(2=int64) + summary(5=message)
        # Summary: value(1=repeated message)
        # Summary.Value: tag(1=string) + simple_value(2=float)
        def _varint(n):
            buf = []
            while n > 0x7F:
                buf.append((n & 0x7F) | 0x80)
                n >>= 7
            buf.append(n)
            return bytes(buf)

        def _len_del(field: int, data: bytes) -> bytes:
            tag_b = _varint((field << 3) | 2)
            return tag_b + _varint(len(data)) + data

        def _fixed64(field: int, val: float) -> bytes:
            return _varint((field << 3) | 1) + struct.pack('<d', val)

        def _fixed32(field: int, val: float) -> bytes:
            return _varint((field << 3) | 5) + struct.pack('<f', val)

        def _varint_field(field: int, val: int) -> bytes:
            return _varint((field << 3) | 0) + _varint(val)

        # Summary.Value
        sv_tag = _len_del(1, tag.encode())
        sv_val = _fixed32(2, value)
        sum_value = sv_tag + sv_val

        # Summary
        summary = _len_del(1, sum_value)

        # Event
        wall = _fixed64(1, time.time())
        step_b = _varint_field(2, step)
        sum_b = _len_del(5, summary)
        event = wall + step_b + sum_b

        # TFRecord: len + masked_crc(len) + data + masked_crc(data)
        length = struct.pack('<Q', len(event))
        f = self._get_writer(tag)
        f.write(length)
        f.write(struct.pack('<I', self._masked_crc32c(length)))
        f.write(event)
        f.write(struct.pack('<I', self._masked_crc32c(event)))
        f.flush()

    def on_epoch_end(self, epoch: int, logs=None):
        for k, v in (logs or {}).items():
            self._write_scalar(k, float(v), epoch)
        self._step = epoch + 1

    def on_train_end(self, logs=None):
        for f in self._writers.values():
            f.close()


class WandBLogger(Callback):
    """Log metrics to Weights & Biases (requires `pip install wandb`)."""

    def __init__(self, project: str, run_name: Optional[str] = None,
                 config: Optional[dict] = None):
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self._run = None

    def on_train_begin(self, logs=None):
        try:
            import wandb
            self._run = wandb.init(project=self.project, name=self.run_name,
                                   config=self.config, reinit=True)
        except ImportError:
            logger.warning('wandb not installed — WandBLogger disabled')

    def on_epoch_end(self, epoch: int, logs=None):
        if self._run:
            import wandb
            wandb.log(logs or {}, step=epoch)

    def on_train_end(self, logs=None):
        if self._run:
            import wandb
            wandb.finish()


class CSVLogger(Callback):
    """Log epoch metrics to a CSV file."""

    def __init__(self, filepath: str, separator: str = ',', append: bool = False):
        self.filepath = filepath
        self.separator = separator
        self.append = append
        self._writer = None
        self._file = None
        self._fieldnames = None

    def on_train_begin(self, logs=None):
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if self.append else 'w'
        self._file = open(self.filepath, mode, newline='')

    def on_epoch_end(self, epoch: int, logs=None):
        logs = dict(logs or {})
        logs['epoch'] = epoch + 1
        if self._writer is None:
            self._fieldnames = ['epoch'] + [k for k in logs if k != 'epoch']
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames,
                                          delimiter=self.separator)
            self._writer.writeheader()
        self._writer.writerow({k: f'{v:.6f}' if isinstance(v, float) else v
                                for k, v in logs.items()})
        self._file.flush()

    def on_train_end(self, logs=None):
        if self._file:
            self._file.close()


class ProgressBar(Callback):
    """Terminal progress bar with real-time loss/metric display."""

    def __init__(self, width: int = 30):
        self.width = width
        self._epoch_start = 0.0
        self._total_epochs = 0

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch: int, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs=None):
        elapsed = time.time() - self._epoch_start
        metrics_str = '  '.join(f'{k}={v:.4f}' for k, v in (logs or {}).items())
        print(f'\rEpoch {epoch+1}  [{elapsed:.1f}s]  {metrics_str}')


class GradientMonitor(Callback):
    """Log gradient norms per layer to detect vanishing/exploding gradients.

    Args:
        log_every: log every N batches
    """

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self._batch = 0
        self.history: Dict[str, List[float]] = {}

    def on_batch_end(self, batch: int, logs=None):
        self._batch += 1
        if self._batch % self.log_every != 0:
            return
        if not hasattr(self, 'model'):
            return
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = float((_asarray(param.grad) ** 2).sum() ** 0.5)
                self.history.setdefault(name, []).append(norm)

    def report(self) -> dict:
        """Return dict of param_name → list of gradient norms."""
        return dict(self.history)


def _asarray(x):
    if hasattr(x, '_data'):
        return x._data
    import numpy as np
    return np.asarray(x)


class LRFinder(Callback):
    """Learning rate range test — Smith 2017.

    Usage::

        finder = LRFinder(min_lr=1e-7, max_lr=10, num_iter=100)
        trainer.fit(model, train_loader, callbacks=[finder])
        print('Optimal LR:', finder.optimal_lr)
        finder.plot()
    """

    def __init__(self, min_lr: float = 1e-7, max_lr: float = 10,
                 num_iter: int = 100, smooth_f: float = 0.05):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iter = num_iter
        self.smooth_f = smooth_f
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self._batch = 0
        self._best_loss = float('inf')
        self._avg_loss = 0.0
        self.stop_training = False

    def on_train_begin(self, logs=None):
        self.lrs = []
        self.losses = []
        self._batch = 0
        if hasattr(self, 'optimizer'):
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.min_lr

    def on_batch_end(self, batch: int, logs=None):
        loss = (logs or {}).get('loss', None)
        if loss is None or not hasattr(self, 'optimizer'):
            return
        # Exponential LR increase
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (self._batch / self.num_iter)
        self._avg_loss = self.smooth_f * loss + (1 - self.smooth_f) * self._avg_loss
        smoothed = self._avg_loss / (1 - (1 - self.smooth_f) ** (self._batch + 1))
        self.lrs.append(lr)
        self.losses.append(smoothed)
        if smoothed < self._best_loss:
            self._best_loss = smoothed
        if self._batch > 10 and smoothed > 4 * self._best_loss:
            self.stop_training = True
        # Update LR
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self._batch += 1
        if self._batch >= self.num_iter:
            self.stop_training = True

    @property
    def optimal_lr(self) -> float:
        """Return LR at the point of steepest loss decrease."""
        if len(self.losses) < 2:
            return self.min_lr
        import numpy as np
        losses = np.array(self.losses)
        grads = np.gradient(losses)
        idx = int(np.argmin(grads))
        return float(self.lrs[idx])

    def plot(self):
        """Display LR vs loss curve using vectorphile.viz."""
        try:
            from vectorphile import viz
            viz.plot(self.lrs, self.losses, xlabel='Learning Rate', ylabel='Loss',
                     title='LR Finder', xscale='log')
            viz.show()
        except Exception:
            try:
                import matplotlib.pyplot as plt
                plt.semilogx(self.lrs, self.losses)
                plt.xlabel('Learning Rate')
                plt.ylabel('Loss')
                plt.title('LR Finder')
                plt.show()
            except ImportError:
                print('Install matplotlib to use LRFinder.plot()')
