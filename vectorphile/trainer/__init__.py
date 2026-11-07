"""vectorphile.trainer — High-level Trainer (PyTorch Lightning-style API)."""
from __future__ import annotations
import os
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union


class Trainer:
    """High-level training loop with callbacks, logging, and mixed precision.

    Example::

        trainer = Trainer(
            max_epochs=10,
            callbacks=[EarlyStopping('val_loss', patience=3)],
            log_every_n_steps=50,
            gradient_clip_val=1.0,
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
    """

    def __init__(
        self,
        max_epochs: int = 10,
        min_epochs: int = 1,
        callbacks: Optional[List] = None,
        logger=None,
        log_every_n_steps: int = 50,
        val_check_interval: Union[int, float] = 1.0,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = 'norm',
        accumulate_grad_batches: int = 1,
        precision: str = '32',
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        num_sanity_val_steps: int = 2,
        deterministic: bool = False,
        seed: Optional[int] = None,
        default_root_dir: str = '.',
        devices: Union[int, str] = 1,
    ):
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.callbacks = callbacks or []
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.num_sanity_val_steps = num_sanity_val_steps
        self.deterministic = deterministic
        self.default_root_dir = default_root_dir
        self.devices = devices

        if seed is not None:
            np.random.seed(seed)

        # State
        self.current_epoch = 0
        self.global_step = 0
        self._logs: Dict[str, List] = {}
        self._should_stop = False

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, name: str, value: float, prog_bar: bool = True):
        if name not in self._logs:
            self._logs[name] = []
        self._logs[name].append(float(value))

    def _get_last(self, name: str) -> Optional[float]:
        return self._logs[name][-1] if name in self._logs and self._logs[name] else None

    # ── Callback dispatch ─────────────────────────────────────────────────────

    def _cb(self, event: str, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn is not None:
                fn(*args, **kwargs)

    # ── Gradient clipping ─────────────────────────────────────────────────────

    def _clip_gradients(self, model):
        if self.gradient_clip_val is None:
            return
        params = [p for p in model.parameters() if getattr(p, 'grad', None) is not None]
        if not params:
            return
        if self.gradient_clip_algorithm == 'norm':
            total_norm = math.sqrt(sum(np.sum(p.grad ** 2) for p in params))
            if total_norm > self.gradient_clip_val:
                scale = self.gradient_clip_val / (total_norm + 1e-6)
                for p in params:
                    p.grad *= scale
        elif self.gradient_clip_algorithm == 'value':
            for p in params:
                p.grad = np.clip(p.grad, -self.gradient_clip_val, self.gradient_clip_val)

    # ── Batch limit helper ────────────────────────────────────────────────────

    def _n_batches(self, loader, limit) -> int:
        total = len(loader) if hasattr(loader, '__len__') else None
        if isinstance(limit, float):
            if total is None:
                return 10**9
            return max(1, int(total * limit))
        return int(limit)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, model, train_dataloader, val_dataloaders=None, optimizer=None):
        """Train the model.

        Args:
            model: a Module with training_step(batch, batch_idx) → loss
                   and optional validation_step(batch, batch_idx) → dict
            train_dataloader: iterable of batches
            val_dataloaders: optional val dataloader or list thereof
            optimizer: optional optimizer; if None model must expose configure_optimizers()
        """
        import math
        self._should_stop = False

        # Get optimizer
        if optimizer is None:
            if hasattr(model, 'configure_optimizers'):
                opt_result = model.configure_optimizers()
                if isinstance(opt_result, tuple):
                    optimizer, scheduler = opt_result[0], opt_result[1] if len(opt_result) > 1 else None
                    if isinstance(optimizer, list):
                        optimizer = optimizer[0]
                    if isinstance(scheduler, list):
                        scheduler = scheduler[0]
                else:
                    optimizer = opt_result
                    scheduler = None
            else:
                raise ValueError("Pass optimizer= or implement model.configure_optimizers()")
        else:
            scheduler = None

        if self.enable_model_summary:
            n_params = sum(p.size for p in model.parameters() if hasattr(p, 'size'))
            print(f'Trainer — model params: {n_params:,}')

        val_loaders = []
        if val_dataloaders is not None:
            val_loaders = val_dataloaders if isinstance(val_dataloaders, list) else [val_dataloaders]

        # Sanity check
        if val_loaders and self.num_sanity_val_steps > 0:
            self._run_validation(model, val_loaders[0], max_batches=self.num_sanity_val_steps)

        self._cb('on_train_begin', logs={})

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._cb('on_epoch_begin', epoch=epoch, logs={})

            if hasattr(model, 'on_epoch_start'):
                model.on_epoch_start()

            # ── Training epoch ─────────────────────────────────────────────
            epoch_logs: Dict[str, float] = {}
            n_train = self._n_batches(train_dataloader, self.limit_train_batches)
            train_iter = iter(train_dataloader)
            batch_loss_sum = 0.0
            n_batches = 0

            for batch_idx in range(n_train):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break

                self._cb('on_batch_begin', batch=batch_idx, logs={})

                # Forward + loss
                loss = model.training_step(batch, batch_idx)
                if hasattr(loss, 'data'):
                    loss_val = float(np.mean(loss.data))
                elif isinstance(loss, dict):
                    loss_val = float(loss.get('loss', 0.0))
                    loss = loss.get('loss', loss_val)
                else:
                    loss_val = float(np.mean(np.asarray(loss)))

                # Backward
                if hasattr(loss, 'backward'):
                    loss.backward()
                elif hasattr(model, 'backward'):
                    model.backward(loss)

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    self._clip_gradients(model)
                    optimizer.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                batch_loss_sum += loss_val
                n_batches += 1

                self.log('train_loss', loss_val)
                self._cb('on_batch_end', batch=batch_idx,
                         logs={'loss': loss_val, 'step': self.global_step})

                if self.enable_progress_bar and batch_idx % max(1, n_train // 10) == 0:
                    print(f'  Epoch {epoch+1}/{self.max_epochs}  '
                          f'[{batch_idx+1}/{n_train}]  loss={loss_val:.4f}', end='\r')

                if self._should_stop:
                    break

            epoch_logs['loss'] = batch_loss_sum / max(n_batches, 1)
            if self.enable_progress_bar:
                print()

            # ── Validation ─────────────────────────────────────────────────
            if val_loaders and self._should_validate(epoch):
                val_logs = self._run_validation(model, val_loaders[0])
                epoch_logs.update(val_logs)
                for k, v in val_logs.items():
                    self.log(k, v)

            # ── LR scheduler ───────────────────────────────────────────────
            if scheduler is not None:
                monitor_val = epoch_logs.get('val_loss')
                if monitor_val is not None:
                    scheduler.step(monitor_val)
                else:
                    scheduler.step()

            self._cb('on_epoch_end', epoch=epoch, logs=epoch_logs)

            if self.enable_progress_bar:
                log_str = '  '.join(f'{k}={v:.4f}' for k, v in epoch_logs.items())
                print(f'Epoch {epoch+1}/{self.max_epochs}  {log_str}')

            if self._should_stop and epoch + 1 >= self.min_epochs:
                print(f'Early stopping at epoch {epoch+1}')
                break

        self._cb('on_train_end', logs=epoch_logs if 'epoch_logs' in dir() else {})

    def _should_validate(self, epoch: int) -> bool:
        if isinstance(self.val_check_interval, float):
            return True  # validate every epoch
        return (epoch + 1) % int(self.val_check_interval) == 0

    def _run_validation(self, model, val_loader, max_batches: Optional[int] = None) -> Dict:
        if hasattr(model, 'eval'):
            model.eval()

        n_val = self._n_batches(val_loader, self.limit_val_batches)
        if max_batches is not None:
            n_val = min(n_val, max_batches)

        all_outputs = []
        val_iter = iter(val_loader)
        for batch_idx in range(n_val):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            if hasattr(model, 'validation_step'):
                out = model.validation_step(batch, batch_idx)
                all_outputs.append(out)

        logs = {}
        if all_outputs:
            if hasattr(model, 'validation_epoch_end'):
                logs = model.validation_epoch_end(all_outputs) or {}
            else:
                # Auto-aggregate dict outputs
                for out in all_outputs:
                    if isinstance(out, dict):
                        for k, v in out.items():
                            if k not in logs:
                                logs[k] = []
                            logs[k].append(float(v) if hasattr(v, '__float__') else v)
                logs = {k: float(np.mean(v)) for k, v in logs.items() if isinstance(v[0], float)}

        if hasattr(model, 'train'):
            model.train()

        return logs

    # ── Test ─────────────────────────────────────────────────────────────────

    def test(self, model, test_dataloaders, verbose: bool = True) -> List[Dict]:
        if hasattr(model, 'eval'):
            model.eval()

        loaders = test_dataloaders if isinstance(test_dataloaders, list) else [test_dataloaders]
        results = []
        for loader in loaders:
            n_test = self._n_batches(loader, self.limit_test_batches)
            all_outputs = []
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= n_test:
                    break
                if hasattr(model, 'test_step'):
                    out = model.test_step(batch, batch_idx)
                    all_outputs.append(out)

            if all_outputs and hasattr(model, 'test_epoch_end'):
                logs = model.test_epoch_end(all_outputs) or {}
            else:
                logs = {}
                for out in all_outputs:
                    if isinstance(out, dict):
                        for k, v in out.items():
                            logs.setdefault(k, []).append(float(v) if hasattr(v, '__float__') else v)
                logs = {k: float(np.mean(v)) for k, v in logs.items() if v and isinstance(v[0], float)}
            if verbose:
                print('Test results:', logs)
            results.append(logs)

        return results

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, model, dataloaders) -> List[Any]:
        if hasattr(model, 'eval'):
            model.eval()
        loaders = dataloaders if isinstance(dataloaders, list) else [dataloaders]
        predictions = []
        for loader in loaders:
            for batch_idx, batch in enumerate(loader):
                if hasattr(model, 'predict_step'):
                    pred = model.predict_step(batch, batch_idx)
                else:
                    pred = model(batch[0] if isinstance(batch, (list, tuple)) else batch)
                predictions.append(pred)
        return predictions

    # ── Stop ─────────────────────────────────────────────────────────────────

    def should_stop(self):
        self._should_stop = True


# ── LightningModule-compatible base class ─────────────────────────────────────

class TrainableModel:
    """Base class for models compatible with Trainer.

    Subclass and implement:
      - training_step(batch, batch_idx) → loss
      - validation_step(batch, batch_idx) → dict (optional)
      - configure_optimizers() → optimizer or (optimizer, scheduler)
    """

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx) -> Optional[Dict]:
        return None

    def test_step(self, batch, batch_idx) -> Optional[Dict]:
        return None

    def predict_step(self, batch, batch_idx) -> Any:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)

    def configure_optimizers(self):
        raise NotImplementedError

    def train(self):
        """Set model to training mode (implemented by Module)."""

    def eval(self):
        """Set model to eval mode (implemented by Module)."""

    def parameters(self):
        return []

    def backward(self, loss):
        if hasattr(loss, 'backward'):
            loss.backward()


try:
    import math
except ImportError:
    import math


__all__ = ['Trainer', 'TrainableModel']
