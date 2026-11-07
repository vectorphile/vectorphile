"""vectorphile.nn.lr_schedulers — Learning rate schedulers (PyTorch-compatible API)."""
from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Union


class LRScheduler:
    """Base LR scheduler."""

    def __init__(self, optimizer, last_epoch: int = -1, verbose: bool = False):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step()

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def step(self, metrics=None):
        self.last_epoch += 1
        lrs = self.get_lr()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg['lr'] = lr
        if self.verbose:
            print(f'LRScheduler: epoch {self.last_epoch}  lr={lrs[0]:.6e}')

    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state: dict):
        self.__dict__.update(state)

    @property
    def _last_lr(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]


class StepLR(LRScheduler):
    """Decay LR by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1, verbose: bool = False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [pg['lr'] for pg in self.optimizer.param_groups]
        return [base * self.gamma ** (self.last_epoch // self.step_size)
                for base in self._base_lrs]


class MultiStepLR(LRScheduler):
    """Decay LR by gamma at each milestone epoch."""

    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1,
                 last_epoch: int = -1, verbose: bool = False):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        n = sum(1 for m in self.milestones if self.last_epoch >= m)
        return [base * self.gamma ** n for base in self._base_lrs]


class ExponentialLR(LRScheduler):
    """Decay LR by gamma every epoch: lr = base * gamma^epoch."""

    def __init__(self, optimizer, gamma: float, last_epoch: int = -1, verbose: bool = False):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        return [base * self.gamma ** self.last_epoch for base in self._base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing: lr oscillates between base_lr and eta_min over T_max steps."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0,
                 last_epoch: int = -1, verbose: bool = False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        t = self.last_epoch
        T = self.T_max
        return [self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * t / T)) / 2
                for base in self._base_lrs]


class CosineAnnealingWarmRestarts(LRScheduler):
    """SGDR — Loshchilov & Hutter 2016. Cosine annealing with warm restarts."""

    def __init__(self, optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0.0,
                 last_epoch: int = -1, verbose: bool = False):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def _get_t_cur(self):
        t = self.last_epoch
        T_i = self.T_0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
        return t, T_i

    def get_lr(self) -> List[float]:
        t_cur, T_i = self._get_t_cur()
        return [self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * t_cur / T_i)) / 2
                for base in self._base_lrs]


class OneCycleLR(LRScheduler):
    """1-cycle LR policy — Smith & Touvron 2019. Super-convergence.

    Three phases:
      1. Linear warmup from base_lr/div_factor to max_lr
      2. Cosine annealing from max_lr to base_lr
      3. Final anneal to base_lr/final_div_factor
    """

    def __init__(self, optimizer, max_lr: Union[float, List[float]],
                 total_steps: int, pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.0, final_div_factor: float = 1e4,
                 last_epoch: int = -1, verbose: bool = False):
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        if isinstance(max_lr, (int, float)):
            max_lr = [max_lr] * len(optimizer.param_groups)
        self.max_lrs = max_lr
        # Override base_lrs from max_lr
        self._init_lrs = [m / div_factor for m in self.max_lrs]
        self._min_lrs = [m / final_div_factor for m in self.max_lrs]
        super().__init__(optimizer, last_epoch, verbose)

    def _anneal(self, start, end, pct):
        if self.anneal_strategy == 'cos':
            return end + (start - end) / 2 * (1 + math.cos(math.pi * pct))
        return start + (end - start) * pct  # linear

    def get_lr(self) -> List[float]:
        t = self.last_epoch
        T = self.total_steps
        warmup = int(T * self.pct_start)
        lrs = []
        for init, mx, mn in zip(self._init_lrs, self.max_lrs, self._min_lrs):
            if t <= warmup:
                lr = init + (mx - init) * t / max(warmup, 1)
            elif t <= T:
                pct = (t - warmup) / max(T - warmup, 1)
                lr = self._anneal(mx, mn, pct)
            else:
                lr = mn
            lrs.append(lr)
        return lrs


class WarmupCosineDecay(LRScheduler):
    """Linear warmup + cosine decay — standard schedule for transformer training."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1, last_epoch: int = -1, verbose: bool = False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        t = self.last_epoch
        w = self.warmup_steps
        T = self.total_steps
        lrs = []
        for base in self._base_lrs:
            if t < w:
                lr = base * t / max(w, 1)
            else:
                progress = (t - w) / max(T - w, 1)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            lrs.append(lr)
        return lrs


class LinearWarmup(LRScheduler):
    """Linear warmup from 0 to base_lr over warmup_steps, then constant."""

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1, verbose: bool = False):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        t = self.last_epoch
        w = self.warmup_steps
        return [base * min(1.0, t / max(w, 1)) for base in self._base_lrs]


class PolynomialLR(LRScheduler):
    """Polynomial decay: lr = base * (1 - t/T)^power."""

    def __init__(self, optimizer, total_iters: int, power: float = 1.0,
                 last_epoch: int = -1, verbose: bool = False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        t = min(self.last_epoch, self.total_iters)
        factor = (1 - t / self.total_iters) ** self.power
        return [base * factor for base in self._base_lrs]


class ReduceLROnPlateauScheduler(LRScheduler):
    """Reduce LR when a metric stops improving (call step(metrics) each epoch)."""

    def __init__(self, optimizer, mode: str = 'min', factor: float = 0.5,
                 patience: int = 10, threshold: float = 1e-4,
                 min_lr: float = 0.0, verbose: bool = False):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self._best = float('inf') if mode == 'min' else -float('inf')
        self._wait = 0
        # Don't call super().__init__ — we override step()
        self.optimizer = optimizer
        self.verbose = verbose
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.last_epoch = 0

    def get_lr(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def step(self, metrics: Optional[float] = None):
        if metrics is None:
            return
        self.last_epoch += 1
        improved = (metrics < self._best - self.threshold) if self.mode == 'min' \
                   else (metrics > self._best + self.threshold)
        if improved:
            self._best = metrics
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._wait = 0
                for pg in self.optimizer.param_groups:
                    new_lr = max(pg['lr'] * self.factor, self.min_lr)
                    if self.verbose and new_lr != pg['lr']:
                        print(f'ReduceLROnPlateau: lr {pg["lr"]:.2e} → {new_lr:.2e}')
                    pg['lr'] = new_lr


def get_scheduler(name: str, optimizer, **kwargs) -> LRScheduler:
    """Factory function for LR schedulers.

    Args:
        name: 'cosine', 'cosine_warmup', 'linear_warmup', 'step', 'exponential',
              'cosine_with_restarts', 'polynomial', 'constant', 'plateau'
        optimizer: the optimizer to wrap
        **kwargs: scheduler-specific arguments

    Examples::

        sched = get_scheduler('cosine_warmup', opt, warmup_steps=100, total_steps=1000)
        sched = get_scheduler('step', opt, step_size=10, gamma=0.5)
        sched = get_scheduler('cosine', opt, T_max=100)
    """
    name = name.lower().replace('-', '_')
    mapping = {
        'cosine':              (CosineAnnealingLR,         ['T_max']),
        'cosine_warmup':       (WarmupCosineDecay,         ['warmup_steps', 'total_steps']),
        'warmup_cosine':       (WarmupCosineDecay,         ['warmup_steps', 'total_steps']),
        'linear_warmup':       (LinearWarmup,              ['warmup_steps']),
        'step':                (StepLR,                    ['step_size']),
        'multistep':           (MultiStepLR,               ['milestones']),
        'exponential':         (ExponentialLR,             ['gamma']),
        'cosine_with_restarts':(CosineAnnealingWarmRestarts, ['T_0']),
        'polynomial':          (PolynomialLR,              ['total_iters']),
        'one_cycle':           (OneCycleLR,                ['max_lr', 'total_steps']),
        'plateau':             (ReduceLROnPlateauScheduler, []),
        'constant':            (StepLR,                    []),  # step_size=huge → constant
    }
    if name not in mapping:
        raise ValueError(f'Unknown scheduler: {name}. Available: {list(mapping.keys())}')
    cls, _ = mapping[name]
    if name == 'constant':
        kwargs.setdefault('step_size', 10**9)
        kwargs.setdefault('gamma', 1.0)
    return cls(optimizer, **kwargs)
