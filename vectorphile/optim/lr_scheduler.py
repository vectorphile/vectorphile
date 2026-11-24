"""Learning rate schedulers."""
import math


class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        lrs = self.get_lr()
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = lr


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** (self.last_epoch // self.step_size) for base in self.base_lrs]


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** self.last_epoch for base in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        return [
            self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
            for base in self.base_lrs
        ]


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=5, last_epoch=-1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.total_iters)
        alpha = t / self.total_iters
        factor = self.start_factor + alpha * (self.end_factor - self.start_factor)
        return [base * factor for base in self.base_lrs]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing with periodic warm restarts (SGDR).

    Resets lr to base_lr every T_0 * T_mult^i steps (geometric progression).
    Args:
        T_0: Initial restart period (steps)
        T_mult: Period multiplier after each restart (default 1 = constant)
        eta_min: Minimum lr
    """
    def __init__(self, optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0.0, last_epoch: int = -1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._T_cur = 0
        self._T_i = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self._T_cur
        return [
            self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * t / self._T_i)) / 2
            for base in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            self._T_cur += 1
            if self._T_cur >= self._T_i:
                self._T_cur = 0
                self._T_i *= self.T_mult
        else:
            self._T_cur = epoch % self._T_i
        self.last_epoch += 1
        lrs = self.get_lr()
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = lr


class OneCycleLR(_LRScheduler):
    """One-cycle LR policy (Smith & Touvron 2018).

    Linearly ramps lr from initial_lr to max_lr over pct_start fraction,
    then cosine-anneals down to min_lr over the rest.
    Args:
        max_lr: Peak learning rate
        total_steps: Total number of steps
        pct_start: Fraction of steps for the warmup phase (default 0.3)
        div_factor: initial_lr = max_lr / div_factor (default 25)
        final_div_factor: min_lr = initial_lr / final_div_factor (default 1e4)
    """
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = max(0, self.last_epoch)
        warmup_steps = int(self.total_steps * self.pct_start)
        init_lr = self.max_lr / self.div_factor
        min_lr = init_lr / self.final_div_factor
        if t <= warmup_steps:
            alpha = t / max(1, warmup_steps)
            lr = init_lr + alpha * (self.max_lr - init_lr)
        else:
            remaining = t - warmup_steps
            total_remaining = self.total_steps - warmup_steps
            alpha = remaining / max(1, total_remaining)
            lr = min_lr + 0.5 * (self.max_lr - min_lr) * (1 + math.cos(math.pi * alpha))
        return [lr for _ in self.base_lrs]


class WarmupScheduler(_LRScheduler):
    """Linear warmup followed by another scheduler.

    Args:
        warmup_steps: steps to linearly ramp from 0 to base_lr
        after_scheduler: scheduler to apply after warmup completes
    """
    def __init__(self, optimizer, warmup_steps: int, after_scheduler=None,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / max(1, self.warmup_steps)
            return [base * alpha for base in self.base_lrs]
        if self.after_scheduler is not None:
            self.after_scheduler.step()
            return [g['lr'] for g in self.optimizer.param_groups]
        return self.base_lrs


class PolynomialLR(_LRScheduler):
    """Polynomial decay: lr = base * (1 - t/total_iters)^power.

    Args:
        total_iters: Total decay steps
        power: Polynomial power (1.0 = linear, 2.0 = quadratic)
        min_lr: Minimum lr floor
    """
    def __init__(self, optimizer, total_iters: int, power: float = 1.0,
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.total_iters)
        decay = (1 - t / max(1, self.total_iters)) ** self.power
        return [max(self.min_lr, base * decay) for base in self.base_lrs]


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0.0, eps=1e-8):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, metrics):
        current = float(metrics)
        if self.best is None:
            self.best = current
            return
        if self.mode == 'min':
            improved = current < self.best - self.threshold
        else:
            improved = current > self.best + self.threshold
        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        if self.num_bad_epochs >= self.patience:
            for g in self.optimizer.param_groups:
                new_lr = max(g['lr'] * self.factor, self.min_lr)
                g['lr'] = new_lr
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0


class CyclicLR:
    """Cyclically vary the learning rate between base_lr and max_lr.

    Implements the triangular, triangular2, and exp_range policies from
    "Cyclical Learning Rates for Training Neural Networks" (Smith 2017).

    Args:
        optimizer: wrapped optimizer
        base_lr: lower bound of cycle (minimum lr)
        max_lr: upper bound of cycle (maximum lr)
        step_size_up: steps in the increasing half-cycle (default 2000)
        step_size_down: steps in the decreasing half-cycle (None = step_size_up)
        mode: 'triangular' | 'triangular2' | 'exp_range'
        gamma: multiplicative factor for 'exp_range' (default 1.0)
        cycle_momentum: not implemented (kept for API compatibility)
    """
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size_up=2000, step_size_down=None,
                 mode='triangular', gamma=1.0, cycle_momentum=False):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.mode = mode
        self.gamma = gamma
        self._step_count = 0

    def _compute_lr(self):
        cycle_size = self.step_size_up + self.step_size_down
        cycle = self._step_count // cycle_size
        x = self._step_count % cycle_size
        if x < self.step_size_up:
            scale_x = x / self.step_size_up
        else:
            scale_x = (cycle_size - x) / self.step_size_down
        if self.mode == 'triangular':
            scale_fn = 1.0
        elif self.mode == 'triangular2':
            scale_fn = 1.0 / (2.0 ** cycle)
        else:  # exp_range
            scale_fn = self.gamma ** self._step_count
        return self.base_lr + (self.max_lr - self.base_lr) * scale_x * scale_fn

    def step(self):
        lr = self._compute_lr()
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self._step_count += 1

    def get_last_lr(self):
        return [self._compute_lr()]

    def state_dict(self):
        return {'step_count': self._step_count, 'mode': self.mode,
                'base_lr': self.base_lr, 'max_lr': self.max_lr}

    def load_state_dict(self, state: dict):
        self._step_count = state.get('step_count', 0)
        self.mode = state.get('mode', self.mode)
        self.base_lr = state.get('base_lr', self.base_lr)
        self.max_lr = state.get('max_lr', self.max_lr)
