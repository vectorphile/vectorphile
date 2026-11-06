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
