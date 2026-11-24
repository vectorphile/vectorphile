"""vectorphile.utils.progress — tqdm-compatible progress bar.

Usage::

    from vectorphile.utils.progress import tqdm, track

    for item in tqdm(iterable, desc='Training'):
        process(item)

    # Or with 'track' alias
    for step, batch in track(enumerate(loader), total=len(loader)):
        ...
"""
from __future__ import annotations
import sys
import time
import math
from typing import Any, Iterable, Iterator, Optional


class tqdm:
    """Lightweight tqdm-compatible progress bar.

    Features: ETA, rate, percentage, postfix metrics, nested support.

    Usage::

        for x in tqdm(range(100), desc='Processing', unit='items'):
            ...

        pbar = tqdm(total=100, desc='Downloading')
        pbar.update(10)
        pbar.close()
    """

    def __init__(self, iterable: Optional[Iterable] = None, *,
                 desc: str = '', total: Optional[int] = None,
                 leave: bool = True, file=None, ncols: int = 80,
                 mininterval: float = 0.1, unit: str = 'it',
                 unit_scale: bool = False, disable: bool = False,
                 initial: int = 0, dynamic_ncols: bool = False,
                 smoothing: float = 0.3, **kwargs):
        self._iterable = iterable
        self.desc = desc
        self.total = total
        if iterable is not None and total is None:
            try:
                self.total = len(iterable)  # type: ignore
            except (TypeError, AttributeError):
                self.total = None
        self.leave = leave
        self._file = file or sys.stderr
        self.ncols = ncols
        self.mininterval = mininterval
        self.unit = unit
        self.unit_scale = unit_scale
        self.disable = disable
        self.n = initial
        self._smoothing = smoothing
        self._start_t = time.time()
        self._last_print_t = 0.0
        self._last_n = initial
        self._rate = 0.0
        self._postfix: dict = {}

    # ── iteration ───────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator:
        if self._iterable is None:
            raise TypeError('tqdm: no iterable provided')
        for obj in self._iterable:
            yield obj
            self.update(1)
        if self.leave:
            self._print_progress(force=True)
            self._file.write('\n')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── update ──────────────────────────────────────────────────────────────

    def update(self, n: int = 1) -> None:
        if self.disable:
            return
        self.n += n
        now = time.time()
        dt = now - self._last_print_t
        if dt >= self.mininterval:
            elapsed = now - self._start_t
            dn = self.n - self._last_n
            if dt > 0:
                inst_rate = dn / dt
                self._rate = (self._smoothing * inst_rate +
                              (1 - self._smoothing) * self._rate) if self._rate else inst_rate
            self._last_n = self.n
            self._last_print_t = now
            self._print_progress()

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        if ordered_dict:
            self._postfix.update(ordered_dict)
        self._postfix.update(kwargs)
        if refresh:
            self._print_progress(force=True)

    def set_description(self, desc: str = '', refresh: bool = True) -> None:
        self.desc = desc
        if refresh:
            self._print_progress(force=True)

    def close(self) -> None:
        if not self.disable:
            self._print_progress(force=True)
            self._file.write('\n')
            self._file.flush()

    def reset(self, total: Optional[int] = None) -> None:
        self.n = 0
        if total is not None:
            self.total = total
        self._start_t = time.time()
        self._last_print_t = 0.0
        self._last_n = 0
        self._rate = 0.0

    # ── rendering ──────────────────────────────────────────────────────────

    def _format_rate(self, rate: float) -> str:
        if rate < 1:
            return f'{1/rate:.2f}s/{self.unit}'
        elif self.unit_scale:
            if rate >= 1e9:
                return f'{rate/1e9:.2f}G{self.unit}/s'
            elif rate >= 1e6:
                return f'{rate/1e6:.2f}M{self.unit}/s'
            elif rate >= 1e3:
                return f'{rate/1e3:.2f}k{self.unit}/s'
        return f'{rate:.2f}{self.unit}/s'

    def _format_time(self, seconds: float) -> str:
        if seconds == float('inf') or math.isnan(seconds):
            return '?'
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f'{h:d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

    def _print_progress(self, force: bool = False) -> None:
        elapsed = time.time() - self._start_t
        # Build bar
        prefix = f'{self.desc}: ' if self.desc else ''
        n_str = str(self.n)
        total_str = str(self.total) if self.total else '?'
        pct = f'{100*self.n/self.total:.0f}%' if self.total else ''
        rate_str = self._format_rate(self._rate) if self._rate else '?'
        elapsed_str = self._format_time(elapsed)
        if self.total and self._rate:
            eta_secs = (self.total - self.n) / self._rate
            eta_str = self._format_time(eta_secs)
        else:
            eta_str = '?'
        # Progress bar fill
        bar = ''
        if self.total:
            fill = int(20 * self.n / self.total)
            bar = '[' + '█' * fill + '░' * (20 - fill) + '] '
        postfix_str = ''
        if self._postfix:
            postfix_str = ', '.join(
                f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                for k, v in self._postfix.items()
            )
            postfix_str = ' [' + postfix_str + ']'
        line = (f'\r{prefix}{pct} {bar}{n_str}/{total_str} '
                f'[{elapsed_str}<{eta_str}, {rate_str}]{postfix_str}')
        self._file.write(line[:self.ncols])
        self._file.flush()

    # ── class methods ───────────────────────────────────────────────────────

    @classmethod
    def write(cls, s: str, file=None, end: str = '\n') -> None:
        f = file or sys.stderr
        f.write(f'\r{s}{end}')
        f.flush()


def track(iterable: Iterable, description: str = '', total: Optional[int] = None,
          **kwargs) -> tqdm:
    """Rich-compatible 'track' alias for tqdm."""
    return tqdm(iterable, desc=description, total=total, **kwargs)


class trange(tqdm):
    """Shortcut: tqdm(range(n))."""

    def __init__(self, n: int, **kwargs):
        super().__init__(range(n), **kwargs)


__all__ = ['tqdm', 'trange', 'track']
