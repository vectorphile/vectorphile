"""vectorphile.profiler — Performance profiling for VP models and ops.

Provides:
- ``profile()``: context manager that records op timing + memory stats
- ``ProfileEvent``: individual recorded event
- ``ProfileSummary``: aggregated stats per op type
- ``OpTimer``: lightweight decorator for timing any function
- ``MemoryTracker``: tracks peak Python-level memory per region

Usage::
    from vectorphile.profiler import profile

    with profile() as prof:
        output = model(x)

    prof.print_summary()
    df = prof.to_dict()
"""
from __future__ import annotations
import time
import sys
import os
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple


class ProfileEvent:
    """Single timed event captured during profiling."""

    __slots__ = ('name', 'start_us', 'end_us', 'mem_before', 'mem_after', 'extra')

    def __init__(self, name: str, start_us: float = 0.0, end_us: float = 0.0,
                 mem_before: int = 0, mem_after: int = 0, extra: Optional[dict] = None):
        self.name = name
        self.start_us = start_us
        self.end_us = end_us
        self.mem_before = mem_before
        self.mem_after = mem_after
        self.extra = extra or {}

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0

    def __repr__(self) -> str:
        return f'ProfileEvent({self.name!r}, {self.duration_ms:.3f}ms)'


class ProfileSummary:
    """Aggregated statistics per operation name."""

    def __init__(self, events: List[ProfileEvent]):
        self.events = events
        self._stats: Dict[str, dict] = {}
        self._compute_stats()

    def _compute_stats(self) -> None:
        from collections import defaultdict
        groups: Dict[str, List[float]] = defaultdict(list)
        for e in self.events:
            groups[e.name].append(e.duration_us)
        for name, times in sorted(groups.items(), key=lambda x: -sum(x[1])):
            total = sum(times)
            n = len(times)
            avg = total / n
            mn = min(times); mx = max(times)
            self._stats[name] = {
                'count': n, 'total_us': total, 'avg_us': avg,
                'min_us': mn, 'max_us': mx,
                'total_ms': total / 1000.0, 'avg_ms': avg / 1000.0,
            }

    def print_summary(self, top_n: int = 20) -> None:
        total_time = sum(e.duration_us for e in self.events)
        rows = sorted(self._stats.items(), key=lambda x: -x[1]['total_us'])
        print(f'\n{"─"*72}')
        print(f'{"Op Name":<36} {"Count":>6} {"Total(ms)":>10} {"Avg(ms)":>9} {"% Time":>7}')
        print(f'{"─"*72}')
        for name, st in rows[:top_n]:
            pct = 100.0 * st['total_us'] / max(total_time, 1)
            print(f'{name:<36} {st["count"]:>6} {st["total_ms"]:>10.3f} {st["avg_ms"]:>9.3f} {pct:>6.1f}%')
        print(f'{"─"*72}')
        print(f'{"TOTAL":<36} {"":>6} {total_time/1000.0:>10.3f}')

    def to_dict(self) -> dict:
        return dict(self._stats)

    def top(self, n: int = 5) -> List[Tuple[str, dict]]:
        return sorted(self._stats.items(), key=lambda x: -x[1]['total_us'])[:n]


class _Profiler:
    """Internal profiler object returned by profile()."""

    def __init__(self, enabled: bool = True, record_shapes: bool = False):
        self.enabled = enabled
        self.record_shapes = record_shapes
        self.events: List[ProfileEvent] = []
        self._stack: List[ProfileEvent] = []

    def _rss_bytes(self) -> int:
        """Current process RSS in bytes (best-effort)."""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            pass
        try:
            with open(f'/proc/{os.getpid()}/status') as f:
                for line in f:
                    if line.startswith('VmRSS'):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return 0

    def record_begin(self, name: str) -> int:
        if not self.enabled:
            return -1
        ev = ProfileEvent(name, start_us=time.perf_counter() * 1e6,
                          mem_before=self._rss_bytes())
        self._stack.append(ev)
        return id(ev)

    def record_end(self) -> None:
        if not self.enabled or not self._stack:
            return
        ev = self._stack.pop()
        ev.end_us = time.perf_counter() * 1e6
        ev.mem_after = self._rss_bytes()
        self.events.append(ev)

    @contextmanager
    def record(self, name: str):
        self.record_begin(name)
        try:
            yield
        finally:
            self.record_end()

    def print_summary(self, top_n: int = 20) -> None:
        ProfileSummary(self.events).print_summary(top_n)

    def summary(self) -> ProfileSummary:
        return ProfileSummary(self.events)

    def to_dict(self) -> dict:
        return ProfileSummary(self.events).to_dict()

    def clear(self) -> None:
        self.events.clear()
        self._stack.clear()

    def flame_graph(self, output_path: str) -> None:
        """Generate and save a flame graph SVG from recorded events."""
        fg = FlameGraph(self.events)
        fg.save(output_path)


# Public alias
Profiler = _Profiler


# Global profiler singleton (disabled by default)
_GLOBAL_PROFILER = _Profiler(enabled=False)


@contextmanager
def profile(enabled: bool = True, record_shapes: bool = False):
    """Context manager that enables profiling and returns a _Profiler.

    Usage::
        with profile() as prof:
            y = model(x)
        prof.print_summary()
    """
    global _GLOBAL_PROFILER
    prev = _GLOBAL_PROFILER
    _GLOBAL_PROFILER = _Profiler(enabled=enabled, record_shapes=record_shapes)
    try:
        yield _GLOBAL_PROFILER
    finally:
        _GLOBAL_PROFILER = prev


def get_profiler() -> _Profiler:
    """Return the currently active global profiler."""
    return _GLOBAL_PROFILER


def record_op(name: str):
    """Decorator: wraps a function and records its time in the active profiler."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            prof = _GLOBAL_PROFILER
            if not prof.enabled:
                return fn(*args, **kwargs)
            prof.record_begin(name)
            try:
                result = fn(*args, **kwargs)
            finally:
                prof.record_end()
            return result
        return wrapper
    return decorator


class OpTimer:
    """Context manager / callable for ad-hoc timing of arbitrary code blocks.

    Usage::
        from vectorphile.profiler import OpTimer
        timer = OpTimer('my_op')
        with timer:
            expensive_computation()
        print(timer.last_ms)
    """

    def __init__(self, name: str = 'op'):
        self.name = name
        self.last_ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.last_ms = (time.perf_counter() - self._t0) * 1000.0

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


class MemoryTracker:
    """Track Python object memory across code regions.

    Uses ``sys.getsizeof`` on tracked objects for lightweight estimation.
    Not a replacement for a real memory profiler but useful for quick checks.

    Usage::
        from vectorphile.profiler import MemoryTracker
        mt = MemoryTracker()
        mt.snapshot('before')
        x = model(big_input)
        mt.snapshot('after')
        mt.print_diff()
    """

    def __init__(self):
        self._snapshots: Dict[str, int] = {}

    def _current_py_bytes(self) -> int:
        try:
            import gc
            total = 0
            for obj in gc.get_objects():
                try:
                    total += sys.getsizeof(obj)
                except Exception:
                    pass
            return total
        except Exception:
            return 0

    def snapshot(self, label: str) -> int:
        b = self._current_py_bytes()
        self._snapshots[label] = b
        return b

    def print_diff(self) -> None:
        labels = list(self._snapshots.keys())
        if len(labels) < 2:
            print('Need at least 2 snapshots.')
            return
        for i in range(1, len(labels)):
            a, b = labels[i-1], labels[i]
            diff = self._snapshots[b] - self._snapshots[a]
            sign = '+' if diff >= 0 else ''
            print(f'{a} → {b}: {sign}{diff/1024/1024:.2f} MB')


class FlameGraph:
    """Generate an SVG flame graph from profiler timing data.

    Uses only stdlib (xml.etree.ElementTree + colorsys).
    Each bar represents an op; width is proportional to total time.
    Hover tooltips show op name + ms via inline SVG <title> elements.

    Usage::
        with profile() as prof:
            model(x)
        fg = FlameGraph(prof.events)
        fg.save('flame.svg')
    """

    _WIDTH = 1200
    _ROW_H = 24
    _FONT_SZ = 11
    _PAD = 4

    def __init__(self, events):
        from collections import defaultdict
        groups: Dict[str, float] = defaultdict(float)
        if isinstance(events, dict):
            # Accept {name: ms} dict directly
            for k, v in events.items():
                groups[k] += float(v)
        else:
            for e in events:
                if isinstance(e, dict):
                    groups[e.get('name', '?')] += float(e.get('duration_ms', 0))
                else:
                    groups[e.name] += e.duration_ms
        self._total_ms = sum(groups.values()) or 1.0
        # Sort descending
        self._rows: List[Tuple[str, float]] = sorted(
            groups.items(), key=lambda x: -x[1]
        )

    def _heat_color(self, fraction: float) -> str:
        import colorsys
        # Hot = red-orange, cold = blue
        hue = 0.67 * (1.0 - min(1.0, fraction))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'

    def to_svg(self) -> str:
        import xml.etree.ElementTree as ET
        height = (len(self._rows) + 2) * self._ROW_H + 2 * self._PAD
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(self._WIDTH),
            'height': str(height),
            'style': 'font-family:monospace;font-size:11px;background:#1e1e2e',
        })
        # Title
        title_el = ET.SubElement(svg, 'text', {
            'x': str(self._PAD), 'y': str(self._ROW_H - 4),
            'fill': '#cdd6f4', 'font-size': '13',
        })
        title_el.text = f'vectorphile Flame Graph — total {self._total_ms:.2f} ms'
        for row_idx, (name, ms) in enumerate(self._rows):
            frac = ms / self._total_ms
            bar_w = max(2, int(frac * (self._WIDTH - 2 * self._PAD)))
            y = self._PAD + (row_idx + 1) * self._ROW_H
            color = self._heat_color(frac)
            g_el = ET.SubElement(svg, 'g')
            # Background rect
            rect = ET.SubElement(g_el, 'rect', {
                'x': str(self._PAD), 'y': str(y),
                'width': str(bar_w), 'height': str(self._ROW_H - 2),
                'fill': color, 'rx': '2',
            })
            # Tooltip
            tt = ET.SubElement(rect, 'title')
            tt.text = f'{name}: {ms:.3f} ms ({frac*100:.1f}%)'
            # Label (clipped to bar width)
            max_chars = max(0, bar_w // (self._FONT_SZ * 6 // 10))
            label = name if len(name) <= max_chars else name[:max_chars - 1] + '…'
            if max_chars > 2:
                txt = ET.SubElement(g_el, 'text', {
                    'x': str(self._PAD + 3), 'y': str(y + self._ROW_H - 7),
                    'fill': '#11111b', 'font-size': str(self._FONT_SZ),
                    'clip-path': f'inset(0 {self._WIDTH - self._PAD - bar_w}px 0 0)',
                })
                txt.text = label
        return ET.tostring(svg, encoding='unicode')

    def save(self, path: str) -> None:
        """Save flame graph as SVG to path."""
        svg_str = self.to_svg()
        with open(path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(svg_str)


# Patch flame_graph method into _Profiler
def _profiler_flame_graph(self, output_path: str) -> FlameGraph:
    """Generate and save an SVG flame graph; also returns the FlameGraph object."""
    fg = FlameGraph(self.events)
    fg.save(output_path)
    return fg

_Profiler.flame_graph = _profiler_flame_graph


__all__ = [
    'profile',
    'get_profiler',
    'record_op',
    'OpTimer',
    'MemoryTracker',
    'ProfileEvent',
    'ProfileSummary',
    'FlameGraph',
    '_Profiler',
]
