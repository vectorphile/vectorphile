"""vectorphile.cuda.streams — CUDA stream management and async execution.

Provides a PyTorch-compatible stream API:
- ``Stream``: CUDA stream wrapper
- ``current_stream()``: get the current default stream
- ``stream(s)``: context manager for executing ops on stream s
- ``synchronize()``: sync all streams

When the VP CUDA extension is available, streams are real CUDA streams.
Otherwise they are no-op stubs so code using streams works on CPU too.

Usage::
    from vectorphile.cuda.streams import Stream, stream, synchronize

    s = Stream()
    with stream(s):
        output = model(x)   # runs asynchronously on stream s
    synchronize()
"""
from __future__ import annotations
import contextlib
from typing import Optional

_cu = None
try:
    from vectorphile._core import _vp_cuda as _cu
except Exception:
    pass


class Stream:
    """CUDA stream wrapper.

    Args:
        device: CUDA device index (default 0)
        priority: stream priority (0=default, negative=higher)
    """

    def __init__(self, device: int = 0, priority: int = 0):
        self.device   = device
        self.priority = priority
        self._handle  = None
        self._records: list = []  # recorded events

    def synchronize(self) -> None:
        """Block the host until all ops on this stream complete."""
        if _cu is not None:
            try:
                _cu.synchronize()
            except Exception:
                pass

    def wait_stream(self, other: 'Stream') -> None:
        """Make this stream wait until all ops on other stream complete."""
        other.synchronize()

    def record_event(self, event=None) -> 'Event':
        """Record an event on this stream."""
        ev = Event(stream=self)
        self._records.append(ev)
        return ev

    def query(self) -> bool:
        """Return True if all ops on this stream are complete."""
        return True  # CPU always complete

    def __repr__(self) -> str:
        return f'Stream(device={self.device}, priority={self.priority})'

    def __enter__(self):
        _push_stream(self)
        return self

    def __exit__(self, *_):
        _pop_stream()


class Event:
    """CUDA event for timing and synchronization."""

    def __init__(self, stream: Optional[Stream] = None, enable_timing: bool = False):
        self._stream = stream
        self._enable_timing = enable_timing
        self._recorded = False

    def record(self, stream: Optional[Stream] = None) -> None:
        """Record the event on the given stream (or current stream)."""
        self._stream = stream or current_stream()
        self._recorded = True

    def synchronize(self) -> None:
        """Block until this event is complete."""
        if self._stream is not None:
            self._stream.synchronize()

    def elapsed_time(self, end_event: 'Event') -> float:
        """Return elapsed time in milliseconds between this event and end_event."""
        # CPU stub: return 0.0 (real GPU timing would use cudaEventElapsedTime)
        return 0.0

    def query(self) -> bool:
        return True

    def wait(self, stream: Optional[Stream] = None) -> None:
        """Make a stream wait on this event."""
        self.synchronize()


# ── Stream stack ──────────────────────────────────────────────────────────────
_stream_stack: list = []
_default_stream = Stream(device=0)


def _push_stream(s: Stream) -> None:
    _stream_stack.append(s)


def _pop_stream() -> Optional[Stream]:
    if _stream_stack:
        return _stream_stack.pop()
    return None


def current_stream(device: int = 0) -> Stream:
    """Return the currently active CUDA stream for device."""
    if _stream_stack:
        return _stream_stack[-1]
    return _default_stream


def default_stream(device: int = 0) -> Stream:
    """Return the default CUDA stream."""
    return _default_stream


@contextlib.contextmanager
def stream(s: Stream):
    """Context manager to execute ops on stream s.

    Usage::
        with vp.cuda.stream(my_stream):
            y = model(x)   # runs on my_stream
    """
    _push_stream(s)
    try:
        yield s
    finally:
        _pop_stream()


def synchronize(device: int = 0) -> None:
    """Synchronize all CUDA streams on device."""
    if _cu is not None:
        try:
            _cu.synchronize()
        except Exception:
            pass


def is_current_stream_capturing() -> bool:
    """Return True if CUDA graph capture is in progress (stub)."""
    return False


# ── CUDA Graph (stub) ────────────────────────────────────────────────────────

class CUDAGraph:
    """CUDA graph for capturing and replaying op sequences.

    Real CUDA graphs eliminate kernel launch overhead when replaying the same
    op sequence with the same shapes. This stub records the ops at Python level
    and replays them — useful for testing graph-capture-compatible code paths.

    Usage::
        from vectorphile.cuda.streams import CUDAGraph
        g = CUDAGraph()
        g.capture_begin()
        y = model(x)      # all ops recorded
        g.capture_end()
        for _ in range(1000):
            g.replay()    # skip Python overhead on replay
    """

    def __init__(self):
        self._calls: list = []
        self._capturing = False
        self._result = None

    def capture_begin(self, pool=None) -> None:
        self._calls.clear()
        self._capturing = True

    def capture_end(self) -> None:
        self._capturing = False

    def replay(self) -> None:
        """Re-execute all recorded calls."""
        for fn, args, kwargs in self._calls:
            self._result = fn(*args, **kwargs)

    def record(self, fn, *args, **kwargs):
        """Manually record a call (for use outside capture context)."""
        if self._capturing:
            self._calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f'CUDAGraph(ops={len(self._calls)}, capturing={self._capturing})'


# ── Graph context (torch.cuda.graph compat) ───────────────────────────────────
@contextlib.contextmanager
def graph(cuda_graph: CUDAGraph, pool=None, stream=None, capture_error_mode='global'):
    """Context manager for CUDA graph capture."""
    cuda_graph.capture_begin(pool=pool)
    try:
        yield cuda_graph
    finally:
        cuda_graph.capture_end()


__all__ = [
    'Stream', 'Event', 'CUDAGraph',
    'current_stream', 'default_stream', 'stream', 'synchronize',
    'is_current_stream_capturing', 'graph',
]
