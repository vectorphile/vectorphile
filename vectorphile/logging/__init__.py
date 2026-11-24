"""vectorphile.logging — Training metrics logging and TensorBoard-compatible writers.

Provides:
- MetricLogger: accumulate and report training metrics per epoch
- SummaryWriter: TensorBoard-compatible event file writer (scalar, histogram, image)
- CSVLogger: log metrics to CSV
- WandB-compatible interface (no wandb required, writes locally)

No external dependencies.

Usage::

    from vectorphile.logging import SummaryWriter, MetricLogger

    writer = SummaryWriter('runs/exp1')
    writer.add_scalar('loss/train', 0.5, global_step=100)
    writer.add_scalar('acc/val', 0.92, global_step=100)
    writer.close()

    # Epoch-level
    logger = MetricLogger(delimiter='  ')
    logger.update(loss=0.5, acc=0.9)
    print(logger)
"""
from __future__ import annotations
import os
import csv
import json
import time
import struct
import hashlib
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# SmoothedValue — running average over a window
# ---------------------------------------------------------------------------

class SmoothedValue:
    """Track a series of values and provide access to smoothed, running averages."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        self.deque: deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or '{median:.4f} ({global_avg:.4f})'

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = sorted(self.deque)
        return d[len(d) // 2] if d else 0.0

    @property
    def avg(self) -> float:
        return sum(self.deque) / len(self.deque) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median, avg=self.avg,
            global_avg=self.global_avg, max=self.max, value=self.value
        )


# ---------------------------------------------------------------------------
# MetricLogger
# ---------------------------------------------------------------------------

class MetricLogger:
    """Accumulate and display training metrics.

    Compatible with FAIR's detectron2/fvcore MetricLogger API.

    Usage::

        logger = MetricLogger()
        for batch in loader:
            loss = train_step(batch)
            logger.update(loss=loss)
        logger.print(epoch=1)
    """

    def __init__(self, delimiter: str = '\t', window_size: int = 20):
        self.meters: Dict[str, SmoothedValue] = defaultdict(
            lambda: SmoothedValue(window_size=window_size))
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self.meters[k].update(float(v))

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.__dict__:
            return self.__dict__[attr]
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f'No metric {attr!r}')

    def __str__(self) -> str:
        parts = []
        for k, v in self.meters.items():
            parts.append(f'{k}: {v.median:.4f} ({v.global_avg:.4f})')
        return self.delimiter.join(parts)

    def print(self, epoch: Optional[int] = None, header: str = '') -> None:
        prefix = f'[Epoch {epoch}] ' if epoch is not None else ''
        if header:
            prefix += header + '  '
        print(prefix + str(self))

    def global_averages(self) -> Dict[str, float]:
        return {k: v.global_avg for k, v in self.meters.items()}

    def reset(self) -> None:
        self.meters.clear()


# ---------------------------------------------------------------------------
# CSVLogger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Log metrics to a CSV file per epoch/step.

    Usage::

        csv_log = CSVLogger('logs/metrics.csv')
        csv_log.log({'epoch': 1, 'train_loss': 0.5, 'val_acc': 0.9})
        csv_log.close()
    """

    def __init__(self, filepath: str, separator: str = ',', append: bool = False):
        self.filepath = filepath
        self.separator = separator
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mode = 'a' if append else 'w'
        self._file = open(filepath, mode, newline='')
        self._writer: Optional[csv.DictWriter] = None

    def log(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(row.keys()),
                delimiter=self.separator)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# SummaryWriter (TensorBoard-compatible, writes JSON events)
# ---------------------------------------------------------------------------

class SummaryWriter:
    """TensorBoard-compatible SummaryWriter.

    Writes scalar, histogram, image, and text events to a JSON lines file.
    Can be read back with SummaryReader.

    Usage::

        writer = SummaryWriter(log_dir='runs/exp1')
        for step in range(1000):
            writer.add_scalar('loss', 0.5 / (step + 1), step)
        writer.close()
    """

    def __init__(self, log_dir: str = 'runs', comment: str = '',
                 purge_step: Optional[int] = None, flush_secs: int = 120,
                 filename_suffix: str = ''):
        run_name = f'{int(time.time())}{comment}{filename_suffix}'
        self.log_dir = os.path.join(log_dir, run_name) if comment else log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._path = os.path.join(self.log_dir, 'events.jsonl')
        self._file = open(self._path, 'w')
        self._global_step = 0

    def _write_event(self, tag: str, type_: str, value: Any,
                     global_step: Optional[int] = None,
                     walltime: Optional[float] = None) -> None:
        event = {
            'wall_time': walltime or time.time(),
            'step': global_step if global_step is not None else self._global_step,
            'tag': tag,
            'type': type_,
            'value': value,
        }
        self._file.write(json.dumps(event) + '\n')

    def add_scalar(self, tag: str, scalar_value: float,
                   global_step: Optional[int] = None,
                   walltime: Optional[float] = None) -> None:
        self._write_event(tag, 'scalar', float(scalar_value), global_step, walltime)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict,
                    global_step: Optional[int] = None) -> None:
        for k, v in tag_scalar_dict.items():
            self.add_scalar(f'{main_tag}/{k}', v, global_step)

    def add_histogram(self, tag: str, values, global_step: Optional[int] = None,
                      bins: int = 64) -> None:
        # Pure Python histogram
        flat = [float(v) for v in (values if isinstance(values, list) else
                (values.to_flat_list() if hasattr(values, 'to_flat_list') else [values]))]
        if not flat: return
        lo, hi = min(flat), max(flat)
        if lo == hi: hi = lo + 1e-8
        step = (hi - lo) / bins
        counts = [0] * bins
        for v in flat:
            idx = min(int((v - lo) / step), bins - 1)
            counts[idx] += 1
        edges = [lo + i*step for i in range(bins+1)]
        self._write_event(tag, 'histogram',
                          {'counts': counts, 'edges': edges},
                          global_step)

    def add_image(self, tag: str, img_tensor, global_step: Optional[int] = None,
                  dataformats: str = 'CHW') -> None:
        sh = img_tensor.shape if hasattr(img_tensor, 'shape') else (len(img_tensor),)
        self._write_event(tag, 'image', {'shape': list(sh)}, global_step)

    def add_text(self, tag: str, text_string: str,
                 global_step: Optional[int] = None) -> None:
        self._write_event(tag, 'text', text_string, global_step)

    def add_hparams(self, hparam_dict: dict, metric_dict: dict) -> None:
        self._write_event('hparams', 'hparams',
                          {'hparams': hparam_dict, 'metrics': metric_dict})

    def add_graph(self, model, input_to_model=None) -> None:
        from vectorphile.profiler import count_parameters
        n_params = count_parameters(model)
        self._write_event('graph', 'graph', {'n_params': n_params,
                                              'model': type(model).__name__})

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Properties
    @property
    def file_writer(self):
        return self._file


class SummaryReader:
    """Read events written by SummaryWriter."""

    def __init__(self, path: str):
        self.path = path

    def scalars(self, tag: Optional[str] = None) -> List[dict]:
        events = []
        with open(self.path) as f:
            for line in f:
                ev = json.loads(line)
                if ev['type'] == 'scalar':
                    if tag is None or ev['tag'] == tag:
                        events.append(ev)
        return events

    def all_events(self) -> List[dict]:
        with open(self.path) as f:
            return [json.loads(l) for l in f]


# ---------------------------------------------------------------------------
# WandB-like interface (logs locally, no network)
# ---------------------------------------------------------------------------

class _WandbLocal:
    """wandb-compatible API stub that writes locally."""

    def __init__(self):
        self._run = None
        self._writer: Optional[SummaryWriter] = None
        self._step = 0

    def init(self, project: str = 'default', name: Optional[str] = None,
             config: Optional[dict] = None, **kwargs) -> '_WandbLocal':
        log_dir = os.path.join('wandb', project, name or f'run-{int(time.time())}')
        self._writer = SummaryWriter(log_dir)
        if config:
            self._writer.add_hparams(config, {})
        return self

    def log(self, data: dict, step: Optional[int] = None, **kwargs) -> None:
        if self._writer is None:
            self.init()
        s = step if step is not None else self._step
        for k, v in data.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, s)
        if step is None:
            self._step += 1

    def finish(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    def config(self):
        return {}


wandb = _WandbLocal()


__all__ = [
    'SmoothedValue', 'MetricLogger', 'CSVLogger',
    'SummaryWriter', 'SummaryReader',
    'wandb',
]
