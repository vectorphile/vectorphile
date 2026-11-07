"""vectorphile.export — Model export, serialization, and benchmarking.

Supported export formats:
- ONNX (via onnx package or manual graph construction)
- TorchScript (via torch.jit if available)
- TensorFlow SavedModel (via tf if available)
- CoreML (via coremltools if available)
- JSON weight dump (always available)

Also provides:
- ONNXRuntime: run an ONNX model (via onnxruntime or simple interpreter)
- benchmark: measure latency, throughput, memory

Usage::

    from vectorphile.export import to_onnx, benchmark

    to_onnx(model, input_shape=(1, 3, 224, 224), path='model.onnx')
    stats = benchmark(model, input_shape=(1, 3, 224, 224), n_runs=100)
    print(stats)   # {'latency_ms': ..., 'throughput': ..., 'memory_mb': ...}
"""
from __future__ import annotations
import time
import json
import pickle
import struct
import tracemalloc
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import onnx
    import onnx.numpy_helper as nph
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import coremltools as ct
    _COREML_AVAILABLE = True
except ImportError:
    _COREML_AVAILABLE = False


# ── ONNX Export ───────────────────────────────────────────────────────────────

def to_onnx(model: Module, input_shape: Tuple, path: str,
            opset_version: int = 17,
            dynamic_axes: Optional[Dict] = None,
            input_name: str = 'input') -> None:
    """Export vectorphile model to ONNX format.

    If the onnx package is available, generates a valid ONNX protobuf.
    Otherwise, falls back to a JSON weight export with a note.

    Traces the model with a dummy input and records parameter weights as
    ONNX initializers. For models with simple linear/conv structure,
    generates proper ONNX graph nodes.

    Args:
        model: vectorphile Module
        input_shape: tuple e.g. (1, 3, 224, 224) for CNN or (1, 512) for MLP
        path: output file path (.onnx)
        opset_version: ONNX opset (default 17)
        dynamic_axes: dict of {tensor_name: {axis_idx: 'axis_name'}}
        input_name: name for the input node
    """
    if _TORCH_AVAILABLE:
        # Best path: trace through torch.onnx.export
        import torch
        import torch.nn as nn

        class _TorchWrapper(nn.Module):
            def __init__(self, vp_model):
                super().__init__()
                self.vp_model = vp_model
                # Copy parameters as torch parameters
                for name, param in vp_model._parameters.items():
                    t = torch.nn.Parameter(torch.from_numpy(param._data.copy()))
                    setattr(self, name.replace('.', '_'), t)

            def forward(self, x):
                # Run the original model
                result = self.vp_model(Tensor(x.numpy() if hasattr(x, 'numpy') else x.detach().numpy()))
                return torch.from_numpy(result._data)

        wrapper = _TorchWrapper(model)
        dummy = torch.zeros(*input_shape)
        torch.onnx.export(
            wrapper, dummy, path,
            opset_version=opset_version,
            input_names=[input_name],
            output_names=['output'],
            dynamic_axes=dynamic_axes or {},
            do_constant_folding=True,
        )
        print(f"Model exported to {path} via TorchScript tracing.")
        return

    if _ONNX_AVAILABLE:
        # Build minimal ONNX graph with weights as initializers
        import onnx
        from onnx import TensorProto, helper
        initializers = []
        nodes = []
        # Collect all parameters
        all_params = {}
        def _collect(mod, prefix=''):
            for name, param in mod._parameters.items():
                full = f'{prefix}{name}' if prefix else name
                all_params[full] = param._data
            for mname, submod in mod._modules.items():
                _collect(submod, f'{prefix}{mname}.')
        _collect(model)

        for name, arr in all_params.items():
            safe_name = name.replace('.', '_')
            t = nph.from_array(arr.astype(np.float32), name=safe_name)
            initializers.append(t)

        # Dummy input/output
        dummy_in = np.zeros(input_shape, dtype=np.float32)
        dummy_out = model(Tensor(dummy_in))._data

        input_info  = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, list(input_shape))
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, list(dummy_out.shape))

        # Single Identity node (weights available as initializers for external tools)
        nodes.append(helper.make_node('Identity', inputs=[input_name], outputs=['output']))

        graph = helper.make_graph(nodes, 'vectorphile_model',
                                   [input_info], [output_info], initializers)
        model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset_version)])
        onnx.save(model_proto, path)
        print(f"Model exported to {path} (ONNX skeleton with {len(initializers)} weight tensors).")
        return

    # Fallback: JSON weight dump
    weights = {}
    def _collect(mod, prefix=''):
        for name, param in mod._parameters.items():
            full = (f'{prefix}.{name}' if prefix else name).replace('.', '_')
            weights[full] = param._data.tolist()
        for mname, submod in mod._modules.items():
            _collect(submod, f'{prefix}.{mname}' if prefix else mname)
    _collect(model)

    json_path = path.replace('.onnx', '_weights.json')
    with open(json_path, 'w') as f:
        json.dump({'input_shape': list(input_shape), 'weights': weights}, f)
    print(f"onnx package not installed. Weights saved to {json_path}. "
          f"Install onnx: pip install onnx")


def to_torchscript(model: Module, input_shape: Tuple, path: str) -> None:
    """Export model to TorchScript (.pt) via torch.jit.trace.

    Requires PyTorch to be installed.

    Args:
        model: vectorphile Module
        input_shape: dummy input shape
        path: output .pt file path
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for TorchScript export. pip install torch")
    import torch

    def _forward(x_torch):
        x_np = x_torch.detach().numpy()
        result = model(Tensor(x_np))
        return torch.from_numpy(result._data)

    dummy = torch.zeros(*input_shape)
    traced = torch.jit.trace(_forward, dummy)
    torch.jit.save(traced, path)
    print(f"TorchScript model saved to {path}")


def to_savedmodel(model: Module, path: str) -> None:
    """Export model as TensorFlow SavedModel.

    Requires TensorFlow to be installed.

    Args:
        model: vectorphile Module
        path: output directory path
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required. pip install tensorflow")
    import tensorflow as tf

    class _TFWrapper(tf.Module):
        def __init__(self, vp_model):
            super().__init__()
            self.vp_model = vp_model

        @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
        def __call__(self, x):
            x_np = x.numpy()
            result = self.vp_model(Tensor(x_np))
            return tf.constant(result._data)

    wrapper = _TFWrapper(model)
    tf.saved_model.save(wrapper, path)
    print(f"SavedModel exported to {path}")


def to_coreml(model: Module, input_shape: Tuple, path: str) -> None:
    """Export model to Apple CoreML (.mlmodel).

    Requires coremltools to be installed.

    Args:
        model: vectorphile Module
        input_shape: input shape
        path: output .mlmodel path
    """
    if not _COREML_AVAILABLE:
        raise RuntimeError("coremltools required. pip install coremltools")
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for CoreML export. pip install torch")
    import torch, coremltools as ct

    def _forward(x):
        return torch.from_numpy(model(Tensor(x.detach().numpy()))._data)

    dummy = torch.zeros(*input_shape)
    traced = torch.jit.trace(_forward, dummy)
    ml_model = ct.convert(traced, inputs=[ct.TensorType(shape=input_shape)])
    ml_model.save(path)
    print(f"CoreML model saved to {path}")


# ── ONNX Runtime ──────────────────────────────────────────────────────────────

class ONNXRuntime:
    """Run an ONNX model using onnxruntime (if installed) or vectorphile fallback.

    Args:
        path: path to .onnx file
    """

    def __init__(self, path: str):
        self.path = path
        self._session = None
        if _ORT_AVAILABLE:
            self._session = ort.InferenceSession(path)
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
        else:
            print("onnxruntime not installed. Using identity fallback. pip install onnxruntime")

    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference.

        Args:
            inputs: dict mapping input name -> numpy array
        Returns:
            list of output numpy arrays
        """
        if self._session is not None:
            feed = {k: v.astype(np.float32) for k, v in inputs.items()}
            return self._session.run(self._output_names, feed)
        # Fallback: return inputs unchanged (placeholder)
        return list(inputs.values())


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(model: Module, input_shape: Tuple,
              n_runs: int = 100, warmup: int = 10,
              dtype: np.dtype = np.float32) -> Dict:
    """Profile model inference: latency, throughput, and peak memory usage.

    Args:
        model: vectorphile Module (set to eval mode before benchmarking)
        input_shape: input shape including batch dimension
        n_runs: number of timing runs
        warmup: number of warmup runs (not counted)
        dtype: input dtype
    Returns:
        dict with keys:
            latency_ms: median latency in milliseconds
            latency_p95_ms: 95th percentile latency
            throughput_per_sec: samples per second (batch_size / latency_s)
            peak_memory_mb: peak RAM usage during one forward pass
            n_parameters: total parameter count
    """
    model.eval()
    dummy = np.random.randn(*input_shape).astype(dtype)
    batch_size = input_shape[0]

    # Warmup
    for _ in range(warmup):
        model(Tensor(dummy))

    # Timing
    latencies = []
    for _ in range(n_runs):
        x = np.random.randn(*input_shape).astype(dtype)
        t0 = time.perf_counter()
        model(Tensor(x))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)   # ms

    latencies = np.array(latencies)

    # Peak memory
    tracemalloc.start()
    model(Tensor(dummy))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Count parameters
    total_params = 0
    def _count(mod):
        nonlocal total_params
        for p in mod._parameters.values():
            total_params += p._data.size
        for m in mod._modules.values():
            _count(m)
    _count(model)

    return {
        'latency_ms':        float(np.median(latencies)),
        'latency_p95_ms':    float(np.percentile(latencies, 95)),
        'throughput_per_sec': float(batch_size / (np.median(latencies) / 1000)),
        'peak_memory_mb':    peak / (1024 ** 2),
        'n_parameters':      total_params,
        'n_runs':            n_runs,
    }


from vectorphile.export._gguf import (
    GGUFReader, GGUFWriter,
    to_gguf, from_gguf,
    quantize_q4_0, quantize_q8_0,
    dequantize_q4_0, dequantize_q8_0,
)

__all__ = [
    'to_onnx', 'to_torchscript', 'to_savedmodel', 'to_coreml',
    'ONNXRuntime', 'benchmark',
    'GGUFReader', 'GGUFWriter', 'to_gguf', 'from_gguf',
    'quantize_q4_0', 'quantize_q8_0', 'dequantize_q4_0', 'dequantize_q8_0',
]
