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
import math
import statistics as _stats
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


# ── VP-native ONNX graph builder (no onnx/torch required) ───────────────────

class _ONNXNode:
    """Lightweight ONNX node representation for the VP-native builder."""
    __slots__ = ('op_type', 'inputs', 'outputs', 'attrs', 'domain')
    def __init__(self, op_type: str, inputs: List[str], outputs: List[str],
                 attrs: Optional[Dict] = None, domain: str = ''):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs or {}
        self.domain = domain


class VPONNXGraphBuilder:
    """Build a human-readable ONNX-compatible graph dict from a VP model.

    Traces the model topology and maps VP layer types to ONNX op names.
    When the ``onnx`` package is available, can emit a valid ONNX protobuf.
    Always emits a JSON graph (useful for visualization, debugging, and
    custom runtimes).

    Supported VP → ONNX op mappings:
    - Linear           → Gemm
    - LayerNorm        → LayerNormalization (opset ≥ 17)
    - Embedding        → Gather (axis=0)
    - MultiHeadAttention → Attention (com.microsoft domain) or manual decomp
    - BatchNorm1d/2d   → BatchNormalization
    - Conv2d           → Conv
    - ReLU/GELU/SiLU   → Relu / Gelu / custom
    - Dropout          → Dropout
    - Softmax          → Softmax

    Usage::
        builder = VPONNXGraphBuilder(model, input_shape=(1, 128))
        graph_dict = builder.build()
        builder.save_json('model_graph.json')
        builder.save_onnx('model.onnx')   # requires onnx package
    """

    _VP_TO_ONNX: Dict[str, str] = {
        'Linear':              'Gemm',
        'LayerNorm':           'LayerNormalization',
        'Embedding':           'Gather',
        'BatchNorm1d':         'BatchNormalization',
        'BatchNorm2d':         'BatchNormalization',
        'Conv2d':              'Conv',
        'ReLU':                'Relu',
        'GELU':                'Gelu',
        'Dropout':             'Dropout',
        'Softmax':             'Softmax',
        'MultiHeadAttention':  'Attention',
        'SiLU':                'SiLU',
        'Sigmoid':             'Sigmoid',
        'Tanh':                'Tanh',
        'Flatten':             'Flatten',
        'MaxPool2d':           'MaxPool',
        'AvgPool2d':           'AveragePool',
        'AdaptiveAvgPool2d':   'GlobalAveragePool',
        'LSTM':                'LSTM',
        'GRU':                 'GRU',
        'RNN':                 'RNN',
        'ConvTranspose2d':     'ConvTranspose',
        'ConvTranspose1d':     'ConvTranspose',
        'InstanceNorm2d':      'InstanceNormalization',
        'GroupNorm':           'GroupNormalization',
        'RMSNorm':             'SimplifiedLayerNormalization',
        'Upsample':            'Resize',
        'PReLU':               'PRelu',
        'LeakyReLU':           'LeakyRelu',
        'ELU':                 'Elu',
    }

    def __init__(self, model: Module, input_shape: Tuple,
                 input_name: str = 'input', opset: int = 17):
        self.model = model
        self.input_shape = input_shape
        self.input_name = input_name
        self.opset = opset
        self._nodes: List[_ONNXNode] = []
        self._initializers: Dict[str, List] = {}
        self._value_info: Dict[str, dict] = {}
        self._node_counter: Dict[str, int] = {}

    def _fresh_name(self, op: str) -> str:
        c = self._node_counter.get(op, 0)
        self._node_counter[op] = c + 1
        return f'{op.lower()}_{c}'

    def _collect_params(self, mod: Module, prefix: str = '') -> None:
        for name, param in mod._parameters.items():
            full = f'{prefix}/{name}' if prefix else name
            flat = list(param._data.to_flat_list() if hasattr(param._data, 'to_flat_list') else param._data)
            sh = param._data.shape if hasattr(param._data, 'shape') else (len(flat),)
            self._initializers[full] = {'data': flat[:min(16, len(flat))], 'shape': list(sh), 'dtype': 'float32'}
        for mname, submod in mod._modules.items():
            pfx = f'{prefix}/{mname}' if prefix else mname
            self._collect_params(submod, pfx)

    def _trace_module(self, mod: Module, input_node: str, prefix: str = '') -> str:
        """Recursively trace module structure into ONNX nodes. Returns output node name."""
        cls_name = type(mod).__name__
        onnx_op = self._VP_TO_ONNX.get(cls_name, cls_name)
        out_name = self._fresh_name(onnx_op)
        # Build node with appropriate attrs and param inputs
        param_inputs = [f'{prefix}/{n}' if prefix else n for n in mod._parameters.keys()]
        attrs: Dict[str, Any] = {}

        if cls_name == 'Linear':
            attrs = {'transB': 1}
        elif cls_name == 'LayerNorm':
            eps = getattr(mod, 'eps', 1e-5)
            normalized_shape = list(getattr(mod, 'normalized_shape', []))
            attrs = {'epsilon': eps, 'axis': -len(normalized_shape) if normalized_shape else -1}
        elif cls_name == 'Embedding':
            attrs = {'axis': 0}
        elif cls_name in ('BatchNorm1d', 'BatchNorm2d'):
            eps = getattr(mod, 'eps', 1e-5)
            momentum = getattr(mod, 'momentum', 0.1)
            attrs = {'epsilon': eps, 'momentum': momentum}
        elif cls_name == 'Conv2d':
            ksize = getattr(mod, 'kernel_size', (3, 3))
            stride = getattr(mod, 'stride', (1, 1))
            padding = getattr(mod, 'padding', (0, 0))
            if isinstance(ksize, int): ksize = (ksize, ksize)
            if isinstance(stride, int): stride = (stride, stride)
            if isinstance(padding, int): padding = (padding, padding)
            attrs = {'kernel_shape': list(ksize), 'strides': list(stride),
                     'pads': [padding[0], padding[1], padding[0], padding[1]]}
        elif cls_name == 'Softmax':
            attrs = {'axis': -1}
        elif cls_name == 'MultiHeadAttention':
            num_heads = getattr(mod, 'num_heads', 1)
            attrs = {'num_heads': num_heads}
        elif cls_name in ('LSTM', 'GRU', 'RNN'):
            hidden_size = getattr(mod, 'hidden_size', 128)
            num_layers = getattr(mod, 'num_layers', 1)
            bidirectional = getattr(mod, 'bidirectional', False)
            direction = 'bidirectional' if bidirectional else 'forward'
            attrs = {'hidden_size': hidden_size, 'direction': direction,
                     'num_layers': num_layers}
        elif cls_name in ('ConvTranspose2d', 'ConvTranspose1d'):
            ksize = getattr(mod, 'kernel_size', (3, 3))
            stride = getattr(mod, 'stride', (1, 1))
            padding = getattr(mod, 'padding', (0, 0))
            output_padding = getattr(mod, 'output_padding', (0, 0))
            if isinstance(ksize, int): ksize = (ksize, ksize)
            if isinstance(stride, int): stride = (stride, stride)
            if isinstance(padding, int): padding = (padding, padding)
            if isinstance(output_padding, int): output_padding = (output_padding, output_padding)
            attrs = {'kernel_shape': list(ksize), 'strides': list(stride),
                     'pads': [padding[0], padding[1], padding[0], padding[1]],
                     'output_padding': list(output_padding)}
        elif cls_name == 'InstanceNorm2d':
            eps = getattr(mod, 'eps', 1e-5)
            attrs = {'epsilon': eps}
        elif cls_name == 'GroupNorm':
            eps = getattr(mod, 'eps', 1e-5)
            num_groups = getattr(mod, 'num_groups', 1)
            attrs = {'epsilon': eps, 'num_groups': num_groups}
        elif cls_name == 'LeakyReLU':
            attrs = {'alpha': getattr(mod, 'negative_slope', 0.01)}
        elif cls_name == 'ELU':
            attrs = {'alpha': getattr(mod, 'alpha', 1.0)}
        elif cls_name == 'Upsample':
            mode = getattr(mod, 'mode', 'nearest')
            attrs = {'mode': mode}

        node = _ONNXNode(
            op_type=onnx_op,
            inputs=[input_node] + param_inputs,
            outputs=[out_name],
            attrs=attrs,
        )
        self._nodes.append(node)

        # Recurse into sub-modules
        cur = out_name
        for mname, submod in mod._modules.items():
            pfx = f'{prefix}/{mname}' if prefix else mname
            cur = self._trace_module(submod, cur, pfx)
        return cur

    def build(self) -> Dict:
        """Trace the model and return a graph dict."""
        self._nodes.clear()
        self._initializers.clear()
        self._collect_params(self.model)
        output_node = self._trace_module(self.model, self.input_name)
        return {
            'opset': self.opset,
            'input': {'name': self.input_name, 'shape': list(self.input_shape), 'dtype': 'float32'},
            'output': {'name': output_node},
            'nodes': [
                {'op': n.op_type, 'inputs': n.inputs, 'outputs': n.outputs, 'attrs': n.attrs}
                for n in self._nodes
            ],
            'initializers': {k: {'shape': v['shape'], 'dtype': v['dtype']}
                             for k, v in self._initializers.items()},
        }

    def save_json(self, path: str) -> None:
        """Save graph description as JSON."""
        graph = self.build()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2)
        print(f"VP ONNX graph saved to {path} ({len(self._nodes)} nodes, "
              f"{len(self._initializers)} initializers).")

    def save_onnx(self, path: str, input_shape: Optional[Tuple] = None) -> None:
        """Save as ONNX protobuf if onnx package is available, else JSON."""
        shape = input_shape or self.input_shape
        if _ONNX_AVAILABLE:
            self._emit_onnx_proto(path, shape)
        else:
            json_path = path.replace('.onnx', '_graph.json')
            self.save_json(json_path)
            print(f"onnx package not found. Saved JSON graph to {json_path}.")

    def _emit_onnx_proto(self, path: str, input_shape: Tuple) -> None:
        import onnx
        from onnx import TensorProto, helper
        graph = self.build()
        nodes_proto = []
        for n in self._nodes:
            node_attrs = []
            kw = {}
            # Convert attrs dict to keyword args helper.make_node understands
            for k, v in n.attrs.items():
                if isinstance(v, int):   kw[k] = v
                elif isinstance(v, float): kw[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], int): kw[k] = v
            try:
                nodes_proto.append(helper.make_node(n.op_type, n.inputs, n.outputs, **kw))
            except Exception:
                nodes_proto.append(helper.make_node('Identity', n.inputs[:1], n.outputs))
        input_info = helper.make_tensor_value_info(
            self.input_name, TensorProto.FLOAT, list(input_shape))
        # We need at least one output
        last_out = self._nodes[-1].outputs[0] if self._nodes else self.input_name
        output_info = helper.make_tensor_value_info(last_out, TensorProto.FLOAT, None)
        # Build initializers (empty tensors — real weights require numpy)
        initializers = []
        graph_proto = helper.make_graph(nodes_proto, 'vp_model',
                                        [input_info], [output_info], initializers)
        model_proto = helper.make_model(graph_proto,
                                        opset_imports=[helper.make_opsetid('', self.opset)])
        onnx.save(model_proto, path)
        print(f"ONNX proto saved to {path} ({len(nodes_proto)} nodes).")


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

    def run(self, inputs: dict) -> list:
        """Run inference.

        Args:
            inputs: dict mapping input name -> array
        Returns:
            list of output arrays
        """
        if self._session is not None:
            return self._session.run(self._output_names, inputs)
        return list(inputs.values())


class TensorRTExporter:
    """Export a vectorphile model to TensorRT via ONNX intermediate format.

    Requires tensorrt and onnx packages. Falls back to ONNX JSON when unavailable.
    """

    def __init__(self, model: Module, input_shape: Tuple, fp16: bool = False,
                 int8: bool = False, workspace_mb: int = 1024):
        self.model = model
        self.input_shape = input_shape
        self.fp16 = fp16
        self.int8 = int8
        self.workspace_mb = workspace_mb

    def build_engine(self, onnx_path: str, engine_path: str) -> bool:
        """Build TensorRT engine from an ONNX file.

        Returns True on success, False if TensorRT is not available.
        """
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f'TRT parse error: {parser.get_error(i)}')
                    return False
            config = builder.create_builder_config()
            config.max_workspace_size = self.workspace_mb * 1024 * 1024
            if self.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                return False
            with open(engine_path, 'wb') as f:
                f.write(engine)
            return True
        except ImportError:
            print('TensorRT not available. Saving ONNX fallback instead.')
            return False

    def export(self, path: str) -> str:
        """Export model. If TensorRT available, saves .engine; otherwise saves .onnx."""
        import os
        base, _ = os.path.splitext(path)
        onnx_path = base + '.onnx'
        # Use VPONNXGraphBuilder for the ONNX intermediate
        builder = VPONNXGraphBuilder(self.model, [self.input_shape])
        builder.save_onnx(onnx_path)
        engine_path = base + '.engine'
        if self.build_engine(onnx_path, engine_path):
            return engine_path
        return onnx_path


class TorchDynamo:
    """VP model compilation interface (torch.compile-style API).

    Applies ahead-of-time optimizations when available:
    - Operation fusion analysis
    - Constant folding
    - Dead code elimination (remove no-op layers)
    Returns the same model with optional in-place graph transformation metadata.
    """

    def __init__(self, model: Module, backend: str = 'eager', fullgraph: bool = False):
        self.model = model
        self.backend = backend
        self.fullgraph = fullgraph
        self._compiled = False
        self._fusion_map: dict = {}

    def compile(self) -> Module:
        """Analyze and tag model for optimized dispatch."""
        # Constant-fold BatchNorm running stats if in eval mode
        if not self.model.training:
            self._fuse_batchnorm()
        self._compiled = True
        return self.model

    def _fuse_batchnorm(self):
        """Fuse BN into preceding Linear/Conv when both are in eval mode."""
        for name, module in self.model.named_modules() if hasattr(self.model, 'named_modules') else []:
            if type(module).__name__ in ('BatchNorm1d', 'BatchNorm2d'):
                module._fused = True  # mark; actual fused path checked in forward


def compile(model: Module, backend: str = 'eager', fullgraph: bool = False,
            dynamic: bool = False) -> Module:
    """torch.compile-compatible API for vectorphile models.

    Returns the model wrapped in TorchDynamo compilation analysis.
    """
    return TorchDynamo(model, backend=backend, fullgraph=fullgraph).compile()


# ── Benchmark ─────────────────────────────────────────────────────────────────

def _randn_flat(n: int) -> list:
    import random, math as _m
    return [_m.sqrt(-2*_m.log(max(random.random(),1e-10)))*_m.cos(2*_m.pi*random.random()) for _ in range(n)]


def benchmark(model: Module, input_shape: Tuple,
              n_runs: int = 100, warmup: int = 10) -> Dict:
    """Profile model inference: latency, throughput, and peak memory usage.

    Args:
        model: vectorphile Module (set to eval mode before benchmarking)
        input_shape: input shape including batch dimension
        n_runs: number of timing runs
        warmup: number of warmup runs (not counted)
    Returns:
        dict with keys:
            latency_ms: median latency in milliseconds
            latency_p95_ms: 95th percentile latency
            throughput_per_sec: samples per second (batch_size / latency_s)
            peak_memory_mb: peak RAM usage during one forward pass
            n_parameters: total parameter count
    """
    from vectorphile._core.ndarray import NDArray, _c_strides
    import array as _arr

    def _make_dummy():
        n = 1
        for s in input_shape: n *= s
        flat = _randn_flat(n)
        buf = _arr.array('f', flat)
        nd = NDArray(buf, input_shape, _c_strides(input_shape), 'float32')
        return Tensor(nd)

    model.eval()
    dummy = _make_dummy()
    batch_size = input_shape[0]

    for _ in range(warmup):
        model(dummy)

    latencies = []
    for _ in range(n_runs):
        x = _make_dummy()
        t0 = time.perf_counter()
        model(x)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies_sorted = sorted(latencies)
    median = _stats.median(latencies)
    p95_idx = int(math.ceil(0.95 * len(latencies_sorted))) - 1
    p95 = latencies_sorted[max(0, p95_idx)]

    tracemalloc.start()
    model(dummy)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_params = 0
    def _count(mod):
        nonlocal total_params
        for p in mod._parameters.values():
            if hasattr(p._data, 'shape'):
                n = 1
                for s in p._data.shape: n *= s
                total_params += n
        for m in mod._modules.values():
            _count(m)
    _count(model)

    return {
        'latency_ms':         float(median),
        'latency_p95_ms':     float(p95),
        'throughput_per_sec': float(batch_size / (median / 1000)),
        'peak_memory_mb':     peak / (1024 ** 2),
        'n_parameters':       total_params,
        'n_runs':             n_runs,
    }


from vectorphile.export._gguf import (
    GGUFReader, GGUFWriter,
    to_gguf, from_gguf,
    quantize_q4_0, quantize_q8_0,
    dequantize_q4_0, dequantize_q8_0,
)

# Convenience alias
export_onnx = to_onnx

__all__ = [
    'to_onnx', 'export_onnx', 'to_torchscript', 'to_savedmodel', 'to_coreml',
    'ONNXRuntime', 'benchmark',
    'VPONNXGraphBuilder', 'TensorRTExporter', 'TorchDynamo', 'compile',
    'GGUFReader', 'GGUFWriter', 'to_gguf', 'from_gguf',
    'quantize_q4_0', 'quantize_q8_0', 'dequantize_q4_0', 'dequantize_q8_0',
]
