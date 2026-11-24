"""
vectorphile — Unified ML/DL Python Library
==========================================

The one library that replaces PyTorch + scikit-learn + NumPy + Pandas + SciPy.

Quick start:
    import vectorphile as vp

    # Tensors (PyTorch-like)
    x = vp.tensor([1, 2, 3], dtype='float32')
    y = vp.randn(3, 4)
    z = x @ y

    # Neural networks
    model = vp.nn.Sequential(
        vp.nn.Linear(4, 128),
        vp.nn.ReLU(),
        vp.nn.Linear(128, 10),
    )

    # ML algorithms
    clf = vp.ml.RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # DataFrames
    df = vp.data.read_csv('data.csv')
    df.groupby('category').mean()
"""

__version__ = '0.5.0'
__author__ = 'vectorphile'
__license__ = 'Apache-2.0'

# ── Core tensor operations ───────────────────────────────────────────────────
from vectorphile.core.tensor import (
    Tensor, SparseTensor,
    tensor, zeros, ones, eye, arange, linspace, full, empty,
    rand, randn, randint, from_numpy,
    zeros_like, ones_like, empty_like, full_like, rand_like, randn_like,
    cat, stack, split, chunk, squeeze, unsqueeze, flatten, transpose, permute,
    matmul, mm, dot, where,
    exp, log, sqrt, clamp, norm,
    softmax, log_softmax, sigmoid, relu, tanh, gelu, silu, swish, einsum,
    tril, triu, bmm, topk, cumsum, multinomial, isnan, isinf, isfinite, sort,
    roll, flip, rot90, diag, trace, diff, meshgrid, cdist,
    vmap, argsort, searchsorted, bincount,
    lerp, addcmul, addcdiv, ger, outer, cross, kron,
    unique, broadcast_tensors, atleast_1d, atleast_2d, atleast_3d,
    nonzero, count_nonzero, logsumexp, fmod, remainder,
    addmm, mv, vdot, sign, ceil, floor, std, var, prod, cumprod,
    eq, ne, lt, le, gt, ge, maximum, minimum,
    repeat_interleave, tile,
    masked_select, masked_scatter, index_add,
    tensordot, inner, gradient, histc, unique_consecutive,
    kthvalue, mode, scatter, gather, index_put,
    cartesian_prod, combinations,
    cosine_similarity, pairwise_distance,
    nan_to_num, clip, atan2, hypot, logaddexp, logaddexp2,
    isreal,
    sin, cos, tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh,
    rsqrt, log2, log10,
    reshape, argmax, argmin, unbind,
    fmax, fmin, bernoulli,
    fliplr, flipud, vstack, hstack, dstack, column_stack, row_stack,
    addr, index_copy,
    block_diag, diag_embed, scatter_add,
    index_select, bucketize,
)
from vectorphile.signal import stft
# shadow builtins only in the vp namespace
from vectorphile.core.tensor import sum as sum
from vectorphile.core.tensor import mean as mean
from vectorphile.core.tensor import max as max
from vectorphile.core.tensor import min as min
from vectorphile.core.tensor import abs as abs
from vectorphile.core.tensor import round as round
from vectorphile.core.tensor import pow as pow

# ── Autograd ─────────────────────────────────────────────────────────────────
from vectorphile import autograd
from vectorphile.autograd import no_grad, gradcheck

# ── Neural network ───────────────────────────────────────────────────────────
from vectorphile import nn
from vectorphile.nn import functional as F
functional = F  # vp.functional is an alias for vp.F / vp.nn.functional

# Install optimized layer forward passes (patches BatchNorm1d/2d, Conv, etc.)
try:
    from vectorphile.autograd.layers_complete import install as _install_layers
    _install_layers()
except Exception:
    pass

# ── Optimizers ───────────────────────────────────────────────────────────────
from vectorphile import optim

# linalg, stats, random, fft, signal, sparse, optimize, amp — moved to lazy registry below

# ── Eagerly-imported utility submodules (always needed) ──────────────────────
from vectorphile import utils
from vectorphile.utils.data import Dataset, TensorDataset, DataLoader

# ── File I/O (PyTorch-compatible save/load) ───────────────────────────────────
def save(obj, path: str) -> None:
    """Save any object to a file (pickle-based, like torch.save)."""
    from vectorphile.checkpoint import save as _ckpt_save
    _ckpt_save(obj, path)

def load(path: str):
    """Load an object saved with vp.save() (like torch.load)."""
    from vectorphile.checkpoint import load as _ckpt_load
    result = _ckpt_load(path)
    return result.get('obj', result) if isinstance(result, dict) and 'obj' in result else result

def manual_seed(seed: int) -> None:
    """Set the random seed for reproducible results (like torch.manual_seed)."""
    import random as _rnd
    _rnd.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    # Also seed vp.random module's internal RNG if already loaded
    try:
        import vectorphile.random as _vp_rand
        _vp_rand.seed(seed)
    except Exception:
        pass

def seed(s: int = None) -> int:
    """Get or set global random seed."""
    import random as _rnd
    if s is not None:
        manual_seed(s)
        return s
    return _rnd.randint(0, 2**32 - 1)

# ── Lazy submodule registry — imported on first attribute access ──────────────
# This keeps `import vectorphile` fast (< 1 s) while keeping full API coverage.
_LAZY_MODULES = {
    # core submodules (lazy — keep import time fast)
    'linalg': 'vectorphile.linalg',
    'stats': 'vectorphile.stats',
    'random': 'vectorphile.random',
    'fft': 'vectorphile.fft',
    'signal': 'vectorphile.signal',
    'sparse': 'vectorphile.sparse',
    'optimize': 'vectorphile.optimize',
    'amp': 'vectorphile.amp',
    # data / ml
    'data': 'vectorphile.data',
    'ml': 'vectorphile.ml',
    # v0.2.0
    'integrate': 'vectorphile.integrate',
    'interpolate': 'vectorphile.interpolate',
    'spatial': 'vectorphile.spatial',
    'ndimage': 'vectorphile.ndimage',
    'transforms': 'vectorphile.transforms',
    'polynomial': 'vectorphile.polynomial',
    'io': 'vectorphile.io',
    'ma': 'vectorphile.ma',
    'viz': 'vectorphile.viz',
    'hub': 'vectorphile.hub',
    'quantization': 'vectorphile.quantization',
    # v0.3.0
    'backend': 'vectorphile.backend',
    'tokenizers': 'vectorphile.tokenizers',
    'distributed': 'vectorphile.distributed',
    'export': 'vectorphile.export',
    'rl': 'vectorphile.rl',
    'pretrained': 'vectorphile.pretrained',
    # v0.4.0
    'metrics': 'vectorphile.metrics',
    'callbacks': 'vectorphile.callbacks',
    'trainer': 'vectorphile.trainer',
    'text': 'vectorphile.text',
    'vision': 'vectorphile.vision',
    'datasets': 'vectorphile.data.datasets',
    'jax': 'vectorphile.jax',
    # v0.5.0
    'diffusion': 'vectorphile.diffusion',
    'generation': 'vectorphile.generation',
    'timeseries': 'vectorphile.timeseries',
    'interpret': 'vectorphile.interpret',
    'nlp': 'vectorphile.nlp',
    'audio': 'vectorphile.audio',
    'graph': 'vectorphile.graph',
    'symbolic': 'vectorphile.symbolic',
    'automl': 'vectorphile.automl',
    'checkpoint': 'vectorphile.checkpoint',
    'fx': 'vectorphile.fx',
    'profiler': 'vectorphile.profiler',
    'federated': 'vectorphile.federated',
    'vp_logging': 'vectorphile.logging',
    'keras': 'vectorphile.keras',
    'causal': 'vectorphile.causal',
    'image': 'vectorphile.image',
    'bench': 'vectorphile.bench',
    'cv': 'vectorphile.cv',
    'gpu': 'vectorphile.gpu',
    'metal': 'vectorphile.metal',
    'peft': 'vectorphile.peft',
    'privacy': 'vectorphile.privacy',
    'special': 'vectorphile.special',
    'serve': 'vectorphile.serve',
    'uncertainty': 'vectorphile.uncertainty',
    'meta': 'vectorphile.meta',
    'agents': 'vectorphile.agents',
    'retrieval': 'vectorphile.retrieval',
    'pipeline': 'vectorphile.pipeline',
    'preprocessing': 'vectorphile.preprocessing',
    'model_selection': 'vectorphile.model_selection',
    'cluster': 'vectorphile.cluster',
    'decomposition': 'vectorphile.decomposition',
    'feature_extraction': 'vectorphile.feature_extraction',
    'manifold': 'vectorphile.manifold',
    'svm': 'vectorphile.svm',
    'neighbors': 'vectorphile.neighbors',
    'naive_bayes': 'vectorphile.naive_bayes',
    'linear_model': 'vectorphile.linear_model',
    'ensemble': 'vectorphile.ensemble',
    'gaussian_process': 'vectorphile.gaussian_process',
    'distributions': 'vectorphile.distributions',
}

# Lazy submodule attribute names that forward to a sub-attribute
_LAZY_FROM = {
    # module_name → (lazy_module_key, attr_name)
    'ModelCheckpoint': ('checkpoint', 'ModelCheckpoint'),
    'EarlyStopping': ('checkpoint', 'EarlyStopping'),
    'compile': ('export', 'compile'),
    'TensorRTExporter': ('export', 'TensorRTExporter'),
    'TorchDynamo': ('export', 'TorchDynamo'),
    'jit': ('vectorphile.jit', 'jit'),
    'script': ('vectorphile.jit', 'script'),
    'trace': ('vectorphile.jit', 'trace'),
    'ScriptModule': ('vectorphile.jit', 'ScriptModule'),
    'profile': ('profiler', 'profile'),
    'OpTimer': ('profiler', 'OpTimer'),
    'MemoryTracker': ('profiler', 'MemoryTracker'),
    'SummaryWriter': ('vp_logging', 'SummaryWriter'),
    'MetricLogger': ('vp_logging', 'MetricLogger'),
    'benchmark': ('bench', 'benchmark'),
    'benchmark_model': ('bench', 'benchmark_model'),
    'Pipeline': ('pipeline', 'Pipeline'),
    'make_pipeline': ('pipeline', 'make_pipeline'),
    'GridSearchCV': ('pipeline', 'GridSearchCV'),
    'RandomizedSearchCV': ('pipeline', 'RandomizedSearchCV'),
    'cross_val_score': ('pipeline', 'cross_val_score'),
    'KFold': ('model_selection', 'KFold'),
    'StratifiedKFold': ('model_selection', 'StratifiedKFold'),
    'LeaveOneOut': ('model_selection', 'LeaveOneOut'),
    'ShuffleSplit': ('model_selection', 'ShuffleSplit'),
    'TimeSeriesSplit': ('model_selection', 'TimeSeriesSplit'),
    'DBSCAN': ('cluster', 'DBSCAN'),
    'GaussianMixture': ('cluster', 'GaussianMixture'),
    'AgglomerativeCluster': ('cluster', 'AgglomerativeCluster'),
    'MeanShift': ('cluster', 'MeanShift'),
    'SpectralClustering': ('cluster', 'SpectralClustering'),
    'TruncatedSVD': ('decomposition', 'TruncatedSVD'),
    'NMF': ('decomposition', 'NMF'),
    'FastICA': ('decomposition', 'FastICA'),
    'FactorAnalysis': ('decomposition', 'FactorAnalysis'),
    'TSNE': ('manifold', 'TSNE'),
    'MDS': ('manifold', 'MDS'),
    'Isomap': ('manifold', 'Isomap'),
    'LocallyLinearEmbedding': ('manifold', 'LocallyLinearEmbedding'),
    'SpectralEmbedding': ('manifold', 'SpectralEmbedding'),
    'SVC': ('svm', 'SVC'),
    'SVR': ('svm', 'SVR'),
    'LinearSVC': ('svm', 'LinearSVC'),
    'LinearSVR': ('svm', 'LinearSVR'),
    'OneClassSVM': ('svm', 'OneClassSVM'),
    'KNeighborsClassifier': ('neighbors', 'KNeighborsClassifier'),
    'KNeighborsRegressor': ('neighbors', 'KNeighborsRegressor'),
    'NearestNeighbors': ('neighbors', 'NearestNeighbors'),
    'RadiusNeighborsClassifier': ('neighbors', 'RadiusNeighborsClassifier'),
    'RadiusNeighborsRegressor': ('neighbors', 'RadiusNeighborsRegressor'),
    'NearestCentroid': ('neighbors', 'NearestCentroid'),
    'GaussianNB': ('naive_bayes', 'GaussianNB'),
    'MultinomialNB': ('naive_bayes', 'MultinomialNB'),
    'BernoulliNB': ('naive_bayes', 'BernoulliNB'),
    'ComplementNB': ('naive_bayes', 'ComplementNB'),
    'CategoricalNB': ('naive_bayes', 'CategoricalNB'),
    'BayesianRidge': ('linear_model', 'BayesianRidge'),
    'HuberRegressor': ('linear_model', 'HuberRegressor'),
    'Perceptron': ('linear_model', 'Perceptron'),
    'SGDClassifier': ('linear_model', 'SGDClassifier'),
    'SGDRegressor': ('linear_model', 'SGDRegressor'),
    'PassiveAggressiveClassifier': ('linear_model', 'PassiveAggressiveClassifier'),
    'PassiveAggressiveRegressor': ('linear_model', 'PassiveAggressiveRegressor'),
    'RANSACRegressor': ('linear_model', 'RANSACRegressor'),
    'VotingClassifier': ('ensemble', 'VotingClassifier'),
    'VotingRegressor': ('ensemble', 'VotingRegressor'),
    'BaggingClassifier': ('ensemble', 'BaggingClassifier'),
    'BaggingRegressor': ('ensemble', 'BaggingRegressor'),
    'AdaBoostClassifier': ('ensemble', 'AdaBoostClassifier'),
    'AdaBoostRegressor': ('ensemble', 'AdaBoostRegressor'),
    'GradientBoostingClassifier': ('ensemble', 'GradientBoostingClassifier'),
    'GradientBoostingRegressor': ('ensemble', 'GradientBoostingRegressor'),
    'StackingClassifier': ('ensemble', 'StackingClassifier'),
    'StackingRegressor': ('ensemble', 'StackingRegressor'),
    'ExtraTreesClassifier': ('ensemble', 'ExtraTreesClassifier'),
    'ExtraTreesRegressor': ('ensemble', 'ExtraTreesRegressor'),
    'GaussianProcessRegressor': ('gaussian_process', 'GaussianProcessRegressor'),
    'GaussianProcessClassifier': ('gaussian_process', 'GaussianProcessClassifier'),
    'CountVectorizer': ('feature_extraction', 'CountVectorizer'),
    'TfidfVectorizer': ('feature_extraction', 'TfidfVectorizer'),
    'TfidfTransformer': ('feature_extraction', 'TfidfTransformer'),
    'HashingVectorizer': ('feature_extraction', 'HashingVectorizer'),
    'DictVectorizer': ('feature_extraction', 'DictVectorizer'),
    'LabelEncoder': ('preprocessing', 'LabelEncoder'),
    'OrdinalEncoder': ('preprocessing', 'OrdinalEncoder'),
    'OneHotEncoder': ('preprocessing', 'OneHotEncoder'),
    'LabelBinarizer': ('preprocessing', 'LabelBinarizer'),
    'SimpleImputer': ('preprocessing', 'SimpleImputer'),
    'PolynomialFeatures': ('preprocessing', 'PolynomialFeatures'),
    'Binarizer': ('preprocessing', 'Binarizer'),
    'FunctionTransformer': ('preprocessing', 'FunctionTransformer'),
    'KBinsDiscretizer': ('preprocessing', 'KBinsDiscretizer'),
    'RobustScaler': ('preprocessing', 'RobustScaler'),
    'Normalizer': ('preprocessing', 'Normalizer'),
    'PowerTransformer': ('preprocessing', 'PowerTransformer'),
    # data convenience
    'DataFrame': ('data', 'DataFrame'),
    'Series': ('data', 'Series'),
    'read_csv': ('data', 'read_csv'),
    'read_json': ('data', 'read_json'),
    # ml convenience
    'LinearRegression': ('ml', 'LinearRegression'),
    'Ridge': ('ml', 'Ridge'),
    'Lasso': ('ml', 'Lasso'),
    'ElasticNet': ('ml', 'ElasticNet'),
    'LogisticRegression': ('ml', 'LogisticRegression'),
    'DecisionTreeClassifier': ('ml', 'DecisionTreeClassifier'),
    'DecisionTreeRegressor': ('ml', 'DecisionTreeRegressor'),
    'RandomForestClassifier': ('ml', 'RandomForestClassifier'),
    'RandomForestRegressor': ('ml', 'RandomForestRegressor'),
    'KMeans': ('ml', 'KMeans'),
    'PCA': ('ml', 'PCA'),
    'StandardScaler': ('ml', 'StandardScaler'),
    'MinMaxScaler': ('ml', 'MinMaxScaler'),
    'train_test_split': ('ml', 'train_test_split'),
    'accuracy_score': ('ml', 'accuracy_score'),
}


def __getattr__(name):
    import importlib as _importlib
    # Lazy submodule
    if name in _LAZY_MODULES:
        mod = _importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = mod
        return mod
    # Lazy attribute from a submodule
    if name in _LAZY_FROM:
        mod_key, attr = _LAZY_FROM[name]
        if mod_key.startswith('vectorphile.'):
            mod = _importlib.import_module(mod_key)
        else:
            mod = __getattr__(mod_key)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'vectorphile' has no attribute {name!r}")

def device(dev: str) -> str:
    return dev


def compile(model, backend: str = 'vp', fullgraph: bool = False,
            dynamic: bool = False, example_inputs=None, fuse: bool = True, **kwargs):
    """Compile a model for faster execution (torch.compile-compatible API).

    backend options:
      'vp'        — VP symbolic trace + fusion passes + VP CUDA dispatch (default)
      'torch'     — torch.compile with inductor backend (requires PyTorch >= 2.0)
      'inductor'  — alias for 'torch'
      'eager'     — no compilation (identity)

    The 'vp' backend applies:
      1. FX symbolic trace (captures the compute graph)
      2. Op fusion passes: Linear+Act, LayerNorm, RMSNorm, QKV fusion
      3. VP CUDA / Triton kernel dispatch for each fused op
    """
    if backend in ('torch', 'inductor', 'aot_eager'):
        try:
            import torch as _tc
            if hasattr(_tc, 'compile'):
                orig_forward = model.forward
                compiled = _tc.compile(orig_forward, fullgraph=fullgraph,
                                       dynamic=dynamic,
                                       backend='inductor' if backend != 'aot_eager' else 'aot_eager')
                model.forward = compiled
                return model
        except Exception:
            pass
        return model

    if backend == 'vp':
        try:
            from vectorphile.fx import compile as _fx_compile
            return _fx_compile(model, example_inputs=example_inputs, fuse=fuse)
        except Exception:
            pass

    return model


def no_grad():
    """Alias for autograd.no_grad() context manager."""
    return autograd.no_grad()


def inference_mode():
    """Alias for autograd.no_grad() (torch.inference_mode compat)."""
    return autograd.no_grad()


def autocast(device_type='cpu', dtype=None, enabled=True, cache_enabled=None):
    """Automatic mixed precision context manager."""
    from vectorphile.amp import autocast as _ac
    return _ac(device_type=device_type, dtype=dtype, enabled=enabled)


class _Backends:
    """Mock torch.backends compatibility object."""
    class _CUDNN:
        enabled = True
        benchmark = False
        deterministic = False
        allow_tf32 = True

    class _MPS:
        is_available = staticmethod(lambda: False)
        is_built = staticmethod(lambda: False)

    cudnn = _CUDNN()
    mps = _MPS()

backends = _Backends()

version = __version__


def set_grad_enabled(enabled: bool):
    """Enable or disable gradient computation globally."""
    from vectorphile.autograd.engine import no_grad as _ng
    _ng._enabled = bool(enabled)


def enable_grad():
    """Context manager to enable gradient computation (even inside no_grad)."""
    from vectorphile.autograd.engine import enable_grad as _eg
    return _eg()


class GradScaler:
    """Automatic Mixed Precision gradient scaler (torch.cuda.amp.GradScaler compat).

    Currently a no-op since we don't have CUDA. Maintains API compatibility.
    """

    def __init__(self, init_scale: float = 2.**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000,
                 enabled: bool = True):
        self.enabled = enabled
        self._scale = init_scale

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self._scale if self.enabled else 1.0

# ── CUDA device namespace (Phase 5 — _vp_cuda native extension) ─────────────
def _cuda_is_available() -> bool:
    try:
        from vectorphile._core import _vp_cuda as _cv
        return _cv.device_count() > 0
    except Exception:
        return False

def _cuda_device_count() -> int:
    try:
        from vectorphile._core import _vp_cuda as _cv
        return _cv.device_count()
    except Exception:
        return 0

def _cuda_get_device_name(i=0) -> str:
    try:
        from vectorphile._core import _vp_cuda as _cv
        return _cv.device_name(i)
    except Exception:
        return 'no CUDA'

def _cuda_synchronize() -> None:
    try:
        from vectorphile._core import _vp_cuda as _cv
        _cv.synchronize()
    except Exception:
        pass

# Use the real vectorphile.cuda module as the vp.cuda namespace so all
# memory management, stream, and kernel APIs are accessible via vp.cuda.*
try:
    import vectorphile.cuda as cuda
except Exception:
    cuda = type('cuda', (), {
        'is_available':   staticmethod(_cuda_is_available),
        'device_count':   staticmethod(_cuda_device_count),
        'get_device_name': staticmethod(_cuda_get_device_name),
        'device_name':    staticmethod(_cuda_get_device_name),
        'synchronize':    staticmethod(_cuda_synchronize),
    })()

def is_cuda_available() -> bool:
    return cuda.is_available()

def is_metal_available() -> bool:
    """True when Apple Metal/MPS backend is available (Apple Silicon)."""
    return backend.is_metal_available()

def is_arm_neon() -> bool:
    """True when running on an ARM processor with NEON SIMD support."""
    return backend.is_arm_neon()

# ── Mojo/MAX backend helpers (Phase 6) ──────────────────────────────────────
def _mojo_is_available() -> bool:
    try:
        from vectorphile._core.mojo_backend import is_available as _ia
        return _ia()
    except Exception:
        return False

mojo = type('mojo', (), {
    'is_available': staticmethod(_mojo_is_available),
})()

def is_mojo_available() -> bool:
    return mojo.is_available()


# ── Inference acceleration ────────────────────────────────────────────────────
def _lazy_speculative_decoder():
    from vectorphile.nn.architectures.transformer import SpeculativeDecoder
    return SpeculativeDecoder

def _lazy_paged_kv_cache():
    from vectorphile.nn.kv_cache import PagedKVCache
    return PagedKVCache

def _lazy_kv_cache_quantized():
    from vectorphile.nn.kv_cache import KVCacheQuantized
    return KVCacheQuantized

def _lazy_continuous_batcher():
    from vectorphile.nn.kv_cache import ContinuousBatcher
    return ContinuousBatcher

def _lazy_flash_decode():
    from vectorphile.cuda.triton_kernels import flash_decode
    return flash_decode

SpeculativeDecoder  = property(lambda self: _lazy_speculative_decoder())
PagedKVCache        = _lazy_paged_kv_cache()
KVCacheQuantized    = _lazy_kv_cache_quantized()
ContinuousBatcher   = _lazy_continuous_batcher()

# Type aliases
float32 = 'float32'
float64 = 'float64'
float16 = 'float16'
bfloat16 = 'bfloat16'
int32 = 'int32'
int64 = 'int64'
int8 = 'int8'
int16 = 'int16'
uint8 = 'uint8'
uint16 = 'uint16'
uint32 = 'uint32'
uint64 = 'uint64'
bool_ = 'bool'
# PyTorch uses `torch.bool`, but we can't shadow Python's bool
# so user should use vp.bool_ — but also expose as `vp.bool` via __getattr__ fallback
half = 'float16'
double = 'float64'
long = 'int64'
short = 'int16'
char = 'int8'

__all__ = [
    'Tensor', '__version__',
    'tensor', 'zeros', 'ones', 'eye', 'arange', 'linspace', 'full', 'empty',
    'rand', 'randn', 'randint', 'from_numpy',
    'zeros_like', 'ones_like', 'empty_like',
    'cat', 'stack', 'split', 'chunk', 'squeeze', 'unsqueeze', 'flatten',
    'transpose', 'permute', 'matmul', 'dot', 'where', 'sum', 'mean', 'max', 'min',
    'exp', 'log', 'sqrt', 'abs', 'clamp', 'softmax', 'log_softmax', 'sigmoid', 'relu', 'tanh', 'einsum',
    'autograd', 'no_grad',
    'nn', 'optim', 'linalg', 'stats', 'random', 'fft', 'signal', 'sparse', 'optimize', 'amp',
    'data', 'ml', 'utils',
    'integrate', 'interpolate', 'spatial', 'ndimage', 'transforms',
    'polynomial', 'io', 'ma', 'viz', 'hub', 'quantization',
    'backend', 'tokenizers', 'distributed', 'export', 'rl', 'pretrained', 'fx',
    'metrics', 'callbacks', 'trainer', 'text', 'vision', 'datasets',
    'Dataset', 'TensorDataset', 'DataLoader',
    'DataFrame', 'Series', 'read_csv', 'read_json',
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression',
    'DecisionTreeClassifier', 'DecisionTreeRegressor',
    'RandomForestClassifier', 'RandomForestRegressor',
    'KMeans', 'PCA', 'StandardScaler', 'MinMaxScaler',
    'train_test_split', 'accuracy_score',
    'cuda', 'device', 'is_available',
    'distributions',
    'unique', 'broadcast_tensors', 'atleast_1d', 'atleast_2d', 'atleast_3d',
    'SAM', 'Lookahead',
    'PagedKVCache', 'KVCacheQuantized', 'ContinuousBatcher',
    'float32', 'float64', 'float16', 'bfloat16', 'int32', 'int64', 'int8', 'uint8', 'bool_', 'half', 'double', 'long',
    'profiler', 'profile', 'OpTimer', 'MemoryTracker',
    'compile', 'inference_mode',
    'fx', 'jit', 'script', 'trace', 'ScriptModule',
]
