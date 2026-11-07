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

__version__ = '0.2.0'
__author__ = 'vectorphile'
__license__ = 'Apache-2.0'

# ── Core tensor operations ───────────────────────────────────────────────────
from vectorphile.core.tensor import (
    Tensor,
    tensor, zeros, ones, eye, arange, linspace, full, empty,
    rand, randn, randint, from_numpy,
    zeros_like, ones_like, empty_like,
    cat, stack, split, chunk, squeeze, unsqueeze, flatten, transpose, permute,
    matmul, dot, where,
    exp, log, sqrt, clamp,
    softmax, log_softmax, sigmoid, relu, tanh, einsum,
)
# shadow builtins only in the vp namespace
from vectorphile.core.tensor import sum as sum
from vectorphile.core.tensor import mean as mean
from vectorphile.core.tensor import max as max
from vectorphile.core.tensor import min as min
from vectorphile.core.tensor import abs as abs

# ── Autograd ─────────────────────────────────────────────────────────────────
from vectorphile import autograd
from vectorphile.autograd import no_grad

# ── Neural network ───────────────────────────────────────────────────────────
from vectorphile import nn

# ── Optimizers ───────────────────────────────────────────────────────────────
from vectorphile import optim

# ── Linear algebra ───────────────────────────────────────────────────────────
from vectorphile import linalg

# ── Statistics ───────────────────────────────────────────────────────────────
from vectorphile import stats

# ── Random ───────────────────────────────────────────────────────────────────
from vectorphile import random

# ── FFT ──────────────────────────────────────────────────────────────────────
from vectorphile import fft

# ── Signal ───────────────────────────────────────────────────────────────────
from vectorphile import signal

# ── Sparse ───────────────────────────────────────────────────────────────────
from vectorphile import sparse

# ── Optimization ─────────────────────────────────────────────────────────────
from vectorphile import optimize

# ── Automatic Mixed Precision ────────────────────────────────────────────────
from vectorphile import amp

# ── Data (DataFrame/Series) ──────────────────────────────────────────────────
from vectorphile import data

# ── Machine learning ─────────────────────────────────────────────────────────
from vectorphile import ml

# ── New modules (v0.2.0) ──────────────────────────────────────────────────────
from vectorphile import integrate
from vectorphile import interpolate
from vectorphile import spatial
from vectorphile import ndimage
from vectorphile import transforms
from vectorphile import polynomial
from vectorphile import io
from vectorphile import ma
from vectorphile import viz
from vectorphile import hub
from vectorphile import quantization

# ── Utils (DataLoader, Dataset) ──────────────────────────────────────────────
from vectorphile import utils
from vectorphile.utils.data import Dataset, TensorDataset, DataLoader

# ── Convenience re-exports ───────────────────────────────────────────────────
from vectorphile.data import DataFrame, Series, read_csv, read_json
from vectorphile.ml import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    KMeans, PCA, StandardScaler, MinMaxScaler,
    train_test_split, accuracy_score,
)

# Device helpers (Phase 1: always CPU)
def device(dev: str) -> str:
    return dev

def is_available() -> bool:
    """Returns True if CUDA is available (always False in Phase 1)."""
    return False

cuda = type('cuda', (), {
    'is_available': staticmethod(lambda: False),
    'device_count': staticmethod(lambda: 0),
    'get_device_name': staticmethod(lambda i=0: 'CPU'),
})()

# Type aliases
float32 = 'float32'
float64 = 'float64'
float16 = 'float16'
int32 = 'int32'
int64 = 'int64'
int8 = 'int8'
uint8 = 'uint8'
bool_ = 'bool'
half = 'float16'
double = 'float64'
long = 'int64'

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
    'Dataset', 'TensorDataset', 'DataLoader',
    'DataFrame', 'Series', 'read_csv', 'read_json',
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression',
    'DecisionTreeClassifier', 'DecisionTreeRegressor',
    'RandomForestClassifier', 'RandomForestRegressor',
    'KMeans', 'PCA', 'StandardScaler', 'MinMaxScaler',
    'train_test_split', 'accuracy_score',
    'cuda', 'device', 'is_available',
    'float32', 'float64', 'float16', 'int32', 'int64', 'int8', 'uint8', 'bool_', 'half', 'double', 'long',
]
