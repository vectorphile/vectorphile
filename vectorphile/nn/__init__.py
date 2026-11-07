from vectorphile.nn.module import Module
from vectorphile.nn.layers import Linear, Conv2d, BatchNorm1d, BatchNorm2d, LayerNorm, Dropout, Embedding
from vectorphile.nn.activations import (
    ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax,
    SELU, SiLU, Mish, Hardswish,
)
from vectorphile.nn.losses import (
    MSELoss, MAELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,
    NLLLoss, HuberLoss, KLDivLoss, CosineSimilarityLoss,
)
from vectorphile.nn.sequential import Sequential
from vectorphile.nn.pooling import (
    MaxPool1d, MaxPool2d, MaxPool3d,
    AvgPool1d, AvgPool2d, AvgPool3d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool2d,
    GlobalAvgPool2d,
)
from vectorphile.nn.recurrent import (
    RNNCell, LSTMCell, GRUCell,
    RNN, LSTM, GRU,
)
from vectorphile.nn.attention import (
    MultiheadAttention,
    TransformerEncoderLayer, TransformerEncoder,
    TransformerDecoderLayer, TransformerDecoder,
    Transformer, PositionalEncoding,
)
from vectorphile.nn.container import (
    ModuleList, ModuleDict, ParameterList, ParameterDict,
)
from vectorphile.nn.norm import (
    GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
    RMSNorm, LocalResponseNorm,
)
from vectorphile.nn import init
from vectorphile.nn import functional as F
from vectorphile.nn.utils import (
    clip_grad_norm_, clip_grad_value_,
    parameters_to_vector, vector_to_parameters,
    model_summary, weight_norm, spectral_norm,
    get_total_params, freeze_params, unfreeze_params,
)

__all__ = [
    # Core
    'Module', 'Sequential', 'F', 'init',
    # Linear / Conv
    'Linear', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'Dropout', 'Embedding',
    # Activations
    'ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
    'SELU', 'SiLU', 'Mish', 'Hardswish',
    # Losses
    'MSELoss', 'MAELoss', 'CrossEntropyLoss', 'BCELoss', 'BCEWithLogitsLoss',
    'NLLLoss', 'HuberLoss', 'KLDivLoss', 'CosineSimilarityLoss',
    # Pooling
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveMaxPool2d',
    'GlobalAvgPool2d',
    # Recurrent
    'RNNCell', 'LSTMCell', 'GRUCell', 'RNN', 'LSTM', 'GRU',
    # Attention / Transformer
    'MultiheadAttention',
    'TransformerEncoderLayer', 'TransformerEncoder',
    'TransformerDecoderLayer', 'TransformerDecoder',
    'Transformer', 'PositionalEncoding',
    # Container
    'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict',
    # Norm
    'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'RMSNorm', 'LocalResponseNorm',
    # Utils
    'clip_grad_norm_', 'clip_grad_value_',
    'parameters_to_vector', 'vector_to_parameters',
    'model_summary', 'weight_norm', 'spectral_norm',
    'get_total_params', 'freeze_params', 'unfreeze_params',
]
