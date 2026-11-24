from vectorphile.nn.module import Module
from vectorphile.nn.layers import (Linear, FusedLinear, Conv1d, Conv2d, ConvTranspose2d,
                                    BatchNorm1d, BatchNorm2d, LayerNorm, Dropout, Embedding, Flatten,
                                    InstanceNorm1d, InstanceNorm2d, GroupNorm,
                                    Conv3d, MaxPool3d as MaxPool3dLayer, AvgPool3d as AvgPool3dLayer,
                                    AdaptiveAvgPool3d,
                                    Upsample, PixelShuffle, PixelUnshuffle, ChannelShuffle,
                                    ZeroPad2d, ReflectionPad2d, ConstantPad2d,
                                    ConstantPad1d, ConstantPad3d,
                                    ReflectionPad1d, ReplicationPad1d, ReplicationPad2d,
                                    UpsamplingNearest2d, UpsamplingBilinear2d,
                                    ConvTranspose3d,
                                    AdaptiveMaxPool1d, AdaptiveMaxPool3d)
from vectorphile.core.tensor import Tensor as Parameter
from vectorphile.nn.activations import (
    ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax,
    SELU, SiLU, Mish, Hardswish, ReLU6, Identity,
)
from vectorphile.nn.losses import (
    MSELoss, MAELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,
    NLLLoss, HuberLoss, KLDivLoss, CosineSimilarityLoss,
    TripletMarginLoss, MultiLabelMarginLoss, PoissonNLLLoss, FocalLoss, CTCLoss,
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
    MultiHeadAttention, MultiheadAttention,
    flash_attn_single_head,
    TransformerEncoderLayer, TransformerEncoder,
    TransformerDecoderLayer, TransformerDecoder,
    Transformer, PositionalEncoding,
    BlockSparseAttention, RingAttention, FlashDecodeAttention,
)
from vectorphile.nn.flash_attention import (
    FlashAttention, RotaryEmbedding, KVCache, MultiheadFlashAttention,
)
from vectorphile.nn.kv_cache import (
    KVCache as SimpleKVCache,
    PagedKVCache,
    KVCacheQuantized,
    ContinuousBatcher,
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
from vectorphile.nn import architectures
from vectorphile.nn import lr_schedulers
from vectorphile.nn import hooks
from vectorphile.nn import einops
from vectorphile.nn.keras_model import (
    Model, History,
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LambdaCallback, CSVLogger, TensorBoard,
    Sequential as KerasSequential,
)
from vectorphile.nn.extra_layers import (
    PReLU, GLU, CELU, LogSigmoid, Tanhshrink, Softshrink, Hardshrink, Swish, Snake,
    Hardtanh, Softplus, Threshold,
    Softsign, Hardsigmoid, Softmin, Softmax2d,
    ConvTranspose1d, DepthwiseSepConv2d, DepthwiseSepConv1d,
    PixelShuffle, ChannelShuffle, Unfold, Fold,
    MultiQueryAttention, CrossAttention, LinearAttention,
    FilterResponseNorm, PowerNorm,
    StochasticDepth, DropBlock2d,
    RelativePositionEncoding, ALiBiPositionBias,
    Lambda, Residual, PreNorm, Scale,
    MixerMLP, GatedMLP, PatchEmbed, VisionTransformerBlock,
    GCNConv, GATConv, SAGEConv, EdgeConv,
    S4Block, Mamba,
    CausalConv1d, WaveNetBlock,
)
from vectorphile.nn.lr_schedulers import (
    LRScheduler, StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    OneCycleLR, WarmupCosineDecay, LinearWarmup, PolynomialLR,
    ReduceLROnPlateauScheduler, get_scheduler,
)
from vectorphile.nn.einops import rearrange, reduce, repeat, pack, unpack, EinMix
from vectorphile.nn import pruning
from vectorphile.nn.moe import MixtureOfExperts, SparseMoE, SwitchFeedForward
# PEFT components available via nn namespace
try:
    from vectorphile.peft import LoRALinear, QLoRALinear, DoRALinear, IA3Linear, AdapterLayer
except ImportError:
    pass
from vectorphile.nn.pruning import (
    prune_unstructured, prune_structured, global_unstructured_prune,
    remove_pruning, apply_mask, sparsity,
    GradualMagnitudePruner, LotteryTicketPruner,
)
from vectorphile.nn.hooks import (
    RemovableHook, HooksMixin,
    register_forward_hook, register_backward_hook,
    FeatureExtractor, GradientRecorder,
)
from vectorphile.nn.utils import (
    clip_grad_norm_, clip_grad_value_,
    parameters_to_vector, vector_to_parameters,
    model_summary, weight_norm, spectral_norm,
    get_total_params, freeze_params, unfreeze_params,
    CheckpointedModule, checkpoint, checkpoint_sequential,
    model_parameter_breakdown,
    PackedSequence, pad_sequence, pack_padded_sequence, pad_packed_sequence,
)

# ── Missing nn classes ────────────────────────────────────────────────────────

class BatchNorm3d(Module):
    """3D batch normalization."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            from vectorphile.core.tensor import ones, zeros
            self.weight = ones(num_features)
            self.bias = zeros(num_features)
            self.weight.requires_grad = True
            self.bias.requires_grad = True

    def forward(self, x):
        from vectorphile.nn.functional import batch_norm
        sh = x.shape
        # Reshape to (B, C, L) to reuse batch_norm
        B, C = sh[0], sh[1]
        x2 = x.reshape((B, C, -1))
        rm = getattr(self, 'running_mean', None)
        rv = getattr(self, 'running_var', None)
        w = self.weight if self.affine else None
        b = self.bias if self.affine else None
        out = batch_norm(x2, rm, rv, weight=w, bias=b, training=self.training, eps=self.eps)
        return out.reshape(sh)


class Dropout1d(Module):
    """1D dropout (drops entire channels)."""
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        from vectorphile.nn.functional import dropout
        return dropout(x, p=self.p, training=self.training, inplace=self.inplace)


class Dropout2d(Module):
    """2D dropout (drops entire channels)."""
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        from vectorphile.nn.functional import dropout2d
        return dropout2d(x, p=self.p, training=self.training, inplace=self.inplace)


class Dropout3d(Module):
    """3D dropout (drops entire channels)."""
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        # Reuse dropout2d logic (same channel-drop semantics)
        from vectorphile.nn.functional import dropout2d
        return dropout2d(x, p=self.p, training=self.training, inplace=self.inplace)


class AlphaDropout(Module):
    """Alpha Dropout (maintains mean and variance)."""
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        from vectorphile.nn.functional import alpha_dropout
        return alpha_dropout(x, p=self.p, training=self.training, inplace=self.inplace)


class EmbeddingBag(Module):
    """Compute sums or means of embeddings."""
    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = 'mean',
                 padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        import math as _m
        from vectorphile.core.tensor import randn
        scale = _m.sqrt(1.0 / num_embeddings)
        self.weight = randn(num_embeddings, embedding_dim) * scale
        self.weight.requires_grad = True

    def forward(self, x, offsets=None):
        from vectorphile.nn.functional import embedding
        emb = embedding(x, self.weight)
        # Bag reduction: mean or sum along token dim
        if self.mode == 'mean':
            return emb.mean(dim=-2) if len(emb.shape) > 2 else emb
        return emb.sum(dim=-2) if len(emb.shape) > 2 else emb


class Unflatten(Module):
    """Unflattens a tensor dimension."""
    def __init__(self, dim: int, unflattened_size):
        super().__init__()
        self.dim = dim
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, x):
        sh = list(x.shape)
        new_sh = sh[:self.dim] + list(self.unflattened_size) + sh[self.dim + 1:]
        return x.reshape(tuple(new_sh))


class L1Loss(Module):
    """Mean absolute error loss."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, target):
        from vectorphile.nn.functional import l1_loss
        return l1_loss(x, target, reduction=self.reduction)


__all__ = [
    # Core
    'Module', 'Model', 'Sequential', 'KerasSequential', 'F', 'init',
    # Keras API
    'History', 'EarlyStopping', 'ReduceLROnPlateau', 'ModelCheckpoint',
    'LambdaCallback', 'CSVLogger', 'TensorBoard',
    # Extra layers
    'PReLU', 'GLU', 'CELU', 'LogSigmoid', 'Tanhshrink', 'Softshrink', 'Hardshrink', 'Swish', 'Snake',
    'Softsign', 'Hardsigmoid', 'Softmin', 'Softmax2d',
    'Hardtanh', 'Softplus', 'Threshold',
    'ConvTranspose1d', 'DepthwiseSepConv2d', 'DepthwiseSepConv1d',
    'PixelShuffle', 'ChannelShuffle', 'Unfold', 'Fold',
    'MultiQueryAttention', 'CrossAttention', 'LinearAttention',
    'FilterResponseNorm', 'PowerNorm',
    'StochasticDepth', 'DropBlock2d',
    'RelativePositionEncoding', 'ALiBiPositionBias',
    'Lambda', 'Residual', 'PreNorm', 'Scale',
    'MixerMLP', 'GatedMLP', 'PatchEmbed', 'VisionTransformerBlock',
    'GCNConv', 'GATConv', 'SAGEConv', 'EdgeConv',
    'S4Block', 'Mamba',
    'CausalConv1d', 'WaveNetBlock',
    # Linear / Conv
    'Linear', 'FusedLinear', 'MultiHeadAttention', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'Dropout', 'Embedding',
    # Activations
    'ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
    'SELU', 'SiLU', 'Mish', 'Hardswish',
    # Losses
    'MSELoss', 'MAELoss', 'L1Loss', 'SmoothL1Loss', 'CrossEntropyLoss', 'BCELoss', 'BCEWithLogitsLoss',
    'NLLLoss', 'HuberLoss', 'KLDivLoss', 'CosineSimilarityLoss',
    'TripletMarginLoss', 'MultiLabelMarginLoss', 'PoissonNLLLoss', 'FocalLoss', 'CTCLoss',
    # Pooling
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveMaxPool2d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool3d',
    'AdaptiveAvgPool3d',
    'GlobalAvgPool2d',
    # Padding
    'ZeroPad2d', 'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ReflectionPad1d', 'ReflectionPad2d',
    'ReplicationPad1d', 'ReplicationPad2d',
    # Conv/Upsample
    'ConvTranspose3d', 'UpsamplingNearest2d', 'UpsamplingBilinear2d',
    # Recurrent
    'RNNCell', 'LSTMCell', 'GRUCell', 'RNN', 'LSTM', 'GRU',
    # Attention / Transformer
    'MultiheadAttention',
    'BlockSparseAttention', 'RingAttention', 'FlashDecodeAttention',
    'TransformerEncoderLayer', 'TransformerEncoder',
    'TransformerDecoderLayer', 'TransformerDecoder',
    'Transformer', 'PositionalEncoding',
    # Container
    'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict',
    # Norm
    'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'RMSNorm', 'LocalResponseNorm',
    # Utils
    'CheckpointedModule', 'checkpoint', 'checkpoint_sequential', 'model_parameter_breakdown',
    'clip_grad_norm_', 'clip_grad_value_',
    'parameters_to_vector', 'vector_to_parameters',
    'model_summary', 'weight_norm', 'spectral_norm',
    'get_total_params', 'freeze_params', 'unfreeze_params',
    'PackedSequence', 'pad_sequence', 'pack_padded_sequence', 'pad_packed_sequence',
    # KV Cache & Serving
    'SimpleKVCache', 'PagedKVCache', 'KVCacheQuantized', 'ContinuousBatcher',
    # Architecture zoo
    'architectures',
    # LR schedulers
    'lr_schedulers', 'hooks', 'einops',
    'LRScheduler', 'StepLR', 'MultiStepLR', 'ExponentialLR',
    'CosineAnnealingLR', 'CosineAnnealingWarmRestarts',
    'OneCycleLR', 'WarmupCosineDecay', 'LinearWarmup', 'PolynomialLR',
    'ReduceLROnPlateauScheduler', 'get_scheduler',
    # Einops
    'rearrange', 'reduce', 'repeat', 'pack', 'unpack', 'EinMix',
    # Hooks
    'RemovableHook', 'HooksMixin',
    'register_forward_hook', 'register_backward_hook',
    'FeatureExtractor', 'GradientRecorder',
]
