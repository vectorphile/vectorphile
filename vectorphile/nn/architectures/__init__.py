"""vectorphile.nn.architectures — production-quality DL architecture zoo.

All implementations are pure-numpy/vectorphile with no external DL dependencies.

Transformers & LLMs:
    from vectorphile.nn.architectures.transformer import GPT2, BERT, T5, LLaMA, Mistral
    from vectorphile.nn.architectures.attention import (
        RotaryEmbedding, ALiBiPositionBias,
        GroupedQueryAttention, MultiQueryAttention, SlidingWindowAttention,
        causal_mask, sliding_window_mask, scaled_dot_product_attention,
    )

CNNs:
    from vectorphile.nn.architectures.cnn import (
        LeNet5, VGG, ResNet, DenseNet, MobileNetV2, EfficientNet,
    )

Vision Transformers:
    from vectorphile.nn.architectures.vision_transformer import ViT, DeiT

State Space Models:
    from vectorphile.nn.architectures.ssm import S4, S4Layer, Mamba, MambaBlock

Generative:
    from vectorphile.nn.architectures.generative import VAE, DCGAN, DDPM

Segmentation:
    from vectorphile.nn.architectures.unet import UNet, ResUNet, AttentionUNet

Graph:
    from vectorphile.nn.architectures.graph import GCN, GAT, GraphSAGE, GIN
"""

from vectorphile.nn.architectures.attention import (
    RotaryEmbedding,
    ALiBiPositionBias,
    GroupedQueryAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
    causal_mask,
    sliding_window_mask,
    scaled_dot_product_attention,
    apply_rotary_pos_emb,
    rotate_half,
    CrossAttention,
    LinearAttention,
)

from vectorphile.nn.architectures.transformer import (
    GPT2,
    BERT,
    BERTForMLM,
    T5,
    LLaMA,
    Mistral,
    SwiGLU,
    GeGLU,
    GELUMLP,
    RMSNorm,
    MTPHead,
    MTPLoss,
    SpeculativeDecoder,
    Qwen2, Qwen2Block,
    Phi3, Phi3Block,
    DeepSeekV2, DeepSeekV2Block,
    MultiHeadLatentAttention,
    GemmaBlock, Gemma,
    MixtralExpert, MixtralSparseMoE, MixtralBlock, Mixtral,
    FalconMQA, FalconBlock, Falcon,
    Phi2Block, Phi2,
    OPTBlock, OPT,
    # Session 10: ViT from transformer.py (native VP impl)
    PatchEmbed, ViTBlock as ViTBlockTf,
    _simple_mha,
    ViT as ViTTf,
    # Mamba (SSM) from transformer.py
    MambaBlock as MambaBlockTf,
    Mamba as MambaTf,
    # RetNet from transformer.py
    RetentionHead, RetNetBlock, RetNet,
    # RWKV from transformer.py
    RWKVTimeMix, RWKVChannelMix, RWKVBlock, RWKV,
)

from vectorphile.nn.architectures.cnn import (
    LeNet5,
    VGG,
    ResNet,
    BasicBlock,
    Bottleneck,
    DenseNet,
    DenseBlock,
    MobileNetV2,
    InvertedResidual,
    EfficientNet,
    MBConv,
    SEBlock,
    ConvBNReLU,
    ConvNeXt,
    ConvNeXtBlock,
)

from vectorphile.nn.architectures.vision_transformer import (
    ViT,
    DeiT,
    PatchEmbedding,
    ViTBlock,
    ViTAttention,
    ViTReg,
    DeiT3,
)

from vectorphile.nn.architectures.ssm import (
    S4,
    S4Layer,
    Mamba,
    MambaBlock,
    hippo_legt_matrix,
    discretize_zoh,
)

from vectorphile.nn.architectures.generative import (
    VAE,
    DCGAN,
    DCGANGenerator,
    DCGANDiscriminator,
    DDPM,
    SinusoidalTimeEmbedding,
)

from vectorphile.nn.architectures.unet import (
    UNet,
    ResUNet,
    AttentionUNet,
    AttentionGate,
    DoubleConv,
    ResConvBlock,
)

from vectorphile.nn.architectures.clip import (
    CLIP, CLIPTextEncoder, CLIPVisionEncoder,
    CLIPEncoderLayer, CLIPAttention, CLIPLayerNorm, CLIPLinear,
)

from vectorphile.nn.architectures.whisper import (
    Whisper, WhisperEncoder, WhisperDecoder,
    WhisperEncoderLayer, WhisperDecoderLayer,
    WhisperMHA, WhisperLayerNorm, WhisperLinear, WhisperConv1d,
)

from vectorphile.nn.architectures.graph import (
    GCN,
    GCNLayer,
    GAT,
    GATLayer,
    GraphSAGE,
    SAGELayer,
    GIN,
    GINLayer,
    _normalize_adj,
    _degree_norm_adj,
)

__all__ = [
    # Attention
    'RotaryEmbedding', 'ALiBiPositionBias', 'GroupedQueryAttention',
    'MultiQueryAttention', 'SlidingWindowAttention',
    'causal_mask', 'sliding_window_mask', 'scaled_dot_product_attention',
    'apply_rotary_pos_emb', 'rotate_half',
    'CrossAttention', 'LinearAttention',
    # Transformer LLMs
    'GPT2', 'BERT', 'BERTForMLM', 'T5', 'LLaMA', 'Mistral',
    'SwiGLU', 'GeGLU', 'GELUMLP', 'RMSNorm', 'MTPHead', 'MTPLoss', 'SpeculativeDecoder',
    'Qwen2', 'Qwen2Block', 'Phi3', 'Phi3Block',
    'DeepSeekV2', 'DeepSeekV2Block', 'MultiHeadLatentAttention',
    'GemmaBlock', 'Gemma',
    'MixtralExpert', 'MixtralSparseMoE', 'MixtralBlock', 'Mixtral',
    'FalconMQA', 'FalconBlock', 'Falcon',
    'Phi2Block', 'Phi2',
    'OPTBlock', 'OPT',
    # Session 10 new architectures
    'PatchEmbed', 'ViTBlockTf', 'ViTTf',
    'MambaBlockTf', 'MambaTf',
    'RetentionHead', 'RetNetBlock', 'RetNet',
    'RWKVTimeMix', 'RWKVChannelMix', 'RWKVBlock', 'RWKV',
    # CNNs
    'LeNet5', 'VGG', 'ResNet', 'BasicBlock', 'Bottleneck',
    'DenseNet', 'DenseBlock', 'MobileNetV2', 'InvertedResidual',
    'EfficientNet', 'MBConv', 'SEBlock', 'ConvBNReLU',
    'ConvNeXt', 'ConvNeXtBlock',
    # ViT
    'ViT', 'DeiT', 'PatchEmbedding', 'ViTBlock', 'ViTAttention', 'ViTReg', 'DeiT3',
    # SSM
    'S4', 'S4Layer', 'Mamba', 'MambaBlock', 'hippo_legt_matrix', 'discretize_zoh',
    # Generative
    'VAE', 'DCGAN', 'DCGANGenerator', 'DCGANDiscriminator', 'DDPM',
    'SinusoidalTimeEmbedding',
    # U-Net
    'UNet', 'ResUNet', 'AttentionUNet', 'AttentionGate', 'DoubleConv', 'ResConvBlock',
    # CLIP (multimodal)
    'CLIP', 'CLIPTextEncoder', 'CLIPVisionEncoder',
    'CLIPEncoderLayer', 'CLIPAttention', 'CLIPLayerNorm', 'CLIPLinear',
    # Whisper (ASR)
    'Whisper', 'WhisperEncoder', 'WhisperDecoder',
    'WhisperEncoderLayer', 'WhisperDecoderLayer',
    'WhisperMHA', 'WhisperLayerNorm', 'WhisperLinear', 'WhisperConv1d',
    # GNN
    'GCN', 'GCNLayer', 'GAT', 'GATLayer', 'GraphSAGE', 'SAGELayer',
    'GIN', 'GINLayer',
]
