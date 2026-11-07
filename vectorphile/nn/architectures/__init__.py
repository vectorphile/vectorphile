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
)

from vectorphile.nn.architectures.vision_transformer import (
    ViT,
    DeiT,
    PatchEmbedding,
    ViTBlock,
    ViTAttention,
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
    # Transformer LLMs
    'GPT2', 'BERT', 'BERTForMLM', 'T5', 'LLaMA', 'Mistral',
    'SwiGLU', 'GeGLU', 'GELUMLP', 'RMSNorm',
    # CNNs
    'LeNet5', 'VGG', 'ResNet', 'BasicBlock', 'Bottleneck',
    'DenseNet', 'DenseBlock', 'MobileNetV2', 'InvertedResidual',
    'EfficientNet', 'MBConv', 'SEBlock', 'ConvBNReLU',
    # ViT
    'ViT', 'DeiT', 'PatchEmbedding', 'ViTBlock', 'ViTAttention',
    # SSM
    'S4', 'S4Layer', 'Mamba', 'MambaBlock', 'hippo_legt_matrix', 'discretize_zoh',
    # Generative
    'VAE', 'DCGAN', 'DCGANGenerator', 'DCGANDiscriminator', 'DDPM',
    'SinusoidalTimeEmbedding',
    # U-Net
    'UNet', 'ResUNet', 'AttentionUNet', 'AttentionGate', 'DoubleConv', 'ResConvBlock',
    # GNN
    'GCN', 'GCNLayer', 'GAT', 'GATLayer', 'GraphSAGE', 'SAGELayer',
    'GIN', 'GINLayer',
]
