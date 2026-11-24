<div align="center">

# vectorphile

**The unified ML/DL/AI library for Python — built entirely from scratch.**

[![version](https://img.shields.io/badge/version-0.5.0-blue)](https://github.com/vectorphile/vectorphile/releases)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-156%20passing-brightgreen)](#testing)

*One import. The entire ML stack — from raw tensor arithmetic to Transformers, from DataFrames to diffusion models — implemented from first principles with zero third-party dependencies under the hood.*

```python
import vectorphile as vp
```

</div>

---

## What is vectorphile?

vectorphile is a **single, coherent library** that replaces the entire modern ML ecosystem:

| Instead of... | Use... |
|---|---|
| `torch` + `torch.nn` | `vectorphile` + `vectorphile.nn` |
| `numpy` | `vectorphile` (Tensor ops) |
| `pandas` | `vectorphile.data` |
| `scikit-learn` | `vectorphile.ml` |
| `scipy` | `vectorphile.linalg`, `.stats`, `.signal`, `.optimize`, `.integrate` |
| `torchvision.transforms` | `vectorphile.transforms` |
| `torchaudio` | `vectorphile.audio` |
| `transformers` (tokenizers) | `vectorphile.tokenizers` |
| `jax.grad` / `jax.vmap` | `vectorphile.autograd` |

**Every internal operation runs on vectorphile's own `NDArray`/`Tensor` engine. No numpy, no torch, no jax under the hood.**

---

## Philosophy

Most ML workflows require assembling a fragmented ecosystem that was never designed to interoperate. Each package carries its own type system, memory model, and API conventions. The cognitive overhead is real, and the hidden incompatibilities are a persistent source of bugs.

vectorphile is built on a single premise: **the entire computational stack required for modern ML research should be expressible through one coherent API**, rooted in a single tensor primitive, backed by a single autodiff engine, and extended consistently across every subdomain.

This is not a wrapper. Every module is a ground-up implementation, designed with a research environment in mind: readable internals, stable gradients, and an architecture that exposes rather than hides the mathematics.

---

## Installation

```bash
pip install vectorphile
```

**With optional extras:**
```bash
pip install "vectorphile[viz]"    # matplotlib
pip install "vectorphile[io]"     # pyarrow, h5py, netCDF4
pip install "vectorphile[cuda]"   # cupy (GPU acceleration)
pip install "vectorphile[dev]"    # pytest, ruff
```

**From source:**
```bash
git clone https://github.com/vectorphile/vectorphile
cd vectorphile
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9

---

## Quick Start

```python
import vectorphile as vp
import vectorphile.nn as nn
import vectorphile.optim as optim

# --- Tensors ---
x = vp.randn(32, 784)
y = vp.randint(0, 10, (32,))

# --- Model ---
model = nn.Sequential(
    nn.Linear(784, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(),
    nn.Linear(128, 10),
)

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=100)

logits = model(x)
loss   = criterion(logits, y)
loss.backward()
optimizer.step()
scheduler.step()
```

---

## Module Reference

### Core

| Module | Description |
|---|---|
| `vp` | `Tensor`, all ndarray ops, broadcasting, slicing, reshaping, type casting, 300+ functions |
| `vp.autograd` | Reverse-mode AD, `no_grad`, `grad`, `jacobian`, `hessian`, `vmap`, `jvp`, `vjp`, `gradcheck`, `checkpoint` |
| `vp.linalg` | `det`, `inv`, `svd`, `eig`, `eigh`, `qr`, `cholesky`, `solve`, `lstsq`, `norm`, `matrix_exp`, `pinv`, `lu_factor`, `slogdet` |
| `vp.random` | `rand`, `randn`, `randint`, `normal`, `uniform`, `binomial`, `multinomial`, `gamma`, `beta`, `dirichlet`, `cauchy`, `gumbel` |
| `vp.fft` | 16 functions: `fft`/`ifft`, `rfft`/`irfft`, `fft2`/`ifft2`, `fftn`/`ifftn`, `rfftn`/`irfftn`, `hfft`/`ihfft`, `fftshift`, `ifftshift`, `fftfreq`, `rfftfreq` |

### Neural Networks

| Module | Description |
|---|---|
| `vp.nn` | 250+ exports — every layer, activation, loss, normalisation, attention, recurrent, and container class |
| `vp.nn.F` | Full functional API: activations, conv, pool, norm, dropout, embedding, all losses, `pad`, `interpolate`, `grid_sample`, `scaled_dot_product_attention` |
| `vp.nn.init` | `xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`, `kaiming_normal_`, `orthogonal_`, `trunc_normal_`, `dirac_`, `lecun_normal_` |
| `vp.optim` | SGD, Adam, AdamW, RMSprop, Adagrad, NAdam, RAdam, LARS, Lion, Muon, Sophia, Adan, SAM, Lookahead, ScheduleFreeAdamW, AdamMini + 10 LR schedulers |
| `vp.amp` | `autocast`, `GradScaler`, `BF16Tensor`, `BF16Linear` |

**Layer categories:**

```
Layers:    Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d, Embedding, EmbeddingBag
Recurrent: RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell
Attention: MultiheadAttention, TransformerEncoder/Decoder, Transformer, FlashDecodeAttention, RingAttention
Norms:     BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d, RMSNorm
Pooling:   MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d, GlobalAvgPool2d
Padding:   ZeroPad2d, ReflectionPad1d/2d, ReplicationPad1d/2d, ConstantPad1d/2d/3d
Dropout:   Dropout, Dropout2d, AlphaDropout
Upsampling:Upsample, UpsamplingNearest2d, UpsamplingBilinear2d, PixelShuffle, PixelUnshuffle
Activations (Module): ReLU, ReLU6, LeakyReLU, ELU, SELU, GELU, SiLU, Mish, Hardswish,
           Sigmoid, Tanh, Softmax, LogSoftmax, Softsign, Hardsigmoid, Softmin, Softmax2d,
           PReLU, GLU, CELU, LogSigmoid, Tanhshrink, Softshrink, Hardshrink, Swish, Snake,
           Hardtanh, Softplus, Threshold, Identity
Losses:    MSELoss, L1Loss, MAELoss, SmoothL1Loss, HuberLoss, CrossEntropyLoss, NLLLoss,
           BCELoss, BCEWithLogitsLoss, KLDivLoss, CosineSimilarityLoss, TripletMarginLoss,
           FocalLoss, CTCLoss, PoissonNLLLoss, MultiLabelMarginLoss
Containers:Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict
Advanced:  Mamba, S4Block, MoE, GCNConv, GATConv, SAGEConv, EdgeConv, CausalConv1d
Utilities: pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence,
           clip_grad_norm_, spectral_norm, weight_norm, model_summary
```

### Machine Learning

| Module | Description |
|---|---|
| `vp.ml` | Full scikit-learn-compatible API |
| `vp.preprocessing` | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `OneHotEncoder`, `PolynomialFeatures`, `PowerTransformer`, `KBinsDiscretizer` |
| `vp.model_selection` | `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, `StratifiedKFold`, `train_test_split` |
| `vp.metrics` | 30+ metrics: accuracy, precision, recall, F1, ROC-AUC, BLEU, ROUGE, WER, ECE, NDCG, silhouette |
| `vp.decomposition` | PCA, TruncatedSVD, NMF, FastICA, FactorAnalysis |
| `vp.cluster` | KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering |
| `vp.neighbors` | KNeighborsClassifier, KNeighborsRegressor, BallTree |
| `vp.svm` | SVC, SVR, LinearSVC |
| `vp.ensemble` | RandomForest, ExtraTrees, GradientBoosting, Bagging, Voting, Stacking |
| `vp.manifold` | TSNE, UMAP, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding |
| `vp.gaussian_process` | `GaussianProcessRegressor`, kernels (RBF, Matern, DotProduct, etc.) |

### Data

| Module | Description |
|---|---|
| `vp.data` | `DataFrame` (60+ methods), `Series` (50+ methods), `read_csv`, `read_json`, GroupBy, merge, pivot, melt |
| `vp.data.datasets` | MNIST, CIFAR-10, TinyShakespeare, IMDb, `DataLoader`, `TensorDataset`, samplers |
| `vp.timeseries` | ARIMA, SARIMA, HoltWinters, GARCH, VAR, KalmanFilter |

**DataFrame highlights:**
```python
df.value_counts(col)        df.nunique()           df.duplicated()
df.drop_duplicates()        df.clip(lower, upper)  df.nlargest(n, col)
df.rolling(window).mean()   df.shift(periods)      df.corr()
df.groupby(col).agg(...)    df.merge(right, on=…)  df.pivot(…)
```

**Series highlights:**
```python
s.shift(n)    s.diff(n)    s.pct_change()  s.rank()
s.skew()      s.kurt()     s.autocorr(lag) s.rolling(w).std()
s.isin(vals)  s.between(l, r)              s.rolling(w).mean()
```

### Scientific Computing

| Module | Description |
|---|---|
| `vp.stats` | Descriptive + inferential stats: t-test, chi2, ANOVA, Mann-Whitney, KS-test, Wilcoxon, Levene, Bartlett, Kruskal |
| `vp.signal` | STFT, spectrogram, butter filter, filtfilt, Hilbert, resample, Welch PSD, find_peaks, sosfilt, freqz |
| `vp.optimize` | `minimize` (BFGS/Nelder-Mead/Powell), `linprog`, `curve_fit`, `differential_evolution`, `dual_annealing`, `basinhopping`, `nnls` |
| `vp.integrate` | `quad`, `dblquad`, `nquad`, `solve_ivp`, `odeint`, `trapezoid`, `simpson`, `romberg`, `cumulative_trapezoid` |
| `vp.interpolate` | `interp1d`, `CubicSpline`, `BSpline`, `griddata`, `Akima1DInterpolator`, `krogh_interpolate`, `pchip_interpolate` |
| `vp.spatial` | `KDTree`, distance metrics (Euclidean, Minkowski, Mahalanobis, Chebyshev, BrayCurtis), `BallTree` |
| `vp.ndimage` | `gaussian_filter`, `sobel`, `label`, `zoom`, `rotate`, `binary_erosion`, `distance_transform_edt` |
| `vp.special` | `gamma`, `lgamma`, `digamma`, `erf`, `erfinv`, `ndtri`, `betainc`, `gammainc`, `j0`, `j1`, `i0`, `hyp2f1`, Legendre/Hermite/Laguerre polynomials |
| `vp.polynomial` | `polyadd`, `polymul`, `polydiv`, `polyfit`, `polyroots`, `polyvander`, `polyint`, `polyder` |
| `vp.sparse` | CSR, CSC, COO sparse matrices |
| `vp.ma` | `MaskedArray`, `masked_where`, `masked_invalid`, all masked statistics |

### Domain-Specific

| Module | Description |
|---|---|
| `vp.transforms` | 22 image transforms: `Compose`, `Resize`, `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, `GaussianBlur`, `RandomErasing`, `FiveCrop`, `TenCrop`, `Normalize`, `AugMix`-compatible pipeline |
| `vp.audio` | `stft`, `istft`, `mel_spectrogram`, `mfcc`, `yin_pitch`, `resample`, `MelSpectrogram`, `MFCC`, `Spectrogram` |
| `vp.vision` | Vision transforms, `make_grid`, `load_image`, `save_image`, `extract_features`, ImageNet defaults |
| `vp.cv` | OpenCV-style API: `imread`, `imwrite`, `GaussianBlur`, `Canny`, `Sobel`, `threshold`, `erode`, `dilate`, `resize`, `cvtColor`, `calcHist` |
| `vp.image` | PIL-style `Image` class with `.crop`, `.resize`, `.rotate`, `.filter`, `.save`, `.convert` |
| `vp.nlp` | `Tokenizer`, `Vocabulary`, `TFIDFVectorizer`, `BM25`, `NGramModel`, `SkipGram` |
| `vp.tokenizers` | BPE, WordPiece, SentencePiece, GPT-2 BPE, Llama, ByteLevelBPE, TikToken-compatible |
| `vp.generation` | `GreedySearchDecoder`, `SamplingDecoder`, `BeamSearchDecoder`, `SpeculativeDecoder`, `generate`, top-k/top-p/min-p/typical filtering, repetition penalty |
| `vp.rl` | DQN, DoubleDQN, DuelingDQN, REINFORCE, A2C, PPO, SAC, TD3, PER, environments (CartPole, MountainCar, Pendulum, GridWorld) |
| `vp.diffusion` | `DDPMScheduler`, `DDIMScheduler`, `FlowMatchingScheduler`, `DPMSolverScheduler` |
| `vp.graph` | `Graph`, `GraphData`, `batch_graphs`, `negative_sampling`, `add_self_loops` |
| `vp.causal` | `CausalGraph`, `PropensityScoreMatching`, `IPWEstimator`, `DifferenceInDifferences`, `DoubleML`, `InstrumentalVariable`, `RegressionDiscontinuity` |
| `vp.manifold` | TSNE, UMAP, MDS, Isomap, LLE, SpectralEmbedding |

### Production & Infrastructure

| Module | Description |
|---|---|
| `vp.quantization` | INT8/INT4/FP8/FP16 quantization, `QuantizedModel`, `MinMaxCalibrator`, `HistogramCalibrator`, `FP8Linear`, magnitude pruning |
| `vp.peft` | LoRA, QLoRA, AdaLoRA, IA³, prefix tuning, prompt tuning |
| `vp.distributed` | `DataParallel`, `DDP`, `GradientAccumulator`, `MixedPrecisionTrainer` |
| `vp.export` | ONNX, TorchScript, CoreML, SavedModel, GGUF (Q4_0/Q8_0), `TensorRTExporter`, `compile` (eager/inductor) |
| `vp.inference` | `PagedKVCache`, `GenerationPipeline`, `BatchedInference`, `speculative_decode`, samplers |
| `vp.trainer` | PyTorch Lightning-style `Trainer` + `TrainableModel` |
| `vp.checkpoint` | `save`, `load`, `ModelCheckpoint`, `EarlyStopping`, `LRSchedulerCallback` |
| `vp.callbacks` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoardLogger, WandBLogger, LRFinder |
| `vp.automl` | `AutoClassifier`, `AutoRegressor`, `MicroNAS`, `create_study`, hyperparameter search |
| `vp.privacy` | `GaussianMechanism`, `LaplaceMechanism`, `DPSGDOptimizer`, `PrivacyAccountant`, `RandomizedResponse` |
| `vp.federated` | Federated averaging, differential privacy integration |
| `vp.uncertainty` | `MCDropout`, `DeepEnsemble`, `SplitConformalPredictor`, `TemperatureScaling`, `PlattScaling`, ECE |
| `vp.pretrained` | HuggingFace Hub download, safetensors parser, weight mappers for GPT-2/BERT/ViT/LLaMA |
| `vp.hub` | `save_model`, `load_model`, `Checkpointer`, model cards |
| `vp.profiler` | Model profiling, FLOPs counter, memory tracker |

---

## Usage Examples

### Autograd

```python
import vectorphile as vp

x = vp.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2 + 3 * x - vp.log(x + 1)).sum()
y.backward()
print(x.grad)   # [2x + 3 - 1/(x+1)] at [2, 3]

# Higher-order: Jacobian and Hessian
from vectorphile.autograd import jacobian, hessian
J = jacobian(lambda x: x ** 2, x)
H = hessian(lambda x: (x ** 2).sum(), x)
```

### Transformers

```python
import vectorphile.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model   = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

src = vp.randn(10, 32, 512)   # (seq_len, batch, d_model)
tgt = vp.randn(20, 32, 512)
out = model(src, tgt)         # (20, 32, 512)
```

### RNNs with Packed Sequences

```python
import vectorphile.nn as nn
from vectorphile.nn import pad_sequence, pack_padded_sequence, pad_packed_sequence

seqs    = [vp.randn(5, 16), vp.randn(3, 16), vp.randn(2, 16)]
padded  = pad_sequence(seqs, batch_first=True)
packed  = pack_padded_sequence(padded, lengths=[5, 3, 2], batch_first=True)

lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, batch_first=True)
out, (h_n, c_n) = lstm(packed)
output, lengths = pad_packed_sequence(out, batch_first=True)
```

### DataFrame & Series

```python
import vectorphile.data as vpd

df = vpd.read_csv('sales.csv')
df['margin'] = (df['revenue'] - df['cost']) / df['revenue']

# New in v0.5.0
print(df.nlargest(5, 'revenue'))
print(df.nunique())
print(df['revenue'].rolling(7).mean())
print(df['region'].value_counts(normalize=True))

s = df['revenue']
print(s.pct_change())
print(s.autocorr(lag=1))
print(f"skew={s.skew():.3f}  kurt={s.kurt():.3f}")
```

### Classical ML Pipeline

```python
from vectorphile import ml
from vectorphile.preprocessing import StandardScaler, PolynomialFeatures
from vectorphile.model_selection import cross_val_score, GridSearchCV

# Pipeline
pipeline = ml.Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('clf', ml.LogisticRegression(C=1.0, max_iter=1000)),
])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')
print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Hyperparameter search
grid = GridSearchCV(ml.RandomForestClassifier(), {
    'n_estimators': [100, 200], 'max_depth': [None, 10, 20]
}, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### FFT & Signal Processing

```python
import vectorphile.fft as vfft
import vectorphile.signal as sig

# N-dimensional FFT
X3d = vfft.fftn(vp.randn(8, 8, 8))
X_shifted = vfft.fftshift(X3d)

# Filtering
b, a = sig.butter(N=4, Wn=0.1, btype='low')
clean = sig.filtfilt(b, a, noisy)
f, t, Zxx = sig.stft(signal, n_fft=512, hop_length=128)
```

### Quantization & Export

```python
from vectorphile.quantization import quantize, QuantizedModel
from vectorphile.export import to_onnx

# Post-training quantization
q_model = quantize(model, scheme='int8', symmetric=True)

# ONNX export
to_onnx(model, input_shape=(1, 3, 224, 224), path='model.onnx', opset=17)
```

### Distributions & Sampling

```python
from vectorphile.distributions import Normal, Beta, StudentT, Categorical

n = Normal(loc=0.0, scale=1.0)
print(n.sample((100,)).shape)   # (100,)
print(n.log_prob(vp.zeros(5)))
print(n.entropy())

b = Beta(concentration1=2.0, concentration0=5.0)
print(b.mean, b.variance)
```

---

## Architecture

```
vectorphile/
├── core/tensor.py           Tensor class — 300+ ops, operator overloads, broadcasting
├── _core/ndarray.py         NDArray primitive (no numpy dependency)
├── autograd/
│   ├── engine.py            Reverse-mode AD, computation graph, Node
│   └── functional.py        grad, jacobian, hessian, vmap, jvp, vjp, checkpoint
├── nn/
│   ├── module.py            Module base, parameter registration, state_dict
│   ├── layers.py            Linear, Conv1d/2d/3d, ConvTranspose, Embedding, Norm, Dropout
│   ├── activations.py       ReLU, GELU, SiLU, Mish, Swish, Snake, Identity...
│   ├── extra_layers.py      PReLU, GLU, Softsign, Hardsigmoid, Softmax2d, Mamba, S4, GNN...
│   ├── attention.py         MultiheadAttention, Transformer, Flash/Ring/Decode attention
│   ├── recurrent.py         RNN, LSTM, GRU, cell variants
│   ├── losses.py            All loss functions incl. L1Loss, SmoothL1Loss, FocalLoss
│   ├── functional.py        100+ functional ops
│   ├── utils.py             clip_grad_norm_, pad_sequence, pack_padded_sequence
│   ├── pooling.py           All pool variants incl. adaptive
│   ├── init.py              Weight initializers
│   └── ...
├── optim/                   15+ optimizers + 10 LR schedulers
├── data/                    DataFrame, Series, GroupBy, read_csv, read_json
├── ml/                      Full sklearn-compatible API
├── linalg/                  LAPACK-equivalent from scratch
├── fft/                     16-function FFT suite
├── distributions/           15+ probability distributions
├── transforms/              22 image/tensor transforms
├── audio/ cv/ image/ vision/ nlp/ tokenizers/ generation/
├── rl/ diffusion/ causal/ graph/ manifold/ gaussian_process/
├── quantization/ peft/ distributed/ export/ inference/
├── trainer/ checkpoint/ callbacks/ automl/ privacy/ federated/
├── stats/ signal/ optimize/ integrate/ interpolate/ spatial/ ndimage/
├── special/ polynomial/ sparse/ ma/ symbolic/ uncertainty/
└── timeseries/ random/ amp/ profiler/ pretrained/ hub/
```

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run specific modules
pytest tests/test_tensor.py tests/test_nn.py -v
pytest tests/test_data.py tests/test_ml.py -v
```

| Test Suite | Count | Status |
|---|---|---|
| `test_tensor` + `test_nn` + `test_ml` + `test_data` | **156** | ✅ Pass |
| `test_session14` + `test_v050_features` + `test_session15` + `test_quantization_v2` | **122** | ✅ Pass |
| `test_data` + `test_flash_attention` + `test_gpu` + `test_layers_extra` | **81** | ✅ Pass |

---

## Changelog

### v0.5.0 — Feature-Complete Release
- **nn (250+ exports):** Softsign, Hardsigmoid, Softmin, Softmax2d activations; L1Loss, SmoothL1Loss; ConstantPad1d/3d, ReflectionPad1d, ReplicationPad1d/2d, UpsamplingNearest2d/Bilinear2d, ConvTranspose3d, AdaptiveMaxPool1d/3d; PackedSequence + RNN sequence utilities (pad_sequence, pack_padded_sequence, pad_packed_sequence)
- **nn.functional:** softmin, softmax2d, lp_pool1d/2d, adaptive_max/avg_pool3d, local_response_norm + 6 more
- **Tensor (14 new functions):** fliplr, flipud, vstack, hstack, dstack, column_stack, row_stack, addr, index_copy, block_diag, diag_embed, scatter_add, index_select, bucketize — all with instance method equivalents
- **FFT (16 total):** fftn, ifftn, rfftn, irfftn, hfft, ihfft added
- **Data:** 9 new DataFrame methods, 10 new Series methods including rolling, rank, skew, kurt, autocorr
- **Transforms:** GaussianBlur, RandomErasing, FiveCrop, TenCrop
- **Bug fixes:** distributions `_math` alias, fft namespace collision, shadowed builtins in tensor ops

### v0.4.0 — Ecosystem Parity
- GGUF binary format (Q4_0, Q8_0), 25+ evaluation metrics, 10 training callbacks
- Keras/Lightning-style Trainer, 11 LR schedulers, einops rearrange/reduce
- Forward/backward hooks, text generation suite, vision transforms, datasets

### v0.3.0 — Production Completeness
- 5-tier CUDA backend, BPE/WordPiece/SentencePiece tokenizers from scratch
- Distributed training (DataParallel, DDP), ONNX/TorchScript/CoreML export
- RL suite (DQN → TD3), pretrained weights hub, Flash Attention, QAT
- Advanced optimizers: Lion, Sophia, Adan, SAM, Lookahead

### v0.1–0.2 — Core
- Full Tensor API with reverse-mode autograd
- Complete nn layer set, all classical ML algorithms, DataFrame, linalg, stats, FFT, signal

---

## Roadmap

- **v0.6.0** — Real CUDA C extensions (cuBLAS GEMM, cuDNN conv), fp16/bf16 AMP on GPU, Triton fused kernels, multi-node NCCL distributed

---

## Contributing

Issues, proposals, and pull requests are welcome at [github.com/vectorphile/vectorphile](https://github.com/vectorphile/vectorphile).

---

## License

Apache-2.0 © vectorphile
