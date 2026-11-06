# vectorphile

**The unified ML/DL Python library.** One package that replaces PyTorch + scikit-learn + NumPy + Pandas + SciPy.

```python
import vectorphile as vp
```

---

## What's inside

| Module | Replaces | What you get |
|---|---|---|
| `vp` (core) | NumPy | Tensor, all ndarray ops, autograd |
| `vp.nn` | PyTorch `nn` | Layers, activations, losses, Sequential |
| `vp.optim` | PyTorch `optim` | SGD, Adam, AdamW, RMSprop, Adagrad, LR schedulers |
| `vp.autograd` | PyTorch autograd | Reverse-mode autodiff, `no_grad` |
| `vp.ml` | scikit-learn | Linear models, trees, forests, boosting, clustering, decomposition, neighbors, Naive Bayes, SVM, preprocessing, metrics, model selection |
| `vp.ml.boost` | XGBoost | `XGBClassifier`, `XGBRegressor` |
| `vp.data` | Pandas | `DataFrame`, `Series`, `read_csv`, `read_json`, GroupBy, merge, pivot, melt |
| `vp.linalg` | NumPy linalg | det, inv, svd, eig, lstsq, solve, norm, qr, cholesky, pinv |
| `vp.stats` | SciPy stats | mean, std, corr, cov, zscore, t-test, chi2, ANOVA, Pearson/Spearman |
| `vp.random` | NumPy random | seed, rand, randn, choice, normal, binomial, multinomial |
| `vp.fft` | NumPy fft | fft, ifft, rfft, fft2, fftshift |
| `vp.signal` | SciPy signal | stft, convolve, correlate, butter, filtfilt, spectrogram |
| `vp.sparse` | SciPy sparse | csr_matrix, csc_matrix, coo_matrix |
| `vp.optimize` | SciPy optimize | minimize, root, curve_fit, differential_evolution |
| `vp.amp` | PyTorch amp | autocast, GradScaler (Phase 2: real CUDA fp16) |
| `vp.utils.data` | PyTorch DataLoader | `Dataset`, `TensorDataset`, `DataLoader` |

---

## Quick start

### Tensors

```python
import vectorphile as vp

x = vp.randn(3, 4)
y = vp.ones(4, 5)
z = x @ y                    # (3, 5)

t = vp.tensor([1, 2, 3], dtype='float32')
t2 = t.reshape(1, 3).T       # (3, 1)
```

### Neural networks

```python
model = vp.nn.Sequential(
    vp.nn.Linear(784, 256),
    vp.nn.BatchNorm1d(256),
    vp.nn.GELU(),
    vp.nn.Dropout(0.3),
    vp.nn.Linear(256, 10),
)

optimizer = vp.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = vp.optim.CosineAnnealingLR(optimizer, T_max=100)
criterion = vp.nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### Classical ML

```python
from vectorphile import ml

# Split
X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2)

# Preprocess
scaler = ml.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
clf = ml.RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(f"Accuracy: {ml.accuracy_score(y_test, clf.predict(X_test)):.4f}")

# Dimensionality reduction
pca = ml.PCA(n_components=2)
X_2d = pca.fit_transform(X)
```

### DataFrames

```python
df = vp.read_csv('data.csv')

# Filter, transform
senior = df[df['years_exp'] > 5]
df['bonus'] = df['salary'] * 0.1

# GroupBy
dept_avg = df.groupby('dept').mean()

# Merge
df2 = df.merge(reviews, on='employee_id', how='left')

# Stats
df.describe()
df[['salary', 'score']].corr()
```

### Autograd

```python
x = vp.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2 + 3 * x).sum()
y.backward()
print(x.grad)  # [2*x+3] = [7, 9]

with vp.no_grad():
    pred = model(X_test)   # no gradient tracking
```

---

## Installation

```bash
# From source
pip install -e .

# With optional dependencies
pip install -e ".[full]"   # adds scipy, pyarrow
pip install -e ".[dev]"    # adds pytest
```

**Requirements:** Python >= 3.9, NumPy >= 1.24

---

## Architecture

```
vectorphile/
├── core/           ← Tensor class, all tensor ops
├── autograd/       ← Reverse-mode autodiff engine
├── nn/             ← Layers, activations, losses, Sequential
├── optim/          ← Optimizers + LR schedulers
├── ml/             ← Classical ML (sklearn-compatible API)
│   └── boost/      ← XGBoost-style gradient boosting
├── data/           ← DataFrame, Series, GroupBy (pandas-compatible)
├── linalg/         ← Linear algebra
├── stats/          ← Statistical functions
├── random/         ← Random number generation
├── fft/            ← FFT operations
├── signal/         ← Digital signal processing
├── sparse/         ← Sparse matrices
├── optimize/       ← Numerical optimization
├── amp/            ← Automatic mixed precision (Phase 2)
└── utils/
    └── data/       ← Dataset, TensorDataset, DataLoader
```

**Phase 1** (current): Pure Python + NumPy backend.
**Phase 2** (roadmap): C extensions for core ops, real CUDA support via cuBLAS/cuDNN, fp16/bf16 AMP.

---

## Running tests

```bash
pytest tests/ -v
```

---

## License

Apache-2.0
