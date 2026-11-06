"""
Example 02: MLP Classifier (MNIST-style synthetic data)

Demonstrates:
- Multi-layer neural network
- Adam optimizer
- LR scheduling
- Training + validation loop
- Accuracy and loss tracking
"""
import numpy as np
import vectorphile as vp

np.random.seed(42)

# ── Synthetic dataset (8-class, 32-feature) ──────────────────────────────────
n_train, n_val = 800, 200
n_features, n_classes = 32, 8

X_all = np.random.randn(n_train + n_val, n_features).astype(np.float32)
y_all = (np.argmax(X_all[:, :n_classes], axis=1)).astype(np.int64)

X_train = vp.tensor(X_all[:n_train])
y_train = vp.tensor(y_all[:n_train].astype(np.int64))
X_val = vp.tensor(X_all[n_train:])
y_val = vp.tensor(y_all[n_train:].astype(np.int64))

# ── Build MLP ─────────────────────────────────────────────────────────────────
model = vp.nn.Sequential(
    vp.nn.Linear(n_features, 128),
    vp.nn.BatchNorm1d(128),
    vp.nn.GELU(),
    vp.nn.Dropout(0.3),
    vp.nn.Linear(128, 64),
    vp.nn.GELU(),
    vp.nn.Linear(64, n_classes),
)

optimizer = vp.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = vp.optim.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = vp.nn.CrossEntropyLoss()

# ── Training loop ─────────────────────────────────────────────────────────────
print("Training MLP on synthetic 8-class dataset")
print("=" * 55)
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Acc':>8}  {'LR':>10}")
print("-" * 55)

batch_size = 64
n = len(X_train)

for epoch in range(1, 31):
    model.train()
    perm = np.random.permutation(n)
    epoch_loss = 0.0
    epoch_correct = 0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        Xb = vp.tensor(X_train._data[idx])
        yb = vp.tensor(y_train._data[idx])
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item()) * len(idx)
        preds = logits._data.argmax(axis=1)
        epoch_correct += (preds == yb._data).sum()

    train_loss = epoch_loss / n
    train_acc = epoch_correct / n

    model.eval()
    val_logits = model(X_val)
    val_preds = val_logits._data.argmax(axis=1)
    val_acc = float((val_preds == y_val._data).mean())

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.1%}  {val_acc:>7.1%}  {current_lr:>10.6f}")

print("=" * 55)
print(f"\nFinal validation accuracy: {val_acc:.1%}")
print(f"Model has {sum(p.size for p in model.parameters()):,} parameters")
