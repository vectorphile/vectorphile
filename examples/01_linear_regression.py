"""
Example 01: Linear Regression with vectorphile

Demonstrates:
- Tensor creation and arithmetic
- Neural network Linear layer for regression
- SGD optimizer
- Training loop
- Comparison with vp.ml.LinearRegression
"""
import numpy as np
import vectorphile as vp

np.random.seed(42)

# ── Generate synthetic data ───────────────────────────────────────────────────
n_samples, n_features = 200, 1
X_np = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_np = 2.5 * X_np.flatten() + 1.0 + np.random.randn(n_samples) * 0.5

X = vp.tensor(X_np, dtype='float32')
y = vp.tensor(y_np.reshape(-1, 1), dtype='float32')

# ── Method 1: Neural network approach ────────────────────────────────────────
print("=" * 50)
print("Method 1: nn.Linear + SGD")
print("=" * 50)

model = vp.nn.Linear(1, 1)
optimizer = vp.optim.SGD(model.parameters(), lr=0.01)
loss_fn = vp.nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f}")

weight = model.weight._data.item()
bias = model.bias_param._data.item()
print(f"\nLearned: y = {weight:.3f}x + {bias:.3f}")
print(f"True:    y = 2.500x + 1.000")

# ── Method 2: Closed-form with vp.ml ─────────────────────────────────────────
print("\n" + "=" * 50)
print("Method 2: vp.ml.LinearRegression (closed-form)")
print("=" * 50)

lr_model = vp.LinearRegression()
lr_model.fit(X_np, y_np)
print(f"Learned: y = {lr_model.coef_[0]:.3f}x + {lr_model.intercept_:.3f}")
print(f"R² score: {lr_model.score(X_np, y_np):.4f}")

# ── Prediction ───────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Predictions at x = [-2, 0, 2]")
print("=" * 50)
test_x = vp.tensor([[-2.0], [0.0], [2.0]])
preds_nn = model(test_x)
preds_lr = lr_model.predict(np.array([[-2.0], [0.0], [2.0]]))
print(f"{'x':>6}  {'NN':>8}  {'LR':>8}  {'True':>8}")
for xi, nn_p, lr_p in zip([-2, 0, 2], preds_nn._data.flatten(), preds_lr):
    true_y = 2.5 * xi + 1.0
    print(f"{xi:>6.0f}  {nn_p:>8.3f}  {lr_p:>8.3f}  {true_y:>8.3f}")
