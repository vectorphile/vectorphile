"""Neural network forward/backward benchmark."""
import time
import numpy as np
import vectorphile as vp
import vectorphile.nn as nn
import vectorphile.nn.functional as F
from vectorphile.core.tensor import Tensor

REPS = 10


def bench(fn, reps=REPS):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    print("=== Forward pass ===")
    model = nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 10),
    )
    x = Tensor(np.random.randn(64, 512).astype(np.float32))
    t = bench(lambda: model(x))
    print(f"MLP (64x512 -> 10): {t:.2f} ms/step")

    print("\n=== Forward + backward ===")
    params = list(model.parameters())
    opt = vp.optim.Adam(params, lr=1e-3)
    y = Tensor(np.random.randint(0, 10, 64).astype(np.int64), dtype='int64')

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    t = bench(step)
    print(f"MLP train step: {t:.2f} ms/step")

    print("\n=== Conv2d forward ===")
    conv = nn.Conv2d(32, 64, 3, padding=1)
    xc = Tensor(np.random.randn(8, 32, 16, 16).astype(np.float32))
    t = bench(lambda: conv(xc))
    print(f"Conv2d (8x32x16x16 -> 64): {t:.2f} ms/step")

    print("\n=== Flash Attention forward ===")
    mha = nn.MultiheadAttention(64, 4, batch_first=True)
    xq = Tensor(np.random.randn(4, 32, 64).astype(np.float32))
    t = bench(lambda: mha(xq, xq, xq, need_weights=False))
    print(f"MHA (4x32x64, 4 heads): {t:.2f} ms/step")


if __name__ == '__main__':
    main()
