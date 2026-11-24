"""Autograd correctness + performance benchmark."""
import time
import numpy as np
import vectorphile as vp
from vectorphile.core.tensor import Tensor
from vectorphile.autograd import gradcheck, jacobian, hessian, vmap


def bench(fn, reps=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    print("=== gradcheck ===")
    x = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
    ok = gradcheck(lambda t: (t * t + t).sum(), [x])
    print(f"gradcheck linear: {'PASS' if ok else 'FAIL'}")

    x2 = Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)
    ok2 = gradcheck(lambda t: (t @ t).sum(), [x2])
    print(f"gradcheck matmul: {'PASS' if ok2 else 'FAIL'}")

    print("\n=== jacobian ===")
    x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    J = jacobian(lambda t: Tensor(np.asarray(t._data) ** 2), x)
    expected = np.diag(2.0 * np.asarray(x._data))
    err = np.abs(J - expected).max()
    print(f"jacobian x^2: max_err={err:.2e} {'PASS' if err < 1e-4 else 'FAIL'}")

    print("\n=== vmap gradient ===")
    x = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
    batched = vmap(lambda xi: xi * xi)
    out = batched(x)
    out.sum().backward()
    err = np.abs(np.asarray(x.grad._data) - 2.0 * np.asarray(x._data)).max()
    print(f"vmap x^2 grad: max_err={err:.2e} {'PASS' if err < 1e-5 else 'FAIL'}")

    print("\n=== hessian ===")
    x = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    H = hessian(lambda t: (Tensor(np.asarray(t._data) ** 2)).sum(), x)
    expected_H = np.diag([2.0, 2.0])
    err = np.abs(H - expected_H).max()
    print(f"hessian x^2: max_err={err:.2e} {'PASS' if err < 1e-2 else 'FAIL'}")

    print("\n=== backward throughput ===")
    import vectorphile.nn as nn, vectorphile.nn.functional as F
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    x = Tensor(np.random.randn(32, 128).astype(np.float32))
    y = Tensor(np.random.randint(0, 10, 32).astype(np.int64), dtype='int64')

    def step():
        for p in model.parameters():
            p.grad = None
        F.cross_entropy(model(x), y).backward()

    t = bench(step, reps=20)
    print(f"backward (32x128->10): {t:.2f} ms/step")


if __name__ == '__main__':
    main()
