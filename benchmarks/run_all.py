"""Run all vectorphile benchmarks."""
import subprocess, sys, os

BENCHMARKS = ['bench_gemm.py', 'bench_nn.py', 'bench_autograd.py']

if __name__ == '__main__':
    here = os.path.dirname(__file__)
    for b in BENCHMARKS:
        path = os.path.join(here, b)
        print(f'\n{"=" * 50}')
        print(f'Running: {b}')
        print('=' * 50)
        result = subprocess.run([sys.executable, path], capture_output=False)
        if result.returncode != 0:
            print(f'ERROR: {b} exited with code {result.returncode}')
