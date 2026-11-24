"""
Build script for vectorphile Mojo kernels.

Run from project root (inside WSL2):
    python vectorphile/_mojo/build.py

Produces:
    vectorphile/_mojo/libvpkernels.so  (Linux)
    vectorphile/_mojo/libvpkernels.dylib  (macOS)
"""
import subprocess
import sys
import shutil
from pathlib import Path

MOJO_DIR = Path(__file__).parent
KERNELS = ["gemm", "elementwise", "reductions", "random", "conv", "fused", "ndarray"]


def find_mojo():
    mojo = shutil.which("mojo")
    if mojo:
        return mojo
    # Magic-installed path
    magic_path = Path.home() / ".modular" / "bin" / "mojo"
    if magic_path.exists():
        return str(magic_path)
    raise RuntimeError(
        "mojo not found. Install via:\n"
        "  curl -ssL https://magic.modular.com | bash\n"
        "  source ~/.bashrc\n"
        "  magic install max"
    )


def build():
    mojo = find_mojo()
    print(f"Found mojo: {mojo}")

    # Check version
    result = subprocess.run([mojo, "--version"], capture_output=True, text=True)
    print(f"Mojo version: {result.stdout.strip()}")

    out_lib = MOJO_DIR / "libvpkernels.so"

    # Build each kernel as an object, then link
    obj_files = []
    for kernel in KERNELS:
        src = MOJO_DIR / f"{kernel}.mojo"
        obj = MOJO_DIR / f"{kernel}.o"
        print(f"Compiling {kernel}.mojo ...")
        r = subprocess.run(
            [mojo, "build", "--emit-object", str(src), "-o", str(obj),
             "-O3", "--march=native"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr}")
            sys.exit(1)
        obj_files.append(str(obj))
        print(f"  OK -> {obj.name}")

    # Link into shared library
    print(f"Linking {out_lib.name} ...")
    cc = shutil.which("cc") or shutil.which("gcc") or "cc"
    r = subprocess.run(
        [cc, "-shared", "-fPIC", "-O3", "-o", str(out_lib)] + obj_files + ["-lm"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  Linker error: {r.stderr}")
        sys.exit(1)

    # Cleanup objects
    for f in obj_files:
        Path(f).unlink(missing_ok=True)

    print(f"\nBuild successful: {out_lib}")
    print("vectorphile will auto-detect and use Mojo kernels on next import.")


if __name__ == "__main__":
    build()
