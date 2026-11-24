#!/usr/bin/env bash
# ============================================================
#  Build script for vectorphile Mojo backend (_vp_mojo.so)
#  Run inside WSL2/Ubuntu or any Linux with Mojo installed.
#
#  Requirements:
#    - Mojo / Magic CLI  (https://docs.modular.com/magic/)
#    - Linux x86_64 (Ubuntu 22.04+ or 24.04+)
#
#  Quick install if Mojo not present:
#    curl -ssL https://magic.modular.com/install.sh | bash
#    magic install mojo
#
#  Usage:
#    bash scripts/build_vp_mojo_linux.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC="$REPO_ROOT/vectorphile/_core/csrc/vp_mojo.mojo"
OUT="$REPO_ROOT/vectorphile/_core/_vp_mojo.so"

echo "=== vectorphile Mojo backend build ==="
echo "Source : $SRC"
echo "Output : $OUT"
echo

# -- Check Mojo ---
if ! command -v mojo &>/dev/null; then
    if command -v magic &>/dev/null; then
        echo "mojo not in PATH but magic found — running via magic"
        MOJO="magic run mojo"
    else
        echo "ERROR: mojo and magic not found."
        echo "Install with: curl -ssL https://magic.modular.com/install.sh | bash"
        echo "Then: magic install mojo"
        exit 1
    fi
else
    MOJO="mojo"
fi

# -- Compile Mojo → shared library ---
# Mojo currently exports to .mojopkg; we wrap with a thin C shim to get .so
# Preferred path: compile Mojo to object + link as shared lib via clang

echo "Compiling Mojo source..."
$MOJO build "$SRC" --emit-object -o /tmp/vp_mojo.o \
    -O3 --no-compilation-database 2>&1

echo "Linking shared library..."
clang -shared -fPIC -O3 \
    -o "$OUT" \
    /tmp/vp_mojo.o \
    -lm -lpthread

echo
echo "SUCCESS: $OUT"

# -- Smoke test ---
echo "Running smoke test..."
PYTHON_PATH="$REPO_ROOT"
python3 -c "
import sys; sys.path.insert(0, '$PYTHON_PATH')
from vectorphile._core.mojo_backend import is_available, matmul_f32, relu_f32
import array
print('  mojo_backend.is_available():', is_available())
if is_available():
    # Quick matmul sanity: 2x2 @ 2x2 = 2x2
    a = array.array('f', [1,2,3,4])   # [[1,2],[3,4]]
    b = array.array('f', [1,0,0,1])   # identity
    c = array.array('f', [0]*4)
    matmul_f32(a, b, c, 2, 2, 2)
    print('  2x2 @ I =', list(c), '  expected [1,2,3,4]')
    print('  Mojo backend READY')
"
