#!/usr/bin/env bash
# Build _vp_core.so on Linux (GCC with AVX2+OpenMP) and macOS (Clang with OpenMP via Homebrew)
# Usage: bash scripts/build_vp_core_linux.sh [python_executable]
#
# Requirements:
#   Linux:  gcc >= 9, openmp (usually included), python3-dev
#   macOS:  clang (Xcode Command Line Tools), libomp via Homebrew (`brew install libomp`)
#           or XCode with OpenMP enabled
#
# Optional: ispc >= 1.25 for ISPC kernel acceleration

set -euo pipefail

PYTHON="${1:-python3}"
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PYINC=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))")
REPOROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRCDIR="$REPOROOT/vectorphile/_core/csrc"
ISPCDIR="$REPOROOT/vectorphile/_core/ispc"
OUTDIR="$REPOROOT/vectorphile/_core"
ISPC="${ISPC_EXE:-ispc}"

OS="$(uname -s)"
echo "Building _vp_core for $OS with Python $PYVER at $PYINC"

# ── Platform-specific compiler flags ──────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
    # macOS: use clang; OpenMP via Homebrew libomp
    CC="clang"
    OMP_FLAGS=""
    if brew --prefix libomp &>/dev/null 2>&1; then
        OMP_PREFIX="$(brew --prefix libomp)"
        OMP_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include -L${OMP_PREFIX}/lib -lomp"
    else
        echo "WARNING: libomp not found (brew install libomp). Building without OpenMP."
    fi
    # ARM (Apple Silicon): native + neon; x86_64: AVX2
    ARCH_FLAGS=""
    MACHINE="$(uname -m)"
    if [ "$MACHINE" = "arm64" ]; then
        ARCH_FLAGS="-mcpu=apple-m1 -DVECTORPHILE_APPLE_SILICON"
    else
        ARCH_FLAGS="-mavx2 -mfma"
    fi
    SHARED_FLAG="-dynamiclib"
    EXT_SUFFIX=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
    OUTPUT="${OUTDIR}/_vp_core${EXT_SUFFIX}"
else
    # Linux: GCC with AVX2 + OpenMP
    CC="gcc"
    OMP_FLAGS="-fopenmp"
    ARCH_FLAGS="-mavx2 -mfma"
    SHARED_FLAG="-shared"
    EXT_SUFFIX=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
    OUTPUT="${OUTDIR}/_vp_core${EXT_SUFFIX}"
fi

# ── Optional: ISPC kernels ────────────────────────────────────────────────────
ISPC_OBJ=""
ISPC_DEFS=""
if command -v "$ISPC" &>/dev/null 2>&1; then
    echo "Compiling ISPC kernels..."
    if "$ISPC" "${ISPCDIR}/vp_kernels.ispc" \
        --target=avx2-i32x8 --arch=x86-64 --opt=fast-math \
        -o "${ISPCDIR}/vp_kernels.o" -h "${ISPCDIR}/vp_kernels_ispc.h" 2>/dev/null; then
        ISPC_OBJ="${ISPCDIR}/vp_kernels.o"
        ISPC_DEFS="-DVP_HAVE_ISPC -I${ISPCDIR}"
        echo "ISPC kernels OK."
    else
        echo "WARNING: ISPC failed, building without ISPC acceleration."
    fi
else
    echo "ISPC not found (optional). Building without ISPC acceleration."
fi

# ── Compile ───────────────────────────────────────────────────────────────────
echo "Compiling ${SRCDIR}/vp_core.c -> ${OUTPUT}"
# shellcheck disable=SC2086
"$CC" \
    -O3 -ffast-math \
    $ARCH_FLAGS \
    $OMP_FLAGS \
    $ISPC_DEFS \
    -fPIC $SHARED_FLAG \
    -I"$PYINC" \
    "${SRCDIR}/vp_core.c" \
    $ISPC_OBJ \
    -o "$OUTPUT" \
    -lm

echo "Build exit code: $?"
echo "Output: $OUTPUT"
