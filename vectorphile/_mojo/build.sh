#!/usr/bin/env bash
# Build vectorphile Mojo CPU kernels into a shared library.
#
# Usage:
#   cd vectorphile/_mojo && bash build.sh
#
# Output:
#   libvpkernels.so   (CPU — always built)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " vectorphile Mojo kernel build"
echo "=================================================="
mojo --version
echo "--------------------------------------------------"

PLATFORM="$(uname -s)"
CPU_LIB="libvpkernels.so"
if [ "$PLATFORM" = "Darwin" ]; then
    CPU_LIB="libvpkernels.dylib"
fi

echo "[1/1] Building CPU kernels → $CPU_LIB"
mojo build --emit shared-lib -O3 -o "$CPU_LIB" vpkernels.mojo
echo "    => $SCRIPT_DIR/$CPU_LIB"

echo ""
echo "=================================================="
echo " Build complete!"
echo "  CPU:  $SCRIPT_DIR/$CPU_LIB"
echo ""
echo " Verify from Windows:"
echo "   python -c 'from vectorphile._mojo import loader; print(loader.available())'"
echo "=================================================="
