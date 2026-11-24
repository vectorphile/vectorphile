#!/usr/bin/env bash
# Run this inside WSL2 Ubuntu to install Mojo and compile vectorphile kernels.
#
# Usage (from WSL2 Ubuntu terminal):
#   cd /mnt/s/My\ Data/Github/vectorphile/vectorphile/_mojo
#   bash install_mojo_and_build.sh

set -e

echo "=================================================="
echo " vectorphile: Install Mojo + Build Kernels"
echo "=================================================="

# Step 1: Install Magic (Modular's package manager, includes Mojo)
if ! command -v magic &>/dev/null; then
    echo "[1/4] Installing Magic (Modular package manager)..."
    curl -ssL https://magic.modular.com | bash
    # Reload PATH
    export PATH="$HOME/.modular/bin:$PATH"
    source "$HOME/.bashrc" 2>/dev/null || true
else
    echo "[1/4] Magic already installed: $(magic --version)"
fi

# Step 2: Install Mojo
if ! command -v mojo &>/dev/null; then
    echo "[2/4] Installing Mojo..."
    magic install mojo
    export PATH="$HOME/.modular/bin:$PATH"
else
    echo "[2/4] Mojo already installed: $(mojo --version)"
fi

echo "[3/4] Mojo version: $(mojo --version)"

# Step 3: Build kernels
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[4/4] Building kernels in: $SCRIPT_DIR"
bash build.sh

echo ""
echo "=================================================="
echo " Done! Now test from Windows:"
echo "   python -c \"from vectorphile._mojo import loader; print('Mojo:', loader.available())\""
echo "=================================================="
