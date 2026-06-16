#!/usr/bin/env bash
# FlowPilot — Jetson Orin Nano environment bootstrap
# Tested against JetPack 6.x (L4T 36.x), Python 3.11
#
# Run once on a fresh Jetson image:
#   chmod +x scripts/setup_jetson.sh && ./scripts/setup_jetson.sh

set -euo pipefail

PYTHON=python3.11
VENV=.venv

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    libopencv-dev python3-opencv \
    gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    can-utils \
    build-essential cmake ninja-build

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo "[2/5] Creating Python 3.11 virtualenv..."
$PYTHON -m venv $VENV
source $VENV/bin/activate
pip install --upgrade pip wheel setuptools

# ── 3. PyTorch (Jetson wheel from NVIDIA) ─────────────────────────────────────
# JetPack 6 ships CUDA 12.x. The NVIDIA Jetson PyTorch wheel index is
# separate from the standard PyPI wheels (which are x86-only).
# Check https://forums.developer.nvidia.com/t/pytorch-for-jetson for the
# latest stable URL for your JetPack version.
echo "[3/5] Installing PyTorch for Jetson (JetPack 6 / CUDA 12)..."
TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
pip install --no-cache-dir \
    --extra-index-url "${TORCH_WHEEL_URL}" \
    "torch>=2.3.0" "torchvision>=0.18.0"

# Verify CUDA is visible
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
    print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, \
device: {torch.cuda.get_device_name(0)}')"

# ── 4. Project dependencies ───────────────────────────────────────────────────
echo "[4/5] Installing FlowPilot dependencies..."
pip install -e ".[dev,cuda]"

# ── 5. Smoke test ─────────────────────────────────────────────────────────────
echo "[5/5] Running hardware smoke tests..."
python -m pytest tests/test_hardware.py -m cuda -v

echo ""
echo "Setup complete. Activate with: source ${VENV}/bin/activate"
echo "Run benchmark:  make bench"
