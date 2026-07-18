#!/bin/bash
# Install EyeVu + ncnn RITnet for Raspberry Pi Zero W (ARMv6)

set -e

echo "========================================="
echo "EyeVu + ncnn RITnet Installation"
echo "========================================="

# ─────────────────────────────────────────
# 1. Update and install system dependencies
# ─────────────────────────────────────────

echo "Updating package lists..."
sudo apt-get update

echo "Installing system dependencies..."

# Core build tools (always required)
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-picamera2 \
    python3-rpi.gpio

# Optional libraries (install if available, skip if not)
echo "Installing optional image/math libraries..."
for pkg in libopenjp2-7 libtiff-dev libjasper-dev libharfbuzz0b libwebp6 libatlas-base-dev libatomic1; do
    if apt-cache search "^$pkg$" | grep -q .; then
        sudo apt-get install -y "$pkg" || true
    else
        echo "  ⚠ Package $pkg not available (skipping)"
    fi
done

# ─────────────────────────────────────────
# 2. Increase swap for ncnn compilation
# ─────────────────────────────────────────

echo ""
echo "Setting up 1 GB swap for ncnn build..."
sudo dphys-swapfile swapoff || true
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# ─────────────────────────────────────────
# 3. Install core Python packages
# ─────────────────────────────────────────

echo "Installing Python packages (numpy, opencv, scipy, etc)..."
pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pillow \
    opencv-python \
    scipy \
    scikit-image

# ─────────────────────────────────────────
# 4. Build ncnn runtime from source
# ─────────────────────────────────────────

echo ""
echo "Building ncnn (this will take 30-60 minutes on Pi Zero)..."
cd /tmp
rm -rf ncnn 2>/dev/null || true
git clone --depth=1 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init --recursive

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_PYTHON=ON \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      ..

echo "Compiling ncnn (using -j1 for low RAM)..."
make -j1

# ─────────────────────────────────────────
# 5. Install Python binding
# ─────────────────────────────────────────

echo ""
echo "Installing ncnn Python binding..."
cd ../python
pip3 install . --break-system-packages

# Verify ncnn installation
echo ""
echo "Verifying ncnn installation..."
python3 -c "import ncnn; print('✓ ncnn', ncnn.__version__)"

# ─────────────────────────────────────────
# 6. Cleanup and summary
# ─────────────────────────────────────────

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Core dependencies installed:"
echo "  ✓ numpy, pillow, opencv-python, scipy, scikit-image"
echo "  ✓ picamera2, RPi.GPIO"
echo "  ✓ ncnn (RITnet ML inference)"
echo ""
echo "Next steps:"
echo "  1. Ensure ritnet.param and ritnet.bin are copied to the Pi"
echo "     (same folder as cap.py)"
echo "  2. Run: cap.py"
echo "  3. At boot you should see:"
echo "     ✓ RITnet ML pupil detection available (ncnn RITnet)"
echo ""
echo "Swap will remain at 1 GB. If you want to reduce it later, run:"
echo "  sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile"
echo "  sudo dphys-swapfile setup && sudo dphys-swapfile swapon"
