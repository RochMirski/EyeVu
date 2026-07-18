#!/bin/bash
# Recovery: install Python packages and build ncnn
# (skips system package install that's already failing)

set -e

echo "========================================="
echo "EyeVu + ncnn Installation (Recovery)"
echo "========================================="

# ─────────────────────────────────────────
# 1. Increase swap for ncnn compilation
# ─────────────────────────────────────────

echo ""
echo "Setting up 1 GB swap for ncnn build..."
sudo dphys-swapfile swapoff || true
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# ─────────────────────────────────────────
# 2. Install core Python packages
# ─────────────────────────────────────────

echo ""
echo "Installing Python packages (numpy, opencv, scipy, etc)..."
pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pillow \
    opencv-python \
    scipy \
    scikit-image

# ─────────────────────────────────────────
# 3. Build ncnn runtime from source
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
# 4. Install Python binding
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
# 5. Summary
# ─────────────────────────────────────────

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Core dependencies installed:"
echo "  ✓ numpy, pillow, opencv-python, scipy, scikit-image"
echo "  ✓ ncnn (RITnet ML inference)"
echo ""
echo "Next steps:"
echo "  1. Ensure ritnet.param and ritnet.bin are copied to ~/EyeVu/"
echo "  2. Run: python3 cap.py"
echo "  3. At boot you should see:"
echo "     ✓ RITnet ML pupil detection available (ncnn RITnet)"
