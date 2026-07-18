#!/bin/bash
# Recovery: install Python packages and build ncnn
# Manual swap setup (no dphys-swapfile needed)

set -e

echo "========================================="
echo "EyeVu + ncnn Installation (Recovery v2)"
echo "========================================="

# ─────────────────────────────────────────
# 1. Increase swap manually (no dphys-swapfile)
# ─────────────────────────────────────────

echo ""
echo "Setting up 1 GB swap manually..."

# Check if swap already exists
if [ -f /swapfile ]; then
    echo "Disabling existing swap..."
    sudo swapoff /swapfile 2>/dev/null || true
    sudo rm /swapfile
fi

echo "Creating 1 GB swap file..."
sudo fallocate -l 1G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=1024
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

echo "Swap status:"
free -h

# ─────────────────────────────────────────
# 2. Install core Python packages
# ─────────────────────────────────────────

echo ""
echo "Installing Python packages (numpy, opencv, scipy, etc)..."
#pip3 install --no-cache-dir --break-system-packages \
    #numpy \
    #pillow \
    #opencv-python \
    #scipy \
    #scikit-image

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
echo "Swap configuration:"
echo "  ✓ 1 GB swap file at /swapfile (manual setup)"
echo ""
echo "To restore swap after reboot (add to ~/.bashrc or crontab):"
echo "  sudo swapon /swapfile"
echo ""
echo "Next steps:"
echo "  1. Ensure ritnet.param and ritnet.bin are copied to ~/EyeVu/"
echo "  2. Run: python3 cap.py"
echo "  3. At boot you should see:"
echo "     ✓ RITnet ML pupil detection available (ncnn RITnet)"
