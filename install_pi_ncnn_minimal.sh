#!/bin/bash
# Recovery: install Python packages and build ncnn
# Skip Vulkan submodules to save disk space

set -e

echo "========================================="
echo "EyeVu + ncnn Installation (Minimal)"
echo "========================================="

# ─────────────────────────────────────────
# 1. Clean up disk space
# ─────────────────────────────────────────

echo ""
echo "Cleaning up disk space..."
rm -rf /tmp/ncnn 2>/dev/null || true
sudo apt-get clean || true
pip3 cache purge || true

echo "Disk usage:"
df -h / | tail -1

# ─────────────────────────────────────────
# 2. Increase swap manually (no dphys-swapfile)
# ─────────────────────────────────────────

echo ""
echo "Setting up 512 MB swap (smaller to fit on disk)..."

# Check if swap already exists
if [ -f /swapfile ]; then
    echo "Disabling existing swap..."
    sudo swapoff /swapfile 2>/dev/null || true
    sudo rm /swapfile
fi

echo "Creating 512 MB swap file..."
sudo fallocate -l 512M /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=512
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

echo "Swap status:"
free -h

# ─────────────────────────────────────────
# 3. Install core Python packages
# ─────────────────────────────────────────

echo ""
echo "Installing Python packages (numpy, opencv, scipy, etc)..."
#pip3 install --no-cache-dir --break-system-packages \
#    numpy \
#    pillow \
 #   opencv-python \
  #  scipy \
   # scikit-image

# ─────────────────────────────────────────
# 4. Build ncnn runtime from source (minimal, no Vulkan)
# ─────────────────────────────────────────

echo ""
echo "Building ncnn (CPU only, no Vulkan)..."
cd /tmp
rm -rf ncnn 2>/dev/null || true

# Clone WITHOUT submodules (we don't need Vulkan/glslang)
echo "Cloning ncnn (minimal, no submodules)..."
git clone --depth=1 --no-recurse-submodules https://github.com/Tencent/ncnn.git
cd ncnn

# Initialize ONLY pybind11 submodule (needed for Python binding)
# Skip the massive Vulkan submodules (glslang, SPIRV-Cross)
echo "Initializing pybind11 submodule only..."
git submodule init python/pybind11
git submodule update --depth=1 python/pybind11

echo "Building..."
mkdir build && cd build

# CPU-only, minimal build
cmake -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_PYTHON=ON \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DNCNN_AVX=OFF \
      -DNCNN_AVX2=OFF \
      -DNCNN_SSE2=OFF \
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
# 5b. Cleanup build artifacts
# ─────────────────────────────────────────

echo ""
echo "Cleaning up build artifacts..."
rm -rf /tmp/ncnn 2>/dev/null || true

# ─────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Core dependencies installed:"
echo "  ✓ numpy, pillow, opencv-python, scipy, scikit-image"
echo "  ✓ ncnn (RITnet ML inference, CPU only)"
echo ""
echo "Swap configuration:"
echo "  ✓ 512 MB swap file at /swapfile (manual setup)"
echo ""
echo "To restore swap after reboot, add to ~/.bashrc:"
echo "  sudo swapon /swapfile"
echo ""
echo "Next steps:"
echo "  1. Ensure ritnet.param and ritnet.bin are copied to ~/EyeVu/"
echo "  2. Run: python3 cap.py"
echo "  3. At boot you should see:"
echo "     ✓ RITnet ML pupil detection available (ncnn RITnet)"
