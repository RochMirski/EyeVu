#!/bin/bash
# Rollback/Recovery script for EyeVu + ncnn installation on Raspberry Pi Zero W
# Use only if the installation fails or you need to revert changes
#
# Usage: bash /home/roch/rollback_ncnn_install.sh

set -e

echo "========================================="
echo "EyeVu + ncnn Installation Rollback"
echo "========================================="
echo ""
echo "WARNING: This will revert installation changes."
echo "This includes:"
echo "  - Uninstalling ncnn Python binding"
echo "  - Resetting swap from 1 GB back to 100 MB"
echo "  - Removing ncnn build artifacts"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled."
    exit 0
fi

# ─────────────────────────────────────────
# 1. Uninstall ncnn Python binding
# ─────────────────────────────────────────

echo ""
echo "Uninstalling ncnn Python binding..."
pip3 uninstall ncnn -y --break-system-packages || true

# ─────────────────────────────────────────
# 2. Reset swap to original size (100 MB)
# ─────────────────────────────────────────

echo ""
echo "Resetting swap to 100 MB (from 1 GB)..."
sudo dphys-swapfile swapoff || true
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
echo "✓ Swap reset"

# ─────────────────────────────────────────
# 3. Clean up build artifacts
# ─────────────────────────────────────────

echo ""
echo "Cleaning up ncnn build artifacts..."
rm -rf /tmp/ncnn 2>/dev/null || true
echo "✓ Removed /tmp/ncnn"

# ─────────────────────────────────────────
# 4. Optional: Remove build tools (optional)
# ─────────────────────────────────────────

echo ""
read -p "Remove build tools (cmake, build-essential)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing build tools..."
    sudo apt-get remove -y build-essential cmake || true
    sudo apt-get autoremove -y || true
    echo "✓ Build tools removed"
fi

# ─────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────

echo ""
echo "========================================="
echo "Rollback Complete"
echo "========================================="
echo ""
echo "Reverted:"
echo "  ✓ ncnn Python binding uninstalled"
echo "  ✓ Swap reset to 100 MB"
echo "  ✓ Build artifacts cleaned"
echo ""
echo "Remaining installed:"
echo "  - System dependencies (opencv, numpy, scipy, etc)"
echo "  - picamera2, RPi.GPIO"
echo ""
echo "To fully uninstall everything Python-related:"
echo "  sudo apt-get remove -y python3-dev python3-pip"
echo "  pip3 uninstall numpy pillow opencv-python scipy scikit-image -y"
echo ""
echo "Current swap size:"
free -h
