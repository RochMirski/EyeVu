# Installation Changes & Rollback Guide

## What Was Installed

If the installation completes successfully, the following changes were made to the Raspberry Pi:

### System Packages (via `apt-get`)
- **build-essential** — C/C++ compiler and build tools
- **cmake** — Build system for ncnn
- **git** — Version control (used to clone ncnn)
- **python3-dev** — Python development headers
- **python3-pip** — Python package manager
- **python3-setuptools** — Package setup utilities
- **OpenCV libraries** (libopenjp2-7, libtiff, libjasper, libharfbuzz0b, libwebp6, libatlas-base-dev, libatomic1)
- **python3-picamera2** — Raspberry Pi camera module
- **python3-rpi.gpio** — GPIO control for LEDs

### Python Packages (via `pip3`)
- **numpy** — Numerical computing
- **pillow** — Image processing
- **opencv-python** — Computer vision
- **scipy** — Scientific computing
- **scikit-image** — Image processing algorithms
- **ncnn** — Neural network inference engine (Python binding)

### System Configuration Changes
- **Swap increased** from 100 MB → 1 GB (in `/etc/dphys-swapfile`)
  - Necessary for ncnn compilation on 512 MB RAM Pi Zero

### Source Code Built
- **ncnn** (Tencent) cloned and compiled from source in `/tmp/ncnn/`
  - Time: ~45 minutes on Pi Zero
  - Used for RITnet ML pupil detection inference

## When to Rollback

Revert the installation if:
1. ✗ The build failed partway through
2. ✗ The Pi is running low on disk space
3. ✗ You want to test without ML (ridge + red-eye only)
4. ✗ You need to free resources
5. ✗ ncnn performance is too slow

## How to Rollback

### Option 1: Use the Rollback Script (Easiest)

From your **Windows machine**, copy and run the rollback script:

```powershell
scp "rollback_ncnn_install.sh" roch@192.168.137.133:/home/roch/

ssh roch@192.168.137.133 "bash /home/roch/rollback_ncnn_install.sh"
```

This will:
1. Uninstall the ncnn Python binding
2. Reset swap from 1 GB → 100 MB
3. Remove build artifacts (`/tmp/ncnn`)
4. Optionally remove build tools (cmake, build-essential)

### Option 2: Manual Rollback

SSH into the Pi and run commands individually:

```bash
ssh roch@192.168.137.133

# Uninstall ncnn
pip3 uninstall ncnn -y --break-system-packages

# Reset swap
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Clean up ncnn build directory
rm -rf /tmp/ncnn

# Optional: remove build tools to free space
sudo apt-get remove -y build-essential cmake git
sudo apt-get autoremove -y
```

### Option 3: Keep Only Core Dependencies

If you want to keep the base Python packages (numpy, opencv, picamera2) but remove ML stuff:

```bash
pip3 uninstall ncnn -y --break-system-packages
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
rm -rf /tmp/ncnn
sudo apt-get remove -y build-essential cmake
```

This leaves you with a functional Pi that can run `cap.py` with **ridge + red-eye pupil detection only** (no ML).

## Verifying Rollback

After rollback, verify the changes:

```bash
# Check ncnn is gone
python3 -c "import ncnn; print('ERROR: ncnn still present')" 2>&1 | grep -q "ModuleNotFoundError" && echo "✓ ncnn uninstalled"

# Check swap is back to 100 MB
free -h | grep Swap

# Check disk space recovered
df -h /tmp

# Boot cap.py and verify fallback to classical detection
python3 cap.py  # Should print "✗ RITnet ML pupil detection NOT available (ridge + red-eye only)"
```

## Disk Space Reference

Approximate sizes of installed components:

| Component | Size |
|-----------|------|
| ncnn source + build | ~500 MB (`/tmp/ncnn/`) |
| ncnn Python binding | ~50 MB |
| Build tools (cmake, build-essential) | ~200 MB |
| Python + opencv-python | ~300 MB |
| **Total** | **~1 GB** |

Removing ncnn (`pip uninstall`) frees ~50 MB.
Removing build tools frees ~200 MB.
Removing `/tmp/ncnn` frees ~500 MB.

## If Something Goes Wrong

If rollback itself fails:

```bash
# Force-clean swap config
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo systemctl restart dphys-swapfile

# Manual pip clean
pip3 cache purge
pip3 uninstall -y ncnn

# Check what was installed
pip3 list | grep -i ncnn
```

## Reverting to Factory Reset (Nuclear Option)

If you need a complete factory reset:

```bash
# Backup important files first!
cp -r ~/EyeVu ~/EyeVu.backup

# Then reflash Raspberry Pi OS from your PC
# (instructions: https://www.raspberrypi.com/software/)
```

---

**Files involved:**
- Rollback script: `rollback_ncnn_install.sh`
- Installation script: `install_pi_ncnn.sh`
- Models (not touched): `models/ritnet/ritnet.param` and `ritnet.bin`
