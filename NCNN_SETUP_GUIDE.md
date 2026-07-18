# NCNN Setup for Raspberry Pi Zero W

This guide walks through installing ncnn + RITnet ML pupil detection on the Raspberry Pi Zero W (ARMv6).

## Overview

PyTorch doesn't have a build for ARMv6 (Pi Zero W), so we use **ncnn** for inference instead:
- **ritnet.param** + **ritnet.bin** (model files) — already built on PC, just copy them
- **ncnn runtime** (Python binding) — compile on the Pi (takes 30-60 min on Zero)

After setup, `cap.py` auto-detects ncnn availability and enables RITnet. The cascade falls back gracefully to ridge + red-eye detection if ncnn is missing.

## Prerequisites

- Raspberry Pi Zero W with Raspberry Pi OS (Bullseye or later)
- SSH access (see `piconnection.md`)
- **~1 GB free disk space** on the Pi
- **Patience** — compilation on Pi Zero takes time

## Step 1: Copy Model Files from PC

On your **Windows machine**, run:

```powershell
cd "c:\Users\zmirs\OneDrive - University of Cambridge\University Work\IIB\EyeVu Container\EyeVu\Software\EyeVu"
.\copy_models_to_pi.ps1 -RemoteHost 192.168.137.133 -RemoteUser roch
```

This copies:
- `models/ritnet/ritnet.param` (8.6 KB)
- `models/ritnet/ritnet.bin` (~500 KB)

to the Pi at `~/EyeVu/`.

## Step 2: Copy Installation Script to Pi

From your **Windows machine**:

```powershell
scp ".\install_pi_ncnn.sh" roch@192.168.137.133:/home/roch/
```

(or transfer manually via SFTP)

## Step 3: Run Installation on the Pi

SSH into the Pi:

```bash
ssh roch@192.168.137.133
# enter password: 1111
```

Run the installation script:

```bash
bash /home/roch/install_pi_ncnn.sh
```

This will:
1. Install system dependencies (build tools, OpenCV, etc.)
2. Increase swap from 100 MB → 1 GB (needed for ncnn compilation)
3. Clone and build ncnn from source (~45 min on Pi Zero)
4. Install the Python ncnn binding
5. Verify the installation

**Monitor output** — if the process gets stuck, check disk space / swap availability.

## Step 4: Copy Code Files to Pi

After installation completes, copy the Python scripts:

```powershell
# From your Windows machine
scp "ritnet_infer.py" roch@192.168.137.133:/home/roch/EyeVu/
scp "ncnn_infer.py" roch@192.168.137.133:/home/roch/EyeVu/
scp "cap.py" roch@192.168.137.133:/home/roch/EyeVu/
```

Alternatively, place them manually in `~/EyeVu/` on the Pi alongside the model files.

Final folder structure on Pi:

```
~/EyeVu/
├── cap.py
├── ncnn_infer.py
├── ritnet_infer.py
├── ritnet.param           (from Step 1)
└── ritnet.bin             (from Step 1)
```

## Step 5: Verify

On the Pi, test ncnn:

```bash
cd ~/EyeVu
python3 -c "import ncnn_infer; print('available:', ncnn_infer.available())"
```

Should print: `available: True`

Then test the full inference:

```bash
python3 -c "
import ncnn_infer
from PIL import Image
import numpy as np

# Create a dummy image
img = np.ones((960, 1280), dtype=np.uint8) * 128
result = ncnn_infer.locate(img)
print(f'Result: ok={result.ok}, error={result.error!r}')
"
```

Finally, run `cap.py`:

```bash
python3 cap.py
```

At boot, you should see:
```
✓ RITnet ML pupil detection available (ncnn RITnet)
```

## Troubleshooting

### ncnn build fails with "Memory exhausted"

The Pi Zero only has 512 MB RAM. The script increases swap to 1 GB, which should be enough, but:
- Check available disk space: `df -h`
- Ensure swap is active: `free -h`
- Try the build again (if it partially completed, `make -j1` in `ncnn/build` will resume)

### "Module ncnn not found" after build

Ensure you ran `pip3 install . --break-system-packages` in `ncnn/python/`.

### ritnet.param / ritnet.bin not found

Check the file is copied to the same folder as `cap.py`:
```bash
ls -la ~/EyeVu/ritnet.*
```

Set the `$RITNET_NCNN` environment variable if they're elsewhere:
```bash
export RITNET_NCNN=/path/to/ritnet.param
python3 cap.py
```

### Very slow inference (5+ minutes per frame)

RITnet is **33 GFLOPs per frame**. On ARMv6 (Pi Zero) without NEON, expect **1–5 minutes per inference**.  

If this is too slow:
- Use cap.py's ridge + red-eye guidance (no ML wait)
- Only run RITnet on high-confidence captures (reduce frequency)
- Upgrade to a **Pi Zero 2 W** (aarch64, where PyTorch works and inference is ~5 sec)

## Reverting Swap

If you want to return swap to its original size (100 MB):

```bash
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
free -h
```

## References

- NCNN: https://github.com/Tencent/ncnn
- RITnet: https://github.com/AayushKrChaudhary/RITnet
- Setup doc: `NCNN_PI_SETUP.md`
