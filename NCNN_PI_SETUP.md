# Running RITnet on the Pi Zero W (ARMv6) via ncnn

PyTorch has no build for ARMv6 + modern Python, so on the Pi Zero W the RITnet
forward pass runs through **ncnn** instead of torch. Everything else (the crop,
normalisation, class handling, ellipse fit) is identical — `ncnn_infer.py` reuses
`ritnet_infer.py`'s helpers and only swaps the executor.

`cap.py` auto-selects the backend at boot:
`torch RITnet` if torch is importable (PC / aarch64), else `ncnn RITnet` if
`ncnn` + the model files are present, else ridge + red-eye only.

```
✓ RITnet ML pupil detection available (ncnn RITnet)
```

---

## Step 1 — The ncnn model is already built (PC) — just copy it

The conversion AND a full numeric check were done on the PC, so you do **not**
need `onnx2ncnn` on the Pi. `export_ritnet_onnx.py` produced the ONNX, then it was
converted to ncnn and validated against the torch model on real captures:

- **`models/ritnet/ritnet.param`** (8.6 KB)  ← graph
- **`models/ritnet/ritnet.bin`** (~500 KB)   ← weights

Verified: **12/12 pupil frames agree with torch within 2 px** (most < 0.6 px),
confidences identical. ncnn `.param`/`.bin` are **platform-independent**, so these
exact two files run on the Pi as-is. Copy just those two to the Pi.

(The `.onnx` / `-sim.onnx` files are intermediates; you don't need them on the Pi.)

## Step 2 — Build the ncnn *runtime* on the Pi (the slow part)

ARMv6 has no prebuilt ncnn wheel, so build the Python binding from source. You
only need the runtime — **not** the tools/`onnx2ncnn` (the model is already
converted). **On a Pi Zero this can take an hour-plus and needs swap.**

```bash
# 1 GB swap so the compiler doesn't OOM on the 512 MB Zero
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon

sudo apt update
sudo apt install -y build-essential cmake git python3-dev python3-pip python3-setuptools

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
make -j1            # -j1 on the Zero (low RAM); be patient
```

Notes:
- The Zero W is **ARMv6, no NEON** — ncnn builds scalar kernels; don't pass
  `-mfpu=neon`.
- If the build OOMs even with swap, there's nothing to parallelise at `-j1`; just
  let it run.

Install the Python binding (system Python 3.13 → PEP 668, so `--break-system-packages`):

```bash
cd ../python
pip install . --break-system-packages
python3 -c "import ncnn; print('ncnn', ncnn.__version__)"
```

*(If you'd rather convert on the Pi yourself, build with `-DNCNN_BUILD_TOOLS=ON`,
copy `ritnet-sim.onnx` over, and run
`./tools/onnx/onnx2ncnn ritnet-sim.onnx ritnet.param ritnet.bin`. Not needed —
Step 1's files already work.)*

## Step 3 — Deploy

Put the two model files **next to `cap.py`** on the Pi (same flat folder as
`cap.py` / `ncnn_infer.py` / `ritnet_infer.py`):

```
~/Desktop/Capture/
├── cap.py
├── ncnn_infer.py
├── ritnet_infer.py        # imported for its helpers (no torch needed)
├── ritnet.param           # <- from Step 3
└── ritnet.bin             # <- from Step 3
```

`ncnn_infer.resolve_model()` finds `ritnet.param` here automatically (or set
`$RITNET_NCNN` to its full path; the `.bin` is assumed beside it).

## Step 4 — Verify

Standalone timing/sanity check on a saved ambient frame (no camera needed):

```bash
python3 - <<'PY'
import time, cv2, ncnn_infer
print("available:", ncnn_infer.available())
g = cv2.imread("/tmp/eyevu_transfers/<a capture>/ambient.jpg", cv2.IMREAD_GRAYSCALE)
t = time.time(); r = ncnn_infer.locate(g); dt = time.time() - t
print(f"ok={r.ok} err={r.error!r} center={r.center} conf={r.confidence:.2f} time={dt:.1f}s")
PY
```

Then run `cap.py` — at boot it should print
`✓ RITnet ML pupil detection available (ncnn RITnet)`, and RITnet becomes the ML
rung of the capture-time `coarse_locate` / `detect_both` cascade.

## Performance reality — measure this before relying on it

This RITnet forward pass is **33 GFLOPs** per frame. For reference, on a fast
desktop x86 core ncnn runs it in ~**1.5 s**. A 1 GHz single-core **ARMv6 with no
NEON** is far weaker, so expect **roughly 1–5 minutes per inference** on the Zero
W, plus heavy memory pressure on 512 MB (hence the swap). It only runs **at
capture time**, never in live preview — but a multi-minute wait per capture may be
too slow to be useful for an alignment loop.

So treat Step 4's measured time as the deciding number. If it's too slow / OOMs:
- **Keep ridge + red-eye on the Pi** for live guidance (RITnet auto-skips; the
  cascade already works without it), or
- **Run RITnet at a smaller input** (e.g. a 320×192 crop) to cut compute ~4× —
  ask and I'll wire a downscale option into `ncnn_infer` and measure the
  speed/accuracy trade-off on the PC first, or
- **Move to a Pi Zero 2 W** (aarch64), where `pip install torch` works and a
  forward pass is seconds, not minutes.
