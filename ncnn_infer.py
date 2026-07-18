"""ncnn RITnet inference — an ARMv6/Pi-friendly drop-in for ritnet_infer.

Same public surface as ritnet_infer (``available()``, ``locate(...)``,
``tangent_crop_center``, ``RitnetResult``) but the forward pass runs through
**ncnn** instead of PyTorch, so it works on boards where torch has no build
(e.g. a Pi Zero W, ARMv6, Python 3.13).  All preprocessing and post-processing is
reused from ritnet_infer (importing it does NOT require torch — torch is loaded
lazily there), so there is exactly one pipeline; only the executor differs.

Model files come from export_ritnet_onnx.py + onnx2ncnn (see NCNN_PI_SETUP.md):
    ritnet.param  +  ritnet.bin
Resolved from, in order:
  1. ``$RITNET_NCNN``  (path to ritnet.param; the .bin is assumed beside it)
  2. ``ritnet.param`` next to this file         (drop it here on the Pi)
  3. ``models/ritnet/ritnet.param`` next to it  (the repo layout on the PC)

The torch ``Normalize([.5],[.5])`` (i.e. ``x/127.5 - 1``) is applied inside ncnn
via ``substract_mean_normalize([127.5], [1/127.5])`` on the uint8 crop, so the
numbers match the PyTorch path exactly.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Reuse ritnet_infer's arch-free helpers, dataclass and constants (no torch).
import ritnet_infer as _ri

NET_W, NET_H = _ri.NET_W, _ri.NET_H            # native crop (field of view)


def _env_size(default_w, default_h):
    v = os.environ.get("RITNET_NCNN_SIZE", "")
    try:
        w, h = (int(x) for x in v.lower().split("x"))
        return max(16, (w // 16) * 16), max(16, (h // 16) * 16)
    except (ValueError, AttributeError):
        return default_w, default_h


# Network forward size: the 640x400 crop (FOV) is resized to this before ncnn.
# Full 640x400 needs ~683 MB and OOM-kills a 512 MB Pi (exit 137); 320x192
# (multiple of 16) fits in RAM (~170 MB) and is ~4x faster.  Override with
# $RITNET_NCNN_SIZE=WxH (e.g. 640x400 for full-res on a machine with the RAM).
NCNN_NET_W, NCNN_NET_H = _env_size(320, 192)

PUPIL_CLASS = _ri.PUPIL_CLASS
RitnetResult = _ri.RitnetResult
tangent_crop_center = _ri.tangent_crop_center          # identical (pure)
inpaint_reflex = _ri.inpaint_reflex
crop_window = _ri.crop_window
normalize = _ri.normalize
_select_contour = _ri._select_contour
_CLASS_COLORS = _ri._CLASS_COLORS

# Blob names emitted by onnx2ncnn (we named them in export_ritnet_onnx.py).
# Auto-detected from the .param at load time; these are the fallbacks.
_INPUT_NAME = os.environ.get("RITNET_NCNN_INPUT", "input")
_OUTPUT_NAME = os.environ.get("RITNET_NCNN_OUTPUT", "logits")

_ncnn = None
_net = None
_load_error = None


# ───────────────────────── model resolution ───────────────────────────────
def resolve_model() -> str:
    """Path to ritnet.param (first that exists, else the next-to-script default)."""
    env = os.environ.get("RITNET_NCNN")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "ritnet.param"),
              os.path.join(here, "models", "ritnet", "ritnet.param")):
        if os.path.isfile(c):
            return c
    return os.path.join(here, "ritnet.param")


def _try_import_ncnn():
    global _ncnn, _load_error
    if _ncnn is not None:
        return _ncnn
    try:
        import ncnn
        _ncnn = ncnn
    except Exception as e:                       # noqa: BLE001
        _load_error = f"ncnn not installed ({e})"
    return _ncnn


def available() -> bool:
    """True if ncnn imports and the model files exist (cheap; no net build)."""
    ncnn = _try_import_ncnn()
    param = resolve_model()
    binp = param[:-6] + ".bin" if param.endswith(".param") else param + ".bin"
    ok = ncnn is not None and os.path.isfile(param) and os.path.isfile(binp)
    if not ok and _load_error is None and ncnn is not None:
        print(f"ncnn model not found (looked for {param} + .bin)")
    return ok


def _auto_names(param_path):
    """Best-effort read of the input + output blob names from the .param text.

    .param line:  <type> <name> <n_in> <n_out> <in_blobs...> <out_blobs...> ...
    The Input layer's output blob is the network input; the last layer's first
    output blob is the network output.  Falls back to the env/defaults on error.
    """
    global _INPUT_NAME, _OUTPUT_NAME
    try:
        with open(param_path) as f:
            layers = f.read().splitlines()[2:]   # skip magic + counts
        for ln in layers:
            t = ln.split()
            if len(t) >= 4 and t[0] == "Input":
                n_in, n_out = int(t[2]), int(t[3])
                outs = t[4 + n_in:4 + n_in + n_out]
                if outs:
                    _INPUT_NAME = outs[0]
                break
        for ln in reversed(layers):
            t = ln.split()
            if len(t) >= 4:
                n_in, n_out = int(t[2]), int(t[3])
                outs = t[4 + n_in:4 + n_in + n_out]
                if outs:
                    _OUTPUT_NAME = outs[0]
                    break
    except OSError:
        pass


def _load_net():
    """Load + cache the ncnn net, or set _load_error and return None."""
    global _net, _load_error
    if _net is not None:
        return _net
    ncnn = _try_import_ncnn()
    if ncnn is None:
        return None
    param = resolve_model()
    binp = param[:-6] + ".bin" if param.endswith(".param") else param + ".bin"
    if not (os.path.isfile(param) and os.path.isfile(binp)):
        _load_error = "ncnn model not found (ritnet.param / ritnet.bin)"
        return None
    try:
        _auto_names(param)
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        # Threads default to 1: ncnn's per-thread conv workspace multiplies memory,
        # and on a 512 MB Pi Zero 2 W even 2 threads at 320x192 OOMs/swap-thrashes
        # (1 thread = ~226 MB and leaves 3 cores free so wifi/UI stay responsive).
        # Override with $RITNET_NCNN_THREADS on a board with more RAM.
        try:
            net.opt.num_threads = max(1, int(os.environ.get("RITNET_NCNN_THREADS", "1")))
        except ValueError:
            net.opt.num_threads = 1
        net.opt.lightmode = True                 # recycle blob memory between layers
        net.load_param(param)
        net.load_model(binp)
        _net = net
    except Exception as e:                       # noqa: BLE001
        _load_error = f"failed to load ncnn model: {e}"
        _net = None
    return _net


def _forward(feed_u8, w, h):
    """Run ncnn on a uint8 h x w grayscale image; return logits (4, h, w) float32."""
    ncnn = _ncnn
    mat_in = ncnn.Mat.from_pixels(np.ascontiguousarray(feed_u8),
                                  ncnn.Mat.PixelType.PIXEL_GRAY, w, h)
    mat_in.substract_mean_normalize([127.5], [1.0 / 127.5])   # == x/127.5 - 1
    ex = _net.create_extractor()
    ex.input(_INPUT_NAME, mat_in)
    _, mat_out = ex.extract(_OUTPUT_NAME)
    return np.array(mat_out)                      # (C, h, w)


def _softmax0(logits):
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


# ───────────────────────────── public API ─────────────────────────────────
def locate(green, reflex_mask=None, anchor=None, crop_center=None,
           min_area=60, search_mask=None, want_stages=False) -> RitnetResult:
    """Segment the pupil with ncnn-RITnet and fit an ellipse.

    Mirrors ritnet_infer.locate exactly (same crop, normalisation, class indices
    and contour selection) — only the forward pass differs.  Never raises; check
    ``.ok``.
    """
    res = RitnetResult()
    net = _load_net()
    if net is None:
        res.ok = False
        res.error = _load_error or "ncnn unavailable"
        return res
    h, w = green.shape[:2]

    net_src = inpaint_reflex(green, reflex_mask)
    if crop_center is not None:
        ccx, ccy = int(crop_center[0]), int(crop_center[1])
    elif anchor is not None:
        ccx, ccy = int(anchor[0]), int(anchor[1])
    else:
        ccx, ccy = w // 2, h // 2
    crop, x0, y0, vh, vw = crop_window(net_src, ccx, ccy, NET_W, NET_H)
    net_in = normalize(crop)
    res.crop_rect = (x0, y0, vw, vh)
    if want_stages:
        res.stages.append(("1_net_input", cv2.cvtColor(net_in, cv2.COLOR_GRAY2BGR)))

    # Downscale the forward to fit memory — full 640x400 needs ~683 MB and
    # OOM-kills a 512 MB Pi (exit 137); 320x192 is ~170 MB and ~4x faster.
    feed = (net_in if (NCNN_NET_W, NCNN_NET_H) == (NET_W, NET_H)
            else cv2.resize(net_in, (NCNN_NET_W, NCNN_NET_H),
                            interpolation=cv2.INTER_AREA))
    logits = _forward(feed, NCNN_NET_W, NCNN_NET_H)   # (4, nH, nW)
    prob = _softmax0(logits)
    cls = prob.argmax(axis=0).astype(np.uint8)
    pupp = prob[PUPIL_CLASS]
    if want_stages:
        res.stages.append(("2_segmentation", _CLASS_COLORS[cls]))
    # Upscale class map + pupil prob to the 640x400 crop grid before mapping back.
    if (NCNN_NET_W, NCNN_NET_H) != (NET_W, NET_H):
        cls = cv2.resize(cls, (NET_W, NET_H), interpolation=cv2.INTER_NEAREST)
        pupp = cv2.resize(pupp, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)

    pupil_mask = np.zeros((h, w), np.uint8)
    pupil_mask[y0:y0 + vh, x0:x0 + vw] = \
        (cls[:vh, :vw] == PUPIL_CLASS).astype(np.uint8) * 255
    pupil_prob = np.zeros((h, w), np.float32)
    pupil_prob[y0:y0 + vh, x0:x0 + vw] = pupp[:vh, :vw]
    if search_mask is not None:
        pupil_mask = cv2.bitwise_and(pupil_mask, search_mask)
    res.pupil_mask = pupil_mask
    if want_stages:
        res.stages.append(("3_pupil_mask", cv2.cvtColor(pupil_mask, cv2.COLOR_GRAY2BGR)))

    final_vis = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(final_vis, (x0, y0), (x0 + vw, y0 + vh), (0, 255, 0), 2)
    cnts, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = [c for c in cnts if cv2.contourArea(c) >= min_area and len(c) >= 5]
    chosen = _select_contour(cand, anchor)
    if chosen is not None:
        ellipse = cv2.fitEllipse(chosen)
        (ex, ey), (MA, ma), _ = ellipse
        res.pupil = ellipse
        res.center = (float(ex), float(ey))
        res.radius = float((MA + ma) / 4.0)
        sel = np.zeros((h, w), np.uint8)
        cv2.drawContours(sel, [chosen], -1, 255, -1)
        pm = pupil_prob[sel > 0]
        res.confidence = float(pm.mean()) if pm.size else 0.0
        res.notes = f"pupil_area={int(cv2.contourArea(chosen))} of {len(cand)} cand"
        cv2.ellipse(final_vis, ellipse, (0, 0, 255), 2)
        cv2.circle(final_vis, (int(ex), int(ey)), 3, (0, 0, 255), -1)
    elif cnts:
        res.notes = "pupil blobs too small / unfittable"
    else:
        res.notes = "no pupil class predicted"
    if want_stages:
        res.stages.append(("4_final", final_vis))
    return res
