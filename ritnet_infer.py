"""Shared RITnet inference — pupil segmentation, usable on the PC and the Pi.

Self-contained (numpy + cv2 + torch only; no cap.py / pupillab imports) so it can
be dropped next to cap.py on the Pi and imported by the cap.py guidance cascade,
the pupillab `ritnet` detector, and receiver.py alike — one implementation, no
drift.

Pipeline: take a (cover-filled) green channel, strongly inpaint the corneal
reflex, crop a NATIVE 640x400 window centred on the pupil (no resize -> no aspect
squash), run RITnet, and fit an ellipse to the pupil class nearest the reflex
anchor.  Everything is guarded: if torch or the weights are missing, `locate()`
returns a result with ``ok=False`` and an explanatory ``error`` instead of raising.

Weights are resolved from, in order:
  1. ``$RITNET_WEIGHTS``
  2. ``best_model.pkl`` next to this file        (drop it here on the Pi)
  3. ``models/ritnet/best_model.pkl`` next to it (the repo layout on the PC)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# RITnet input geometry and class indices (OpenEDS convention).
NET_W, NET_H = 640, 400
PUPIL_CLASS = 3
# Stronger-than-cap reflex inpainting (cap defaults dilate=7, radius=5): remove
# the bright spot + halo aggressively so it can't split the pupil segmentation.
_REFLEX_DILATE = 21
_REFLEX_INPAINT_RADIUS = 15
_CLASS_COLORS = np.array([[0, 0, 0],        # 0 background
                          [60, 60, 60],     # 1 sclera
                          [0, 140, 255],    # 2 iris   (orange, BGR)
                          [0, 0, 255]],     # 3 pupil  (red, BGR)
                         dtype=np.uint8)


# ───────────────────────── weights resolution ─────────────────────────────
def resolve_weights() -> str:
    """Return the RITnet weights path (first that exists, else the default)."""
    env = os.environ.get("RITNET_WEIGHTS")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"Resolving RITnet weights: looking for $RITNET_WEIGHTS, then {here}/best_model.pkl, then {here}/models/ritnet/best_model.pkl")
    candidates = [
        os.path.join(here, "best_model.pkl"),                      # next to script
        os.path.join(here, "models", "ritnet", "best_model.pkl"),  # repo layout
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


# ───────────────────────── model architecture ─────────────────────────────
# RITnet DenseNet2D (Chaudhary et al., ICCVW 2019); layer names match the
# upstream `densenet.py` so the official best_model.pkl state dict loads as-is.
# Source: https://github.com/AayushKrChaudhary/RITnet (MIT).  Built lazily inside
# _load_model so importing this file never requires torch.
def _build_densenet(torch):
    nn = torch.nn

    class _DownBlock(nn.Module):
        def __init__(self, in_ch, out_ch, down_size, dropout=False, prob=0):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1))
            self.conv21 = nn.Conv2d(in_ch + out_ch, out_ch, (1, 1), padding=(0, 0))
            self.conv22 = nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1))
            self.conv31 = nn.Conv2d(in_ch + 2 * out_ch, out_ch, (1, 1), padding=(0, 0))
            self.conv32 = nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1))
            self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else None
            self.relu = nn.LeakyReLU()
            self.down_size = down_size
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)
            self.dropout3 = nn.Dropout(p=prob)
            self.bn = nn.BatchNorm2d(num_features=out_ch)

        def forward(self, x):
            if self.down_size is not None:
                x = self.max_pool(x)
            if self.dropout:
                x1 = self.relu(self.dropout1(self.conv1(x)))
                x21 = torch.cat((x, x1), dim=1)
                x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
                x31 = torch.cat((x21, x22), dim=1)
                out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
            else:
                x1 = self.relu(self.conv1(x))
                x21 = torch.cat((x, x1), dim=1)
                x22 = self.relu(self.conv22(self.conv21(x21)))
                x31 = torch.cat((x21, x22), dim=1)
                out = self.relu(self.conv32(self.conv31(x31)))
            return self.bn(out)

    class _UpBlock(nn.Module):
        def __init__(self, skip_ch, in_ch, out_ch, up_stride, dropout=False, prob=0):
            super().__init__()
            self.conv11 = nn.Conv2d(skip_ch + in_ch, out_ch, (1, 1), padding=(0, 0))
            self.conv12 = nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1))
            self.conv21 = nn.Conv2d(skip_ch + in_ch + out_ch, out_ch, (1, 1), padding=(0, 0))
            self.conv22 = nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1))
            self.relu = nn.LeakyReLU()
            self.up_stride = up_stride
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)

        def forward(self, prev_feature_map, x):
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode="nearest")
            x = torch.cat((x, prev_feature_map), dim=1)
            if self.dropout:
                x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
                x21 = torch.cat((x, x1), dim=1)
                out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            else:
                x1 = self.relu(self.conv12(self.conv11(x)))
                x21 = torch.cat((x, x1), dim=1)
                out = self.relu(self.conv22(self.conv21(x21)))
            return out

    class DenseNet2D(nn.Module):
        def __init__(self, in_channels=1, out_channels=4, channel_size=32,
                     concat=True, dropout=False, prob=0):
            super().__init__()
            cs = channel_size
            self.down_block1 = _DownBlock(in_channels, cs, None, dropout, prob)
            self.down_block2 = _DownBlock(cs, cs, (2, 2), dropout, prob)
            self.down_block3 = _DownBlock(cs, cs, (2, 2), dropout, prob)
            self.down_block4 = _DownBlock(cs, cs, (2, 2), dropout, prob)
            self.down_block5 = _DownBlock(cs, cs, (2, 2), dropout, prob)
            self.up_block1 = _UpBlock(cs, cs, cs, (2, 2), dropout, prob)
            self.up_block2 = _UpBlock(cs, cs, cs, (2, 2), dropout, prob)
            self.up_block3 = _UpBlock(cs, cs, cs, (2, 2), dropout, prob)
            self.up_block4 = _UpBlock(cs, cs, cs, (2, 2), dropout, prob)
            self.out_conv1 = nn.Conv2d(cs, out_channels, kernel_size=1, padding=0)
            self.concat = concat
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)

        def forward(self, x):
            self.x1 = self.down_block1(x)
            self.x2 = self.down_block2(self.x1)
            self.x3 = self.down_block3(self.x2)
            self.x4 = self.down_block4(self.x3)
            self.x5 = self.down_block5(self.x4)
            self.x6 = self.up_block1(self.x4, self.x5)
            self.x7 = self.up_block2(self.x3, self.x6)
            self.x8 = self.up_block3(self.x2, self.x7)
            self.x9 = self.up_block4(self.x1, self.x8)
            out = self.out_conv1(self.dropout1(self.x9)) if self.dropout \
                else self.out_conv1(self.x9)
            return out

    return DenseNet2D


# ───────────────────────── lazy torch + model ─────────────────────────────
_torch = None
_model = None
_load_error = None


def _try_import_torch():
    global _torch, _load_error
    if _torch is not None:
        return _torch
    # OpenMP coexistence: numpy/MKL loads Intel's libiomp5md.dll before torch
    # imports LLVM's libomp.dll, which otherwise aborts ("OMP: Error #15") or
    # surfaces as shm.dll WinError 127 on Windows.  Allow the duplicate runtime
    # and keep torch single-threaded so the two cannot race.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch
        torch.set_num_threads(1)
        _torch = torch
    except ModuleNotFoundError as e:
        _load_error = f"PyTorch not installed ({e}). pip install torch"
        print(_load_error)
    except Exception as e:                       # noqa: BLE001  (e.g. DLL load error)
        _load_error = (f"PyTorch failed to load ({e}). Restart the process; if it "
                       "persists, reinstall CPU torch.")
        print(_load_error)
    return _torch


def available() -> bool:
    """True if torch loads and the weights file exists (cheap, no model build)."""
    x = _try_import_torch()  # populates _load_error if torch is missing or broken
    y = os.path.isfile(resolve_weights())
    print(f"RITnet availability check: torch {'OK' if x else 'MISSING'}; "
          f"weights {'OK' if y else 'MISSING'}")
    return x is not None and y


def _load_model():
    """Load + cache the RITnet model, or set _load_error and return None."""
    global _model, _load_error
    if _model is not None:
        return _model
    torch = _try_import_torch()
    if torch is None:
        return None
    weights = resolve_weights()
    if not os.path.isfile(weights):
        _load_error = (f"weights not found (looked for $RITNET_WEIGHTS, "
                       f"best_model.pkl next to ritnet_infer.py, and "
                       f"models/ritnet/best_model.pkl)")
        return None
    try:
        model = _build_densenet(torch)(in_channels=1, out_channels=4,
                                       channel_size=32, dropout=False, prob=0)
        ckpt = torch.load(weights, map_location="cpu")
        state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        _model = model
    except Exception as e:                       # noqa: BLE001
        _load_error = f"failed to load RITnet weights: {e}"
        _model = None
    return _model


# ───────────────────────── preprocessing helpers ──────────────────────────
def normalize(green):
    """RITnet intensity normalisation: gamma -> CLAHE (no resize)."""
    table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
    img = cv2.LUT(green, table)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(img)


def crop_window(img, cx, cy, out_w, out_h):
    """Crop a native out_w x out_h window centred on (cx, cy), clamped to `img`.

    Returns (crop, x0, y0, vh, vw); zero-padded only if the image is smaller."""
    h, w = img.shape[:2]
    x0 = int(np.clip(cx - out_w // 2, 0, max(0, w - out_w)))
    y0 = int(np.clip(cy - out_h // 2, 0, max(0, h - out_h)))
    crop = img[y0:y0 + out_h, x0:x0 + out_w]
    vh, vw = crop.shape[:2]
    if (vh, vw) != (out_h, out_w):
        crop = cv2.copyMakeBorder(crop, 0, out_h - vh, 0, out_w - vw,
                                  cv2.BORDER_CONSTANT, value=0)
    return crop, x0, y0, vh, vw


def tangent_crop_center(cover_mask, frame_shape, out_w=NET_W, out_h=NET_H):
    """Crop centre placing the out_w x out_h window TANGENT to the LED cover.

    The window sits just past the cover's inner edge, on the side away from the
    frame border the cover intrudes from (detected from the mask): below a cover
    in the top half, above a cover in the bottom half.  It is centred on that
    edge's middle x.  Falls back to the image centre with no cover mask.
    (crop_window clamps the result to the frame.)
    """
    h, w = frame_shape[:2]
    if cover_mask is None or int(cv2.countNonZero(cover_mask)) == 0:
        return (w // 2, h // 2)
    ys, xs = np.where(cover_mask > 0)
    top = int(ys.min()); bot = int(ys.max())
    if (top + bot) * 0.5 < h * 0.5:        # cover TOP half -> pupil below; window under cover
        band = ys >= bot - 0.10 * (bot - top + 1)
        cy = bot + out_h // 2              # so y0 == cover bottom (top-tangent)
    else:                                  # cover BOTTOM half -> pupil above; window over cover
        band = ys <= top + 0.10 * (bot - top + 1)
        cy = top - out_h // 2             # so y0 + out_h == cover top (bottom-tangent)
    cx = int(round(xs[band].mean())) if np.any(band) else int(round(xs.mean()))
    return (cx, cy)


def inpaint_reflex(green, reflex_mask):
    """Strongly inpaint the corneal reflex blob out of `green` (copy)."""
    if reflex_mask is None:
        return green
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_REFLEX_DILATE, _REFLEX_DILATE))
    m = cv2.dilate(reflex_mask, k)
    return cv2.inpaint(green, m, _REFLEX_INPAINT_RADIUS, cv2.INPAINT_TELEA)


def _select_contour(contours, anchor):
    """Pupil contour closest to the reflex anchor (else largest); or None."""
    if not contours:
        return None
    if anchor is None:
        return max(contours, key=cv2.contourArea)
    ax, ay = anchor[0], anchor[1]

    def dist2(c):
        m = cv2.moments(c)
        if m["m00"] == 0:
            return float("inf")
        return (m["m10"] / m["m00"] - ax) ** 2 + (m["m01"] / m["m00"] - ay) ** 2

    return min(contours, key=dist2)


# ───────────────────────────── public API ─────────────────────────────────
@dataclass
class RitnetResult:
    ok: bool = True
    error: str = ""
    pupil: Optional[tuple] = None          # OpenCV ellipse ((cx,cy),(MA,ma),ang)
    center: Optional[tuple] = None         # (x, y)
    radius: Optional[float] = None
    confidence: float = 0.0
    pupil_mask: Optional[np.ndarray] = None
    crop_rect: Optional[tuple] = None      # (x0, y0, vw, vh)
    notes: str = ""
    stages: list = field(default_factory=list)   # list[(name, bgr)]


def locate(green, reflex_mask=None, anchor=None, crop_center=None,
           min_area=60, search_mask=None, want_stages=False) -> RitnetResult:
    """Segment the pupil with RITnet and fit an ellipse.

    `green` is a (cover-filled) grayscale frame in display orientation.  The crop
    is centred on `crop_center` (else `anchor`, else image centre).  `search_mask`
    (optional) drops predictions outside the allowed region.  Set `want_stages`
    to also return debug images.  Never raises — check `.ok`.
    """
    res = RitnetResult()
    model = _load_model()
    if model is None:
        res.ok = False
        res.error = _load_error or "RITnet unavailable"
        return res
    torch = _torch
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

    x = (net_in.astype(np.float32) / 127.5) - 1.0      # == Normalize([.5],[.5])
    x = torch.from_numpy(x)[None, None, :, :]
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    cls = prob.argmax(axis=0).astype(np.uint8)
    if want_stages:
        res.stages.append(("2_segmentation", _CLASS_COLORS[cls]))

    pupil_mask = np.zeros((h, w), np.uint8)
    pupil_mask[y0:y0 + vh, x0:x0 + vw] = \
        (cls[:vh, :vw] == PUPIL_CLASS).astype(np.uint8) * 255
    pupil_prob = np.zeros((h, w), np.float32)
    pupil_prob[y0:y0 + vh, x0:x0 + vw] = prob[PUPIL_CLASS][:vh, :vw]
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
