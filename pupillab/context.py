"""Build the shared DetectionContext for one capture.

This reproduces exactly the preprocessing that ``cap.detect_and_annotate`` does
(rotation to display orientation, channel split, calibrated cover mask, corneal
reflex anchor) — reusing ``cap``'s own helpers so there is a single source of
truth.  The result is computed once and shared by every detector module.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ── Preload PyTorch before OpenCV ─────────────────────────────────────────
# On Windows, once OpenCV (imported and *used* by cap.py) initialises its
# OpenMP runtime, torch's shm.dll later fails to load (WinError 127) and the
# RITnet module becomes unavailable.  Importing torch here — before `import cap`
# pulls in cv2 — lets torch's DLLs load first.  Guarded: a missing/broken torch
# is a silent no-op (the RITnet module reports it later) and the classical
# detectors never need torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch as _torch_preload
    _torch_preload.set_num_threads(1)
except Exception:       # noqa: BLE001
    pass

import numpy as np
from PIL import Image

# Make the repo root (which holds cap.py) importable regardless of CWD, so this
# works under `streamlit run pupillab/app.py`, `python pupillab/run_batch.py`,
# and plain `import pupillab` alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cap  # noqa: E402  (cap.py guards its picamera2 / RPi.GPIO imports)

from .base import DetectionContext  # noqa: E402

TRANSFERS_DIR = os.path.join(_REPO_ROOT, "Transfers")

# ── Search ROI defaults ───────────────────────────────────────────────────
# The pupil sits in a box just PAST the LED cover's inner edge, on the side of the
# cover away from the frame border it intrudes from.  The box is a square whose
# side spans the image's smaller dimension (scaled by SIDE_FRAC), centred
# horizontally on the cover and offset OFFSET_FRAC*side past the cover's inner
# edge.  Which edge is "inner" is detected from the mask (cover in the top half ->
# pupil below it; cover in the bottom half -> pupil above it), so the box follows
# the cover whichever way the rig / eye is oriented.  The cover region is then
# subtracted from the box.
ROI_ENABLE = True
ROI_SIDE_FRAC = 1.0     # box side as a fraction of min(H, W)
ROI_OFFSET_FRAC = 0.05  # how far past the cover's inner edge to centre the box (×side)

# LED-cover fill: the cover region is inpainted then its (feathered) boundary is
# smoothed, so detectors see a soft fill instead of a hard black/edge.  Shared by
# every detector (classical + RITnet).
COVER_INPAINT_RADIUS = 15   # cv2.inpaint TELEA neighbourhood radius
COVER_SMOOTH = 31           # Gaussian kernel for the feathered transition (odd)


def _rotate(bgr, rot):
    return cap.cv2.rotate(bgr, rot) if rot is not None else bgr


def compute_search_roi(cover_mask, shape, side_frac=ROI_SIDE_FRAC,
                       offset_frac=ROI_OFFSET_FRAC):
    """Box (x0, y0, x1, y1) just past the LED cover's inner edge, or None.

    The cover side is detected from the mask, so the box sits below a top cover
    and above a bottom cover.  Needs the calibrated cover mask to anchor the box;
    returns None when there is no cover mask (so detection falls back to the full
    frame).
    """
    cv2 = cap.cv2
    h, w = shape[:2]
    if cover_mask is None or int(cv2.countNonZero(cover_mask)) == 0:
        return None
    ys, xs = np.where(cover_mask > 0)
    cover_top = int(ys.min()); cover_bot = int(ys.max())
    cx = int(round(xs.mean()))         # cover horizontal centre
    side = int(round(min(h, w) * side_frac))
    half = side // 2
    if (cover_top + cover_bot) * 0.5 < h * 0.5:        # cover in TOP half -> pupil below
        cy = int(round(cover_bot + offset_frac * side))
    else:                                              # cover in BOTTOM half -> pupil above
        cy = int(round(cover_top - offset_frac * side))
    x0 = max(0, cx - half); x1 = min(w, cx + half)
    y0 = max(0, cy - half); y1 = min(h, cy + half)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _roi_search_mask(roi, cover_mask, shape):
    """uint8 mask (255 = searchable) = the ROI box minus the known cover."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = roi
    mask[y0:y1, x0:x1] = 255
    if cover_mask is not None:
        mask[cover_mask > 0] = 0       # never search the known-occluded cover
    return mask


def fill_cover(img, cover_mask, radius=COVER_INPAINT_RADIUS, smooth=COVER_SMOOTH):
    """Inpaint the LED-cover region and smooth its transition into the image.

    `img` is grayscale or BGR.  The cover is inpainted (TELEA) so the near-black
    occluder is replaced by surrounding eye texture; then the fill and a feathered
    halo around it are Gaussian-blurred and alpha-blended back, so the seam fades
    in gradually rather than appearing as a sharp patch.  No-op without a cover.
    """
    cv2 = cap.cv2
    if cover_mask is None or int(cv2.countNonZero(cover_mask)) == 0:
        return img
    filled = cv2.inpaint(img, cover_mask, radius, cv2.INPAINT_TELEA)
    if smooth and smooth > 1:
        k = int(smooth) | 1
        # Feathered alpha: 1 deep inside the cover, fading to 0 outside it.
        alpha = cv2.GaussianBlur(cover_mask.astype(np.float32) / 255.0, (k, k), 0)
        blur = cv2.GaussianBlur(filled, (k, k), 0)
        if img.ndim == 3:
            alpha = alpha[:, :, None]
        filled = (filled * (1.0 - alpha) + blur * alpha).astype(np.uint8)
    return filled


def draw_roi(img_bgr, ctx, color=(255, 255, 0)):
    """Draw the search-ROI rectangle on a BGR image (no-op if no ROI)."""
    if ctx.roi is None:
        return img_bgr
    x0, y0, x1, y1 = ctx.roi
    cap.cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, 2)
    return img_bgr


def build_context(ambient_rgb, flash_rgb, both_rgb=None, meta=None,
                  use_roi=ROI_ENABLE, roi_side_frac=ROI_SIDE_FRAC,
                  roi_offset_frac=ROI_OFFSET_FRAC, use_cover=True) -> DetectionContext:
    """Build a DetectionContext from raw RGB capture arrays.

    Inputs are RGB uint8 (as produced by PIL / picamera2).  ``meta`` may carry
    ``live_rotation``; if present it is applied to ``cap.LIVE_ROTATION`` so the
    rotation matches the Pi's capture-time orientation, exactly as the offline
    harness does.

    ``use_roi`` restricts the search to a box just above the LED cover (see
    ``compute_search_roi``); the corneal-reflex anchor and every detector are
    confined to it.  Falls back to the full frame when there is no cover mask.
    """
    cv2 = cap.cv2
    meta = meta or {}
    if "live_rotation" in meta:
        cap.LIVE_ROTATION = int(meta["live_rotation"])
    rot = cap._CV2_ROTATIONS[cap.LIVE_ROTATION]

    amb = _rotate(cv2.cvtColor(ambient_rgb, cv2.COLOR_RGB2BGR), rot)
    flash = _rotate(cv2.cvtColor(flash_rgb, cv2.COLOR_RGB2BGR), rot)
    both = None
    if both_rgb is not None:
        both = _rotate(cv2.cvtColor(both_rgb, cv2.COLOR_RGB2BGR), rot)

    gray = cv2.cvtColor(amb, cv2.COLOR_BGR2GRAY)
    green = amb[:, :, 1]
    h, w = gray.shape[:2]

    cover_mask = cap.load_cover_mask((h, w)) if use_cover else None

    # Search ROI: a box just above the LED cover, with the cover subtracted.
    roi = None
    search_mask = None
    box_mask = None
    if use_roi:
        roi = compute_search_roi(cover_mask, (h, w), roi_side_frac, roi_offset_frac)
        if roi is not None:
            search_mask = _roi_search_mask(roi, cover_mask, (h, w))
            box_mask = np.zeros((h, w), dtype=np.uint8)
            x0, y0, x1, y1 = roi
            box_mask[y0:y1, x0:x1] = 255

    # Cover-filled images (inpainted + smoothed) shared by every detector, so the
    # near-black LED cover is a soft fill rather than a hard edge/black hole.
    ambient_filled = fill_cover(amb, cover_mask)
    green_filled = ambient_filled[:, :, 1]

    # Corneal-reflex anchor on the green channel (same cue cap.detect_pupil uses
    # to seed the search), confined to the ROI so it can't latch onto a bright
    # spot outside the search box.  Tolerant: None when no reflex is found.
    green_for_anchor = green if search_mask is None else np.where(
        search_mask > 0, green, 0).astype(np.uint8)
    anchor = None
    reflex_mask = None
    reflex = cap._find_corneal_reflex(green_for_anchor)
    if reflex is not None:
        cx, cy, rr, reflex_mask = reflex
        anchor = (cx, cy, rr)

    rmin = cap._SW_MIN_R_FRAC * min(h, w)
    rmax = cap._SW_MAX_R_FRAC * min(h, w)

    return DetectionContext(
        ambient=amb, flash=flash, both=both, gray=gray, green=green,
        cover_mask=cover_mask, anchor=anchor, reflex_mask=reflex_mask,
        rmin=rmin, rmax=rmax, meta=meta, roi=roi, search_mask=search_mask,
        box_mask=box_mask, ambient_filled=ambient_filled, green_filled=green_filled,
    )


def load_capture(folder: str):
    """Load (ambient_rgb, flash_rgb, both_rgb_or_None, meta) from a capture folder.

    Mirrors ``test_pupil_detection.load_capture`` but returns raw RGB arrays for
    ``build_context``.  Returns None if ambient.jpg or flash.jpg is missing.
    """
    import json
    ap = os.path.join(folder, "ambient.jpg")
    fp = os.path.join(folder, "flash.jpg")
    if not (os.path.isfile(ap) and os.path.isfile(fp)):
        return None
    ambient = np.array(Image.open(ap).convert("RGB"))
    flash = np.array(Image.open(fp).convert("RGB"))
    both = None
    bp = os.path.join(folder, "both.jpg")
    if os.path.isfile(bp):
        both = np.array(Image.open(bp).convert("RGB"))
    meta = {}
    mp = os.path.join(folder, "meta.json")
    if os.path.isfile(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
    return ambient, flash, both, meta


def context_for_folder(folder: str) -> Optional[DetectionContext]:
    """Convenience: load a capture folder and build its context, or None."""
    loaded = load_capture(folder)
    if loaded is None:
        return None
    ambient, flash, both, meta = loaded
    return build_context(ambient, flash, both, meta)


def find_capture_folders() -> "list[str]":
    """Sorted paths of all capture_* folders in Transfers/.

    Includes one level of subfolders, so the receiver's category dirs
    (no_pupil/ ridge_only/ ritnet_only/ both/) and any session subdir are picked
    up too.
    """
    if not os.path.isdir(TRANSFERS_DIR):
        return []
    out = []
    for entry in sorted(os.listdir(TRANSFERS_DIR)):
        path = os.path.join(TRANSFERS_DIR, entry)
        if not os.path.isdir(path):
            continue
        if entry.startswith("capture_"):
            out.append(path)
        else:                                   # a category / session subfolder
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isdir(subpath) and sub.startswith("capture_"):
                    out.append(subpath)
    return out
