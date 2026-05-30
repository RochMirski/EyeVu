#!/usr/bin/env python3
"""
Retina Flash Photography
Raspberry Pi Zero + Camera Module V2

CONTROLS
────────
The program boots straight into streaming mode.  Exiting streaming (s) drops
into a command prompt:

ENTER               Capture photo (ambient + flash pair)
s + ENTER           Re-enter streaming mode
q + ENTER           Quit

Each capture's raw ambient + flash pair is also pushed to the dev machine's
Transfers folder via scp, for offline pupil-detection testing.

Streaming mode
──────────────
SPACE (hold)        Flash LED (GPIO 17) on
r                   Toggle flash LED lock on/off
a                   Toggle ambient LED (GPIO 27) on/off
p                   Toggle live Orlosky pupil detection
←/→                 Rotate live feed
ENTER or e          Capture still
s                   Exit streaming

PUPIL DETECTION
───────────────
Each capture takes two successive images: an ambient image lit by the
GPIO 27 LED (pupil is a dark disc) and a retina-flash image lit by the
GPIO 17 LED.  The Orlosky method detects the pupil in the ambient image
and the resulting ellipse is drawn onto the flash photo.

PARAMETERS
──────────
exposure 32000
flash_gain 1.6
live_gain 1.0

pre_delay 0.05
post_delay 0.05
flash_duration 0.075

brightness 2
contrast 1

red_gain 2.2
blue_gain 1.4

exclude_bottom 0.0
debug 1
"""

import time
import os
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available.")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available.")

# ───────── CONFIG ─────────

LED_PIN   = 17     # flash LED   — retina retroreflection capture
LED_PIN_2 = 27     # ambient LED — diffuse light for pupil detection

PHOTO_PATH   = "/tmp/retina_preview.jpg"   # annotated flash photo
AMBIENT_PATH = "/tmp/retina_ambient.jpg"   # raw ambient image (Swirski input)

# ───────── IMAGE TRANSFER ─────────
# Each capture's raw ambient + flash pair is HTTP-POSTed to the dev machine's
# receiver.py, which writes them into its Transfers/ folder for offline
# pupil-detection testing.  The Pi joins the dev machine's Windows mobile
# hotspot; the machine is the hotspot gateway.
TRANSFER_ENABLED  = True
REMOTE_HOST       = "192.168.137.1"        # Windows hotspot gateway IP
REMOTE_PORT       = 8000                   # must match receiver.py
LOCAL_STAGING_DIR = "/tmp/eyevu_transfers"  # capture folders staged here first
TRANSFER_TIMEOUT  = 10                     # seconds per HTTP request

# Reduced slightly from 1640x1232
# Faster on Pi Zero while still sharp
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960

# Camera settings
FLASH_GAIN = 1.6
LIVE_GAIN = 1.0
EXPOSURE_TIME = 32000

# Manual white balance
RED_GAIN = 2.2
BLUE_GAIN = 1.4

# Flash timing
FLASH_PRE_DELAY = 0.05
FLASH_POST_DELAY = 0.05
FLASH_DURATION = 0.075

# Frame-drain — picamera2 buffers a couple of frames; after flipping an LED or
# changing the gain the first frame returned by capture_array() may still be a
# stale exposure under the previous state (visible as dark flash photos).  We
# discard this many frames after each state change before the keeper capture.
FLUSH_FRAMES = 2

# Image processing
BRIGHTNESS = 2.0
CONTRAST = 1.0

# Rotation (index into ROTATION_* tables below)
# 0=none, 1=90°CCW, 2=180°, 3=90°CW
LIVE_ROTATION = 1

# ──────────────────────────

# Rotation lookup tables
# cv2 values: None means skip the rotate call
_CV2_ROTATIONS = [
    None,
    cv2.ROTATE_90_COUNTERCLOCKWISE if CV2_AVAILABLE else None,
    cv2.ROTATE_180 if CV2_AVAILABLE else None,
    cv2.ROTATE_90_CLOCKWISE if CV2_AVAILABLE else None,
]
# PIL degrees (anticlockwise)
_PIL_ROTATIONS = [0, 90, 180, 270]


def setup_gpio():

    if not GPIO_AVAILABLE:
        return

    GPIO.setmode(GPIO.BCM)

    GPIO.setwarnings(False)

    GPIO.setup(LED_PIN,  GPIO.OUT)
    GPIO.setup(LED_PIN_2, GPIO.OUT)

    GPIO.output(LED_PIN,  GPIO.LOW)
    GPIO.output(LED_PIN_2, GPIO.LOW)


def cleanup_gpio():

    if not GPIO_AVAILABLE:
        return

    GPIO.output(LED_PIN,  GPIO.LOW)
    GPIO.output(LED_PIN_2, GPIO.LOW)

    GPIO.cleanup()


def flash_on():
    """Turn on the GPIO 17 flash LED (retina retroreflection)."""
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH)


def flash_off():
    """Turn off the GPIO 17 flash LED."""
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.LOW)


def ambient_on():
    """Turn on the GPIO 27 ambient LED (diffuse light for pupil detection)."""
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN_2, GPIO.HIGH)


def ambient_off():
    """Turn off the GPIO 27 ambient LED."""
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN_2, GPIO.LOW)


def apply_camera_settings(picam2, gain):

    picam2.set_controls({
        "ExposureTime": EXPOSURE_TIME,
        "AnalogueGain": gain,
        "AwbEnable": False,
        "ColourGains": (
            RED_GAIN,
            BLUE_GAIN
        )
    })


def print_settings():

    print("\nCurrent Settings")
    print("────────────────────────")

    print(f"resolution         = {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    print()

    print(f"exposure           = {EXPOSURE_TIME}")
    print(f"flash_gain         = {FLASH_GAIN}")
    print(f"live_gain          = {LIVE_GAIN}")

    print()

    print(f"red_gain           = {RED_GAIN}")
    print(f"blue_gain          = {BLUE_GAIN}")

    print()

    print(f"pre_delay          = {FLASH_PRE_DELAY}")
    print(f"post_delay         = {FLASH_POST_DELAY}")
    print(f"flash_duration     = {FLASH_DURATION}")

    print()

    print(f"brightness         = {BRIGHTNESS}")
    print(f"contrast           = {CONTRAST}")

    print()

    print(f"exclude_bottom     = {DETECT_EXCLUDE_BOTTOM}")
    print(f"debug              = {SWIRSKI_DEBUG}")

    print("────────────────────────\n")


# ───────── PUPIL DETECTION (Orlosky) ─────────
# Orlosky pupil detector — port of OrloskyPupilDetectorRaspberryPi.py.
# Four stages:
#   1. coarse: find the darkest small region
#   2. threshold the image at (darkest pixel value + offset), inverted
#   3. mask to a square ROI around the darkest point
#   4. dilate, find contours, fit an ellipse to the largest plausible one
# Run on the ambient image, where the pupil reads as a dark disc.
# (The Swirski detector further below is commented out — kept for reference.)

SWIRSKI_LIVE_DEFAULT  = False    # live-mode detection off by default

# ── Swirski Haar coarse-seed parameters (the dark-centre/light-surround
#    response seeds the Orlosky pipeline; the rest of Swirski stays commented) ──
_SW_MIN_R_FRAC = 0.03            # min pupil radius, as a fraction of min(h, w)
_SW_MAX_R_FRAC = 0.13            # max pupil radius, as a fraction of min(h, w)
_SW_N_RADII = 6                  # number of candidate radii to test
_SW_COARSE_DOWNSCALE = 4         # downscale factor for the cheap coarse search

# ── Swirski RANSAC ellipse-fit parameters (used by _swirski_fit_ellipse /
#    _swirski_support — the gradient-aware fit that tolerates a partially
#    occluded pupil boundary) ──
_SW_INLIER_DIST = 2.5            # px — boundary point counts as an inlier if this
                                 # close to the candidate ellipse
_SW_RANSAC_ITERS = 120           # RANSAC hypotheses (offline / capture mode)
_SW_RANSAC_ITERS_LIVE = 30       # fewer hypotheses for streaming-mode speed
_SW_MAX_AXIS_RATIO = 2.5         # reject ellipses more elongated than this

# ── Corneal-reflex anchor — the bright specular spot sits on/beside the pupil
#    centre and is the most reliable seed; a large dark surround distinguishes
#    it from the diffuse light-leak glow and from sclera/skin reflections ──
_REFLEX_BRIGHT_PCTILE = 99.5     # brightness percentile for specular candidates
_REFLEX_MIN_AREA = 4             # px — ignore single-pixel noise specks
_REFLEX_MAX_AREA_FRAC = 0.01     # max blob area (fraction of frame) — reflex is small
_REFLEX_MIN_FILL = 0.35          # min bbox fill — reflex is compact, glow is ragged
_PUPIL_ROI_R_MULT = 2.6          # detection ROI half-side = this x max pupil radius

# ── Radial pupil-boundary search (anchored at the reflex) ──
# Cast rays out from the anchor; the pupil edge is the strongest dark->light
# rise per ray.  Rays into the dark occluder patch show no rise and are dropped,
# so the ellipse is fit from the visible iris-side arc only.
_RAD_N_ANGLES = 96               # number of rays cast from the anchor
_RAD_SEARCH_MULT = 1.15          # search radius = this x max pupil radius (kept
                                 # tight so a ray stops at the pupil edge and does
                                 # not run on to the much brighter lower eyelid)
_RAD_GRAD_K = 2.2                # a ray's peak outward gradient must exceed this x
                                 # its own mean |gradient| to count as a boundary
                                 # ridge (per-ray, so contrast/illumination-robust)
_RAD_INLIER_PX = 9.0             # px — edge point counts toward the fit if this close
_RAD_RECENTRE_ITERS = 5          # radial-fit recentring passes (reflex is at the
                                 # pupil edge, so the centre must be iterated)
_RAD_MIN_COVER = 0.30            # min fraction of rays that must hit an edge — too
                                 # few means the boundary arc is too short to trust

# ── Concentric iris refinement — pupil and iris share a centre; the larger iris
#    arc stabilises the common centre (see _fit_concentric) ──
_IRIS_RP_MIN = 1.3               # iris radius must be at least this x pupil radius
_IRIS_RP_MAX = 5.0               # ...and at most this x (else it is not the iris)
_IRIS_SEARCH_MAX = 4.5           # search the iris ridge out to this x pupil radius
_IRIS_MIN_COVER = 0.35           # min iris-arc coverage before trusting the refine
_IRIS_MAX_RMS_FRAC = 0.10        # iris points must lie on a circle to this RMS
                                 # (fraction of iris radius), else they are noise

# ── Flash red-eye (retroreflection) detection + ambient/flash fusion ──
_REDEYE_WIN_MULT = 1.8           # search the flash glow within this x max pupil
                                 # radius of the ambient reflex anchor
_REDEYE_BLUE_SUB = 0.4           # red-weighting: glow map = R - this x B
_REDEYE_MIN_PEAK = 70            # min glow peak to consider a candidate at all
_REDEYE_TRUST = 120              # glow peak above which the red-eye is trusted to
                                 # override a disagreeing ambient fit
_AMBIENT_MIN_CONF = 0.25         # ambient fit confidence (coverage x inlier frac)
                                 # below which the result is flagged low-confidence

# ── Orlosky detector parameters ──
_ORL_THRESHOLD_OFFSET = 15       # binary threshold = darkest pixel value + this
_ORL_MASK_FRAC = 0.5             # ROI square side, as a fraction of min(h, w)
_ORL_ROI_R_MULT = 4.0            # ROI square side = this × Haar seed radius
_ORL_SEED_WIN_MULT = 1.0         # seed threshold patch half-size = this × radius
_ORL_MIN_AREA_FRAC = 0.003       # min contour area, as a fraction of frame area
_ORL_MAX_RATIO = 3.0             # max contour bounding-box aspect ratio
_ORL_DARKEST_WIN = 20            # window size for the darkest-region search
_ORL_DILATE_KERNEL = 5           # dilation kernel side
_ORL_DILATE_ITERS = 2            # dilation iterations

# ── CLAHE contrast boost — lift the pupil/iris edge out of the eye-socket
#    shadow before thresholding ──
_CLAHE_CLIP = 2.0                # CLAHE clip limit
_CLAHE_TILE = 8                  # CLAHE tile grid side

# ── Reflection inpainting — remove bright specular artefacts (corneal LED
#    reflexes) before detection so they do not punch holes in the dark pupil ──
_INPAINT_BRIGHT_THRESH = 220     # pixels brighter than this are specular
_INPAINT_DILATE = 7              # dilate the reflection mask to cover the halo
_INPAINT_RADIUS = 5              # cv2.inpaint neighbourhood radius

# Flash-LED-cover exclusion — the LED's physical cover intrudes into the frame
# as a large near-black blob.  It is detected dynamically (a large,
# border-touching, very dark region); DETECT_EXCLUDE_BOTTOM is an optional
# manual fallback band (see handle_parameter_command's `exclude_bottom`).
_LED_DARK_THRESH = 30            # intensity below which a pixel is "near-black"
_LED_MIN_AREA_FRAC = 0.05        # min blob area (fraction of frame) to qualify
_LED_MAX_AREA_FRAC = 0.45        # max blob area — bigger means the cover has
                                 # merged with the eye-socket shadow; skip it
                                 # rather than blanking half the eye
DETECT_EXCLUDE_BOTTOM = 0.0      # manual fallback: blank this bottom fraction

# Live mode: only recompute the detection every N frames
_LIVE_DETECT_SKIP = 5

# Debug — annotate EVERY detected pupil candidate (no plausibility filtering)
# instead of a single filtered result.  Toggle at runtime with `debug 0` / `debug 1`.
SWIRSKI_DEBUG = True

# Per-stage debug sink.  When this is a list, detect_pupil() appends a
# (stage_name, BGR_image) tuple after every pipeline stage so the caller can
# write them out (the offline harness turns these into stage_*.jpg or a montage).
# Leave as None in normal/live operation — _dbg() is then a no-op.
PUPIL_DEBUG_STAGES = None

# Stashed result of the last ambient detect_pupil() call, for ambient/flash
# fusion in detect_and_annotate().  (cx, cy) is the reflex anchor; pupil is
# (cx, cy, r) or None; conf in [0, 1] from edge coverage x inlier tightness.
_LAST_ANCHOR = None
_LAST_PUPIL = None
_LAST_CONF = 0.0


def _dbg(name, img):
    """Record an intermediate pipeline image for visual debugging.

    No-op unless PUPIL_DEBUG_STAGES is a list.  Grayscale inputs are promoted to
    BGR so every recorded stage can be written/montaged uniformly.
    """
    if PUPIL_DEBUG_STAGES is None:
        return
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    PUPIL_DEBUG_STAGES.append((name, out))


def _find_led_cover_mask(gray):
    """Locate the dark flash-LED cover so detection can ignore it.

    The cover is the only large, near-black region that touches the frame
    border.  A pupil is neither large nor border-touching, so it can never be
    matched here.  Returns a uint8 mask of the cover, or None if none qualifies.
    """
    h, w = gray.shape[:2]
    dark = (gray < _LED_DARK_THRESH).astype(np.uint8)
    if int(dark.sum()) == 0:
        return None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    min_area = _LED_MIN_AREA_FRAC * h * w
    mask = np.zeros((h, w), dtype=np.uint8)
    found = False
    for lbl in range(1, n):
        x  = stats[lbl, cv2.CC_STAT_LEFT]
        y  = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        touches_border = (x == 0 or y == 0 or x + bw == w or y + bh == h)
        if touches_border and area > min_area:
            mask[labels == lbl] = 255
            found = True

    if not found:
        return None
    # If the total dark border-region is implausibly large, the cover has fused
    # with the eye-socket shadow and dark iris into one mass.  Blanking it would
    # erase the pupil's lighter surround (and the pupil itself), so treat it as
    # "no reliable cover" and skip exclusion entirely.
    if int(mask.sum()) / 255 > _LED_MAX_AREA_FRAC * h * w:
        return None
    # grow slightly — the out-of-focus cover has a soft, fuzzy edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, k)


def _apply_exclusions(gray):
    """Return a copy of `gray` with the flash-LED cover blanked to white.

    The cover is a large, border-touching, near-black blob that can intrude
    from any edge of the (rotated) frame.  Its 2D mask is whited out (255) so
    it can never be mistaken for a dark pupil — no assumption is made about
    which edge it enters from.

    If no LED cover is detected, the optional manual DETECT_EXCLUDE_BOTTOM band
    is used as a fallback.
    """
    h = gray.shape[0]
    g = gray.copy()

    cover = _find_led_cover_mask(g)
    if cover is not None:
        g[cover > 0] = 255                     # white-out the cover, wherever it is
        return g

    # fallback — no LED cover found: optional manual bottom band only
    if DETECT_EXCLUDE_BOTTOM > 0:
        y_ex = int(h * (1.0 - min(0.9, DETECT_EXCLUDE_BOTTOM)))
        g[y_ex:, :] = 255
    return g


# ── Swirski detector (commented out — Orlosky is used instead; see below) ──
'''
def _swirski_coarse(gray, rmin, rmax):
    """Stage 1 — Haar-like coarse pupil detection.

    The pupil is a dark disc surrounded by a lighter iris.  For each candidate
    radius (rmin..rmax) we evaluate, at every pixel, (mean of a surrounding box)
    − (mean of a central box); the pupil centre maximises this response.  Box
    means are O(1) via cv2.boxFilter, so testing a handful of radii is cheap.

    Returns (cx, cy, r) in `gray` coordinates, or None.
    """
    h, w = gray.shape[:2]
    ds = _SW_COARSE_DOWNSCALE
    small = cv2.resize(gray, (max(1, w // ds), max(1, h // ds)))
    small = small.astype(np.float32)
    sh, sw = small.shape[:2]

    best_resp = -1e9
    best = None
    for r in np.linspace(rmin, rmax, _SW_N_RADII):
        rs = max(1, int(r / ds))          # pupil-scale box half-size
        inner_k = rs * 2 + 1
        outer_k = rs * 6 + 1              # 3× scale surround
        m = rs * 3                        # border to keep both boxes in-frame
        if outer_k >= min(sh, sw) or m * 2 >= min(sh, sw):
            continue

        inner = cv2.boxFilter(small, -1, (inner_k, inner_k),
                              normalize=True, borderType=cv2.BORDER_REPLICATE)
        outer = cv2.boxFilter(small, -1, (outer_k, outer_k),
                              normalize=True, borderType=cv2.BORDER_REPLICATE)

        # surround mean = (outer*outer_area − inner*inner_area) / surround_area
        ia = float(inner_k * inner_k)
        oa = float(outer_k * outer_k)
        surround = (outer * oa - inner * ia) / (oa - ia)
        resp = surround - inner           # large where centre dark, ring light

        roi = resp[m:sh - m, m:sw - m]
        _, mx, _, mx_loc = cv2.minMaxLoc(roi)
        if mx > best_resp:
            best_resp = mx
            cx = (mx_loc[0] + m) * ds
            cy = (mx_loc[1] + m) * ds
            best = (int(cx), int(cy), int(r))

    return best


def _swirski_segment(gray, cx, cy, r, rmin, rmax):
    """Stage 2 — intensity-histogram segmentation around the coarse centre.

    Crops a tight ROI (~2.2× the coarse radius), applies an Otsu threshold (the
    two-cluster intensity split that separates the dark pupil from the lighter
    iris), cleans it morphologically and keeps the blob containing the coarse
    centre.  The blob is then checked for pupil plausibility — it must not run
    off the ROI edge, must be near-convex, and must be of pupil scale.

    Returns (mask, x0, y0) — binary pupil mask and ROI top-left offset — or None.
    """
    h, w = gray.shape[:2]
    half = max(20, int(r * 2.2))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0 or min(roi.shape[:2]) < 10:
        return None

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    # THRESH_BINARY_INV → dark pupil becomes 255
    _, mask = cv2.threshold(roi, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Keep only the connected component containing the coarse centre
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    lx, ly = cx - x0, cy - y0
    lbl = 0
    if 0 <= ly < mask.shape[0] and 0 <= lx < mask.shape[1]:
        lbl = int(labels[ly, lx])
    if lbl == 0:
        # coarse centre not on a blob — fall back to the largest blob
        lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # ── pupil-plausibility checks ──────────────────────────────────────────
    mh, mw = mask.shape[:2]
    bx = stats[lbl, cv2.CC_STAT_LEFT]
    by = stats[lbl, cv2.CC_STAT_TOP]
    bw = stats[lbl, cv2.CC_STAT_WIDTH]
    bh = stats[lbl, cv2.CC_STAT_HEIGHT]
    area = stats[lbl, cv2.CC_STAT_AREA]
    # touches the ROI border → blob runs off the crop, not a clean pupil
    if bx == 0 or by == 0 or bx + bw == mw or by + bh == mh:
        return None
    # equivalent radius must be of pupil scale
    equiv_r = np.sqrt(area / np.pi)
    if equiv_r < rmin or equiv_r > rmax:
        return None

    mask = np.where(labels == lbl, 255, 0).astype(np.uint8)

    # solidity — a pupil blob is near-convex
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if hull_area <= 0 or cv2.contourArea(c) / hull_area < _SW_MIN_SOLIDITY:
        return None

    return mask, x0, y0


def _swirski_support(ellipse, pts, gx, gy):
    """Score a candidate ellipse by image-aware support.

    A contour point supports the ellipse if it lies within _SW_INLIER_DIST of
    the ellipse boundary AND its local image gradient points outward (dark
    pupil → light iris) — Swirski's image-aware support term.

    Returns (inlier_count, inlier_mask).
    """
    (ex, ey), (MA, ma), ang = ellipse
    a, b = MA / 2.0, ma / 2.0
    if a < 1.0 or b < 1.0:
        return 0, None

    th = np.deg2rad(ang)
    cos_t, sin_t = np.cos(th), np.sin(th)

    dx = pts[:, 0] - ex
    dy = pts[:, 1] - ey
    # rotate into the ellipse's canonical frame
    xr =  cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    t = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)   # 1.0 on the boundary

    mean_r = (a + b) / 2.0
    near = np.abs(t - 1.0) < (_SW_INLIER_DIST / mean_r)

    # gradient must point outward from the ellipse centre (dark → light)
    ix = np.clip(pts[:, 0].astype(int), 0, gx.shape[1] - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, gx.shape[0] - 1)
    g_dot = gx[iy, ix] * dx + gy[iy, ix] * dy

    inliers = near & (g_dot > 0)
    return int(inliers.sum()), inliers


def _swirski_fit_ellipse(mask, gray_roi, iters):
    """Stage 3 — RANSAC ellipse fit with image-aware support.

    Fits an ellipse to the pupil-mask boundary.  Each RANSAC hypothesis is
    built from 5 random boundary points and scored by _swirski_support; the
    best hypothesis is refined with a final fit to its inliers.

    Returns an ellipse ((cx,cy),(MA,ma),angle) in ROI coordinates, or None.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None

    # image gradient — used by the image-aware support test
    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)

    n = len(pts)
    rng = np.random.default_rng(0)
    best_score = 0
    best_inliers = None
    for _ in range(iters):
        idx = rng.choice(n, 5, replace=False)
        try:
            ell = cv2.fitEllipse(pts[idx])
        except cv2.error:
            continue
        score, inliers = _swirski_support(ell, pts, gx, gy)
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 5:
        # fall back to a plain fit on all boundary points
        try:
            return cv2.fitEllipse(pts)
        except cv2.error:
            return None
    try:
        return cv2.fitEllipse(pts[best_inliers])
    except cv2.error:
        return None


def swirski_detect_pupil(gray, live=False):
    """Detect the pupil with the Swirski method.

    `gray` is a single-channel uint8 image (the ambient image).  Returns an
    ellipse ((cx,cy),(MA,ma),angle) in `gray` coordinates, or None.
    `live=True` uses fewer RANSAC iterations for streaming-mode speed.

    The dark flash-LED cover, and an optional manual bottom band, are blanked
    out so they cannot be mistaken for the pupil.
    """
    if not CV2_AVAILABLE:
        return None
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    # Blank the flash-LED cover and the optional manual fallback band
    gray = _apply_exclusions(gray)

    rmin = _SW_MIN_R_FRAC * min(h, w)
    rmax = _SW_MAX_R_FRAC * min(h, w)

    coarse = _swirski_coarse(gray, rmin, rmax)
    if coarse is None:
        return None
    cx, cy, r = coarse

    seg = _swirski_segment(gray, cx, cy, r, rmin, rmax)
    if seg is None:
        return None
    mask, x0, y0 = seg

    gray_roi = gray[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    iters = _SW_RANSAC_ITERS_LIVE if live else _SW_RANSAC_ITERS
    ell = _swirski_fit_ellipse(mask, gray_roi, iters)
    if ell is None:
        return None

    (ex, ey), (MA, ma), ang = ell
    # reject implausible ellipses (too elongated or larger than a pupil)
    if min(MA, ma) <= 0:
        return None
    if max(MA, ma) / min(MA, ma) > _SW_MAX_AXIS_RATIO:
        return None
    if max(MA, ma) / 2.0 > rmax * 1.3:
        return None

    return ((ex + x0, ey + y0), (MA, ma), ang)


def _swirski_all_candidates(gray):
    """Debug — segment the whole (exclusion-masked) image and fit an ellipse to
    every dark blob.  Returns a list of candidate dicts with shape metrics; no
    plausibility filtering is applied (nothing is discarded by thresholds)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for lbl in range(1, n):
        comp = np.where(labels == lbl, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 5:                       # cv2.fitEllipse needs >= 5 points
            continue
        try:
            ell = cv2.fitEllipse(c)
        except cv2.error:
            continue
        MA, ma = ell[1]
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        equiv_r = float(np.sqrt(area / np.pi))
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = (cv2.contourArea(c) / hull_area) if hull_area > 0 else 0.0
        axis_ratio = (max(MA, ma) / min(MA, ma)) if min(MA, ma) > 0 else 0.0
        out.append({
            "ellipse": ell,
            "equiv_r": equiv_r,
            "solidity": solidity,
            "axis_ratio": axis_ratio,
        })
    return out


def swirski_detect(gray, live=False):
    """Return a list of candidate overlays to draw on the image.

    Normal mode: the single best, plausibility-checked pupil (or an empty list).
    Debug mode (`SWIRSKI_DEBUG`): every detected candidate, unfiltered, each
    labelled with its shape metrics, plus the stage-1 Haar coarse pick.

    Each overlay is a dict: {"ellipse": ((cx,cy),(MA,ma),ang),
                             "label": str, "color": (b,g,r)}.
    """
    if not CV2_AVAILABLE:
        return []
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if not SWIRSKI_DEBUG:
        ell = swirski_detect_pupil(gray, live=live)
        if ell is None:
            return []
        (ex, ey), _, _ = ell
        if not live:
            print(f"Pupil detected at ({int(ex)}, {int(ey)}).")
        return [{"ellipse": ell, "label": "", "color": (0, 255, 0)}]

    # ── debug: annotate every candidate, discard nothing ──────────────────
    masked = _apply_exclusions(gray)
    cands = _swirski_all_candidates(masked)
    if not live:
        print(f"[debug] {len(cands)} pupil candidate(s):")
        for i, c in enumerate(cands, 1):
            print(f"  {i}: r={c['equiv_r']:.0f}  solidity={c['solidity']:.2f}"
                  f"  axis_ratio={c['axis_ratio']:.2f}")

    overlays = []
    for i, c in enumerate(cands, 1):
        overlays.append({
            "ellipse": c["ellipse"],
            "label": (f"{i} r={c['equiv_r']:.0f} "
                      f"sol={c['solidity']:.2f} ar={c['axis_ratio']:.1f}"),
            "color": (0, 255, 255),          # yellow — candidate
        })

    # also show where the Haar coarse stage thinks the pupil is
    h, w = masked.shape[:2]
    coarse = _swirski_coarse(masked, _SW_MIN_R_FRAC * min(h, w),
                             _SW_MAX_R_FRAC * min(h, w))
    if coarse is not None:
        cx, cy, r = coarse
        overlays.append({
            "ellipse": ((float(cx), float(cy)),
                        (float(2 * r), float(2 * r)), 0.0),
            "label": "coarse",
            "color": (255, 0, 255),          # magenta — coarse pick
        })
    return overlays
'''
# ── end Swirski detector (commented out) ────────────────────────────────────


# ───────── ORLOSKY DETECTOR — active ─────────

def _swirski_coarse(gray, rmin, rmax):
    """Stage 1 — Haar-like coarse pupil detection (dark centre / light ring).

    The pupil is a dark disc surrounded by a lighter iris.  For each candidate
    radius (rmin..rmax) we evaluate, at every pixel, (mean of a surrounding box)
    − (mean of a central box); the pupil centre maximises this response.  A flat
    dark region (LED cover, eye-socket shadow) has no light surround and scores
    low, so this seeds the pupil far more reliably than a plain darkest-pixel
    search.  Box means are O(1) via cv2.boxFilter, so a handful of radii is cheap.

    Returns (cx, cy, r) in `gray` coordinates, or None.
    """
    h, w = gray.shape[:2]
    ds = _SW_COARSE_DOWNSCALE
    small = cv2.resize(gray, (max(1, w // ds), max(1, h // ds)))
    small = small.astype(np.float32)
    sh, sw = small.shape[:2]

    best_resp = -1e9
    best = None
    for r in np.linspace(rmin, rmax, _SW_N_RADII):
        rs = max(1, int(r / ds))          # pupil-scale box half-size
        inner_k = rs * 2 + 1
        outer_k = rs * 6 + 1              # 3x scale surround
        m = rs * 3                        # border to keep both boxes in-frame
        if outer_k >= min(sh, sw) or m * 2 >= min(sh, sw):
            continue

        inner = cv2.boxFilter(small, -1, (inner_k, inner_k),
                              normalize=True, borderType=cv2.BORDER_REPLICATE)
        outer = cv2.boxFilter(small, -1, (outer_k, outer_k),
                              normalize=True, borderType=cv2.BORDER_REPLICATE)

        # surround mean = (outer*outer_area - inner*inner_area) / surround_area
        ia = float(inner_k * inner_k)
        oa = float(outer_k * outer_k)
        surround = (outer * oa - inner * ia) / (oa - ia)
        resp = surround - inner           # large where centre dark, ring light

        roi = resp[m:sh - m, m:sw - m]
        _, mx, _, mx_loc = cv2.minMaxLoc(roi)
        if mx > best_resp:
            best_resp = mx
            cx = (mx_loc[0] + m) * ds
            cy = (mx_loc[1] + m) * ds
            best = (int(cx), int(cy), int(r))

    return best


def _enhance_contrast(gray):
    """Apply CLAHE to lift the pupil/iris edge out of the eye-socket shadow."""
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP,
                            tileGridSize=(_CLAHE_TILE, _CLAHE_TILE))
    return clahe.apply(gray)


def _orlosky_seed(gray):
    """Coarse seed for the Orlosky pipeline.

    Prefers the Swirski Haar dark-centre/light-surround response; falls back to
    the plain darkest-area search if no Haar response is found.  Returns
    (cx, cy, r) — r is None for the darkest-area fallback.
    """
    h, w = gray.shape[:2]
    rmin = _SW_MIN_R_FRAC * min(h, w)
    rmax = _SW_MAX_R_FRAC * min(h, w)
    coarse = _swirski_coarse(gray, rmin, rmax)
    if coarse is not None:
        return coarse
    darkest = _orlosky_darkest_area(gray)
    if darkest is None:
        return None
    return (darkest[0], darkest[1], None)


def _orlosky_darkest_area(gray):
    """Coarse stage — return (x, y) of the centre of the darkest small region.

    Equivalent to Orlosky's sparse darkest-area search, done here with a
    box-filter local mean + minMaxLoc for speed.
    """
    win = _ORL_DARKEST_WIN
    blurred = cv2.boxFilter(gray, -1, (win, win),
                            normalize=True, borderType=cv2.BORDER_REPLICATE)
    h, w = gray.shape[:2]
    b = win
    if h > 2 * b and w > 2 * b:
        roi = blurred[b:h - b, b:w - b]
        ox, oy = b, b
    else:
        roi = blurred
        ox, oy = 0, 0
    _minv, _maxv, min_loc, _maxloc = cv2.minMaxLoc(roi)
    return (min_loc[0] + ox, min_loc[1] + oy)


def _mask_outside_square(image, center, size):
    """Zero every pixel outside a `size`×`size` square centred on `center`."""
    x, y = center
    half = size // 2
    mask = np.zeros_like(image)
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(image.shape[1], x + half)
    y1 = min(image.shape[0], y + half)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(image, mask)


def _orlosky_contours(gray):
    """Run the Orlosky pipeline on an exclusion-masked grayscale image:
    Haar coarse seed → relative threshold → ROI mask → dilate → contours.

    Returns (contours, seed_point)."""
    seed = _orlosky_seed(gray)
    if seed is None:
        return [], None
    sx, sy, r = seed

    h, w = gray.shape[:2]
    # threshold value from the darkest pixel in a small patch around the seed,
    # not the global darkest pixel (which may sit in residual shadow)
    win = int(r * _ORL_SEED_WIN_MULT) if r else _ORL_DARKEST_WIN
    win = max(2, win)
    x0 = max(0, sx - win); x1 = min(w, sx + win + 1)
    y0 = max(0, sy - win); y1 = min(h, sy + win + 1)
    patch = gray[y0:y1, x0:x1]
    dval = int(patch.min()) if patch.size else int(gray[sy, sx])
    thresh = dval + _ORL_THRESHOLD_OFFSET

    # dark pupil → white
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    # keep only a square ROI around the seed — sized from the seed radius when
    # available, else the fixed fraction of the frame
    roi_side = int(r * _ORL_ROI_R_MULT) if r else int(_ORL_MASK_FRAC * min(h, w))
    binary = _mask_outside_square(binary, (sx, sy), roi_side)
    # dilate to close the pupil blob
    kernel = np.ones((_ORL_DILATE_KERNEL, _ORL_DILATE_KERNEL), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=_ORL_DILATE_ITERS)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours, seed


def _inpaint_reflections(gray, mask=None):
    """Inpaint bright specular reflection artefacts (corneal LED reflexes).

    These near-saturated spots sit on top of the dark pupil and would punch
    bright holes through the threshold mask, breaking the pupil contour.

    If `mask` is given (the located corneal-reflex blob), only that region is
    repaired — keeping the fix tight on the pupil and leaving the light-leak
    glow untouched.  Otherwise every near-saturated pixel is inpainted.
    Returns the repaired grayscale image (unchanged if nothing to repair).
    """
    if mask is None:
        _, mask = cv2.threshold(gray, _INPAINT_BRIGHT_THRESH, 255,
                                cv2.THRESH_BINARY)
    if int(cv2.countNonZero(mask)) == 0:
        return gray
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (_INPAINT_DILATE, _INPAINT_DILATE))
    mask = cv2.dilate(mask, k)              # cover the soft reflection halo
    return cv2.inpaint(gray, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA)


def _annulus_mean(gray, cx, cy, r):
    """Mean intensity of a ring just outside radius `r` around (cx, cy).

    Used to test whether a bright blob sits in a dark surround (the pupil): the
    corneal reflex does, the diffuse light-leak glow does not.
    """
    h, w = gray.shape[:2]
    r_in = int(r * 1.6) + 2
    r_out = int(r * 3.2) + 4
    x0 = max(0, cx - r_out); x1 = min(w, cx + r_out + 1)
    y0 = max(0, cy - r_out); y1 = min(h, cy + r_out + 1)
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return 255.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    if not np.any(ring):
        return 255.0
    return float(win[ring].mean())


def _find_corneal_reflex(gray):
    """Locate the corneal reflex — the bright specular spot on/beside the pupil.

    Thresholds the brightest pixels, then scores each compact blob by
    (bbox fill) x (dark surround): the reflex is small, near-circular and ringed
    by the dark pupil, which rejects the large diffuse light-leak glow and the
    brighter-surrounded sclera/skin reflections.

    Returns (cx, cy, equiv_r, blob_mask) for the best reflex, or None.
    """
    h, w = gray.shape[:2]
    thr = max(_INPAINT_BRIGHT_THRESH,
              float(np.percentile(gray, _REFLEX_BRIGHT_PCTILE)))
    _, bright = cv2.threshold(gray, int(thr), 255, cv2.THRESH_BINARY)
    if int(cv2.countNonZero(bright)) == 0:
        return None

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
    max_area = _REFLEX_MAX_AREA_FRAC * h * w

    best = None
    best_lbl = 0
    best_score = -1.0
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < _REFLEX_MIN_AREA or area > max_area:
            continue
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        fill = area / float(bw * bh) if bw * bh > 0 else 0.0
        if fill < _REFLEX_MIN_FILL:
            continue
        cx = int(round(centroids[lbl][0]))
        cy = int(round(centroids[lbl][1]))
        equiv_r = float(np.sqrt(area / np.pi))
        dark_surround = 1.0 - _annulus_mean(gray, cx, cy, equiv_r) / 255.0
        score = dark_surround * fill
        if score > best_score:
            best_score = score
            best = (cx, cy, max(2, int(round(equiv_r))))
            best_lbl = lbl

    if best is None:
        return None
    blob_mask = np.where(labels == best_lbl, 255, 0).astype(np.uint8)
    return best[0], best[1], best[2], blob_mask


def _segment_pupil_at(gray, cx, cy, rmin, rmax):
    """Build the pupil mask in an ROI centred on the (reflex) anchor.

    Relative-thresholds the ROI at (local-darkest + offset), inverts so the dark
    pupil becomes white, cleans it morphologically and keeps the connected
    component under the anchor.  Deliberately does NOT reject border-touching or
    low-solidity blobs: when the dark occluder patch merges with the pupil the
    blob is irregular, and it is the gradient-aware RANSAC fit downstream that
    recovers the true pupil ellipse from the visible iris-side arc.

    Returns (mask, x0, y0) — mask in ROI coords, ROI top-left in `gray` — or None.
    """
    h, w = gray.shape[:2]
    half = max(20, int(rmax * _PUPIL_ROI_R_MULT))
    x0 = max(0, cx - half); x1 = min(w, cx + half)
    y0 = max(0, cy - half); y1 = min(h, cy + half)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0 or min(roi.shape[:2]) < 10:
        return None

    roi_b = cv2.GaussianBlur(roi, (5, 5), 0)
    lx, ly = cx - x0, cy - y0
    win = max(4, int(rmin))
    px0 = max(0, lx - win); px1 = min(roi.shape[1], lx + win + 1)
    py0 = max(0, ly - win); py1 = min(roi.shape[0], ly + win + 1)
    patch = roi_b[py0:py1, px0:px1]
    dval = int(patch.min()) if patch.size else int(roi_b[ly, lx])
    thresh = dval + _ORL_THRESHOLD_OFFSET

    _, mask = cv2.threshold(roi_b, thresh, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    lbl = 0
    if 0 <= ly < mask.shape[0] and 0 <= lx < mask.shape[1]:
        lbl = int(labels[ly, lx])
    if lbl == 0:                              # anchor not on a blob — take largest
        lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == lbl, 255, 0).astype(np.uint8)
    return mask, x0, y0


def _swirski_support(ellipse, pts, gx, gy):
    """Score a candidate ellipse by image-aware support.

    A contour point supports the ellipse if it lies within _SW_INLIER_DIST of
    the ellipse boundary AND its local image gradient points outward (dark
    pupil -> light iris) — Swirski's image-aware support term.  Boundary points
    on the dark occluder patch have no such outward gradient, so they do not
    support the fit, which is what makes this occlusion-tolerant.

    Returns (inlier_count, inlier_mask).
    """
    (ex, ey), (MA, ma), ang = ellipse
    a, b = MA / 2.0, ma / 2.0
    if a < 1.0 or b < 1.0:
        return 0, None

    th = np.deg2rad(ang)
    cos_t, sin_t = np.cos(th), np.sin(th)

    dx = pts[:, 0] - ex
    dy = pts[:, 1] - ey
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    t = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)   # 1.0 on the boundary

    mean_r = (a + b) / 2.0
    near = np.abs(t - 1.0) < (_SW_INLIER_DIST / mean_r)

    ix = np.clip(pts[:, 0].astype(int), 0, gx.shape[1] - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, gx.shape[0] - 1)
    g_dot = gx[iy, ix] * dx + gy[iy, ix] * dy

    inliers = near & (g_dot > 0)
    return int(inliers.sum()), inliers


def _swirski_fit_ellipse(mask, gray_roi, iters):
    """RANSAC ellipse fit with image-aware support (Swirski stage 3).

    Fits an ellipse to the pupil-mask boundary.  Each RANSAC hypothesis is built
    from 5 random boundary points and scored by _swirski_support; the best
    hypothesis is refined with a final fit to its inliers.

    Returns an ellipse ((cx,cy),(MA,ma),angle) in ROI coordinates, or None.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None

    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)

    n = len(pts)
    rng = np.random.default_rng(0)
    best_score = 0
    best_inliers = None
    for _ in range(iters):
        idx = rng.choice(n, 5, replace=False)
        try:
            ell = cv2.fitEllipse(pts[idx])
        except cv2.error:
            continue
        score, inliers = _swirski_support(ell, pts, gx, gy)
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 5:
        try:
            return cv2.fitEllipse(pts)
        except cv2.error:
            return None
    try:
        return cv2.fitEllipse(pts[best_inliers])
    except cv2.error:
        return None


def _ray_ridges(gray, cx, cy, r_lo, r_hi, gx=None, gy=None):
    """Cast _RAD_N_ANGLES rays from (cx, cy) over radii [r_lo, r_hi); return
    (angles, radii).

    radii[i] is the radius of the strongest OUTWARD intensity gradient (dark ->
    bright) along ray i — a circular-boundary ridge — or np.nan if that ray has
    no clear outward edge (e.g. a direction into the equally-dark occluder patch,
    where the radial gradient stays near zero, so it drops out).  Using the
    gradient ridge, not an intensity step, locks onto the true boundary and
    ignores the slow violet-glow ramp.  Pass precomputed gx/gy to avoid recompute.
    """
    h, w = gray.shape[:2]
    r_lo = max(3, int(r_lo))
    r_hi = max(r_lo + 4, int(r_hi))
    angles = np.linspace(0, 2 * np.pi, _RAD_N_ANGLES, endpoint=False)
    radii = np.full(_RAD_N_ANGLES, np.nan, np.float32)
    if gx is None or gy is None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    rs = np.arange(r_lo, r_hi, dtype=np.float32)
    if len(rs) < 3:
        return angles, radii
    for i, a in enumerate(angles):
        ca, sa = np.cos(a), np.sin(a)
        xs = np.clip((cx + rs * ca).astype(int), 0, w - 1)
        ys = np.clip((cy + rs * sa).astype(int), 0, h - 1)
        grad = gx[ys, xs] * ca + gy[ys, xs] * sa        # outward radial derivative
        thr = _RAD_GRAD_K * (float(np.mean(np.abs(grad))) + 1e-6)
        # Innermost clear outward ridge, not the global max: when both the
        # pupil/iris and the (stronger) iris/sclera edges are on the ray, this
        # keeps the radius on the pupil instead of overshooting to the iris.
        inner = grad[1:-1]
        cand = np.where((inner > thr) & (inner >= grad[:-2])
                        & (inner > grad[2:]))[0] + 1     # local maxima above thr
        if len(cand):
            radii[i] = float(rs[cand[0]])
    return angles, radii


def _ray_edges(gray, cx, cy, rmin, rmax, r_start=0):
    """Pupil-boundary ridges: _ray_ridges over the plausible-pupil radius band.

    `r_start` skips the innermost radii (set it past the inpainted-reflex blob).
    """
    r_lo = max(3, int(rmin * 0.5), int(r_start))
    r_hi = int(rmax * _RAD_SEARCH_MULT)
    return _ray_ridges(gray, cx, cy, r_lo, r_hi)


def _fit_concentric(P, Q):
    """Joint least-squares fit of two CONCENTRIC circles to point sets P (pupil)
    and Q (iris), sharing one centre.  Returns (cx, cy, rp, ri) or None.

    The iris ring is larger and usually has a longer visible arc, so tying its
    centre to the pupil's pins the common centre far more stably than the short
    pupil arc alone — this is what keeps the centre right when the reflex sits on
    the iris or the pupil's upper edge is lost in the dark socket.

    Linear (Kasa) form with a shared centre (a, b): for every point
    x^2+y^2 = 2a*x + 2b*y + c, where the constant c is per-ring
    (c_p = rp^2-a^2-b^2 for pupil points, c_i for iris points); solve for
    [a, b, c_p, c_i] in one least-squares system.
    """
    if len(P) < 3 or len(Q) < 3:
        return None
    rows = []
    rhs = []
    for (x, y) in P:
        rows.append([2 * x, 2 * y, 1.0, 0.0]); rhs.append(x * x + y * y)
    for (x, y) in Q:
        rows.append([2 * x, 2 * y, 0.0, 1.0]); rhs.append(x * x + y * y)
    A = np.array(rows, np.float64)
    b = np.array(rhs, np.float64)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, bb, cp, ci = sol
    rp2 = cp + a * a + bb * bb
    ri2 = ci + a * a + bb * bb
    if rp2 <= 0 or ri2 <= 0:
        return None
    return float(a), float(bb), float(np.sqrt(rp2)), float(np.sqrt(ri2))


def _points_from_radii(cx, cy, angles, radii):
    """Convert (angles, radii) edge hits into an (N, 2) float32 point array."""
    m = ~np.isnan(radii)
    if not np.any(m):
        return np.empty((0, 2), np.float32)
    xs = cx + radii[m] * np.cos(angles[m])
    ys = cy + radii[m] * np.sin(angles[m])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _refine_center(gray, cx, cy, rmin, rmax, iters):
    """Recenter onto the pupil using opposing ray pairs (robust to occlusion).

    For each axis where BOTH opposite rays hit a pupil edge, the midpoint along
    that axis estimates the centre; the median offset over all two-sided axes is
    the recentre step.  One-sided axes (the occluded patch side, or the bright
    glow running off one way) are skipped, so the estimate cannot run away the
    way a full ellipse-fit recentre does.  Returns the refined (cx, cy).
    """
    half = _RAD_N_ANGLES // 2
    for _ in range(iters):
        angles, radii = _ray_edges(gray, cx, cy, rmin, rmax)
        offs = []
        for k in range(half):
            rp, rn = radii[k], radii[k + half]
            if not np.isnan(rp) and not np.isnan(rn):
                shift = (rp - rn) / 2.0          # along +angles[k]
                offs.append((shift * np.cos(angles[k]),
                             shift * np.sin(angles[k])))
        if not offs:
            break
        ox = float(np.median([o[0] for o in offs]))
        oy = float(np.median([o[1] for o in offs]))
        if abs(ox) < 1.5 and abs(oy) < 1.5:
            break
        cx = int(round(cx + ox))
        cy = int(round(cy + oy))
    return cx, cy


def _fit_ellipse_robust(pts):
    """Least-squares ellipse fit to boundary points, with one outlier-rejecting
    refit (drop points more than _RAD_INLIER_PX from the first fit).  Returns
    ((cx,cy),(MA,ma),angle) or None."""
    if len(pts) < 5:
        return None
    try:
        ell = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    except cv2.error:
        return None

    (ex, ey), (MA, ma), ang = ell
    a, b = MA / 2.0, ma / 2.0
    if a < 1.0 or b < 1.0:
        return ell
    th = np.deg2rad(ang)
    cos_t, sin_t = np.cos(th), np.sin(th)
    dx = pts[:, 0] - ex
    dy = pts[:, 1] - ey
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    t = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    mean_r = (a + b) / 2.0
    inl = np.abs(t - 1.0) * mean_r < _RAD_INLIER_PX
    if int(inl.sum()) >= 5:
        try:
            return cv2.fitEllipse(pts[inl].reshape(-1, 1, 2))
        except cv2.error:
            return ell
    return ell


def _fit_circle(pts):
    """Algebraic (Kasa) least-squares circle fit.  Returns (cx, cy, r) or None.

    A circle (3 DOF) — not an ellipse (5 DOF) — because the pupil's upper edge
    merges into the equally-dark socket, so usually only a partial boundary arc
    is visible; an ellipse fit to a <180-degree arc is ill-posed and explodes,
    whereas a circle is recovered stably (pupils are near-circular here anyway).
    """
    if len(pts) < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones(len(x))], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, bb, c = sol
    r2 = c + a * a + bb * bb
    if r2 <= 0:
        return None
    return float(a), float(bb), float(np.sqrt(r2))


def _fit_circle_robust(pts):
    """Circle fit with one median-radius outlier-rejection refit (drops stray
    eyelid-crease edge points).  Returns (cx, cy, r) or None."""
    fit = _fit_circle(pts)
    if fit is None:
        return None
    a, b, r = fit
    d = np.hypot(pts[:, 0] - a, pts[:, 1] - b)
    inl = np.abs(d - np.median(d)) < _RAD_INLIER_PX * 2.0
    if int(inl.sum()) >= 3 and int(inl.sum()) < len(pts):
        f2 = _fit_circle(pts[inl])
        if f2 is not None:
            return f2
    return fit


def detect_redeye(flash_bgr, ax, ay, rmin, rmax):
    """Detect the pupil from the flash image's red-eye retroreflection.

    When the illumination, pupil and camera are well aligned the flash returns a
    bright warm (reddish) glow straight back through the pupil — a strong
    *positive* pupil cue that survives when the ambient dark-disc is occluded.
    The search is constrained to a window around the ambient reflex anchor
    (`ax, ay`) so stray warm skin/sclera reflections elsewhere are ignored.

    Returns (cx, cy, r, conf) for the glow, or None.  `conf` is the glow peak on
    a red-weighted, blurred map (flash is otherwise near-black, so a true
    retroreflection stands out); callers gate on it.  Fires only when the glow is
    bright, pupil-scale and genuinely warm (R > B) — otherwise returns None.
    """
    if flash_bgr is None or flash_bgr.ndim != 3:
        return None
    h, w = flash_bgr.shape[:2]
    win = int(rmax * _REDEYE_WIN_MULT)
    x0 = max(0, ax - win); x1 = min(w, ax + win)
    y0 = max(0, ay - win); y1 = min(h, ay + win)
    roi = flash_bgr[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return None
    b, g, r = cv2.split(roi)
    glow = cv2.GaussianBlur(np.clip(r - _REDEYE_BLUE_SUB * b, 0, 255), (15, 15), 0)
    peak = float(glow.max())
    if peak < _REDEYE_MIN_PEAK:
        return None
    _, _, _, mxloc = cv2.minMaxLoc(glow)
    gx, gy = mxloc[0] + x0, mxloc[1] + y0

    # blob at >60% of peak -> radius; reject pin-point speculars and huge spills
    _, mask = cv2.threshold(np.uint8(255 * glow / (peak + 1e-6)),
                            int(255 * 0.6), 255, cv2.THRESH_BINARY)
    area = int(cv2.countNonZero(mask))
    rr = float(np.sqrt(area / np.pi))
    if rr < rmin * 0.5 or rr > rmax * 1.6:
        return None
    # must be genuinely warm at the peak (retinal reflex), not a white specular
    if float(r[mxloc[1], mxloc[0]]) <= float(b[mxloc[1], mxloc[0]]):
        return None
    return gx, gy, rr, peak


def detect_pupil(img, live=False):
    """Detect the pupil: reflex anchor -> green-channel pupil blob -> radial fit.

    `img` is the colour (BGR) ambient image, or a grayscale image.  Returns a
    list of overlay dicts to draw — see _draw_overlays().

    Pipeline: the bright corneal reflex anchors which dark region is the pupil;
    the pupil blob is segmented on the GREEN channel (cleanest dark-pupil /
    bright-iris contrast under the violet light) and its centroid recenters the
    search (the reflex sits at the pupil *edge*, not its centre); radial rays
    from that centre give boundary points whose ellipse is robustly fit.  Rays
    into the dark occluder patch never brighten, so they drop out and a partially
    occluded boundary is tolerated.

    Normal mode: the single best plausible pupil ellipse (green).
    Debug mode (`SWIRSKI_DEBUG`): the fitted ellipse (yellow/red) plus the anchor
    and recentred centre.  Set PUPIL_DEBUG_STAGES to a list to also collect
    per-stage images.
    """
    global _LAST_ANCHOR, _LAST_PUPIL, _LAST_CONF
    _LAST_ANCHOR = None
    _LAST_PUPIL = None
    _LAST_CONF = 0.0
    if not CV2_AVAILABLE:
        return []
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        green = img[:, :, 1]                 # BGR -> green channel
    else:
        gray = img
        green = img
    _dbg("1_green", green)

    h, w = gray.shape[:2]
    rmin = _SW_MIN_R_FRAC * min(h, w)
    rmax = _SW_MAX_R_FRAC * min(h, w)

    # ── Stage 1: locate the corneal reflex (anchor) ───────────────────────
    reflex = _find_corneal_reflex(gray)
    if reflex is not None:
        _LAST_ANCHOR = (reflex[0], reflex[1])
    if PUPIL_DEBUG_STAGES is not None:
        vis = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
        if reflex is not None:
            cv2.circle(vis, (reflex[0], reflex[1]), max(6, reflex[2] * 2),
                       (0, 0, 255), 2)
        _dbg("2_reflex", vis)

    # ── Stage 2: inpaint the reflex on the green channel, then blur ────────
    reflex_mask = reflex[3] if reflex is not None else None
    work = _inpaint_reflections(green, reflex_mask)
    work = cv2.GaussianBlur(work, (7, 7), 0)
    _dbg("3_green_prepped", work)

    # ── Stage 3: seed centre — reflex anchor, else Haar coarse fallback ────
    if reflex is not None:
        ax, ay = reflex[0], reflex[1]
    else:
        coarse = _swirski_coarse(work, rmin, rmax)
        if coarse is None:
            if not live:
                print("No pupil detected (no reflex, no coarse seed).")
            return []
        ax, ay = coarse[0], coarse[1]

    # The reflex sits at the pupil *edge*, not its centre.  Iterate a robust
    # CIRCLE fit: cast rays from the current centre, fit a circle to the visible
    # arc, move the centre to the fit centre, re-cast.  A circle fit is stable on
    # a partial arc (unlike an ellipse fit), so this converges instead of running
    # away into the bright glow the way the ellipse recentre did.
    # start rays just beyond the reflex blob so its inpaint boundary is not read
    # as the pupil edge (which collapsed the fit to a tiny reflex-sized circle)
    r_start = int(reflex[2] * 1.7) if reflex is not None else 0
    cx, cy = ax, ay
    pts = np.empty((0, 2), np.float32)
    fit = None
    trail = [(cx, cy)]
    for _ in range(_RAD_RECENTRE_ITERS):
        angles, radii = _ray_edges(work, cx, cy, rmin, rmax, r_start)
        p = _points_from_radii(cx, cy, angles, radii)
        if len(p) < 5:
            break
        pts = p
        f = _fit_circle_robust(p)
        if f is None:
            break
        fit = f
        ncx, ncy = int(round(f[0])), int(round(f[1]))
        trail.append((ncx, ncy))
        moved = abs(ncx - cx) > 2 or abs(ncy - cy) > 2
        cx, cy = ncx, ncy
        if not moved:
            break

    if PUPIL_DEBUG_STAGES is not None:
        edgevis = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        for px, py in pts:
            cv2.circle(edgevis, (int(px), int(py)), 3, (0, 255, 255), -1)
        for i in range(1, len(trail)):
            cv2.line(edgevis, trail[i - 1], trail[i], (255, 255, 0), 1)
        cv2.circle(edgevis, (ax, ay), 5, (255, 0, 255), -1)    # anchor
        cv2.circle(edgevis, (cx, cy), 5, (0, 255, 0), -1)      # refined centre
        _dbg("4_recentre_edges", edgevis)

    if fit is None or len(pts) < 5:
        if not live:
            print(f"No pupil detected ({len(pts)} edge point(s), fit={fit}).")
        return []

    fcx, fcy, r = fit

    # ── Stage 5: concentric iris refinement ───────────────────────────────
    # The pupil and iris are concentric.  Detect the iris/limbus ridge in the
    # annulus beyond the pupil and jointly fit both rings about a shared centre;
    # the larger iris arc pins the centre far more stably than the short pupil
    # arc, which corrects the centre when the reflex sat on the iris or the
    # pupil's upper edge was lost.  Adopted only if the iris ring is well covered.
    iris = None
    ia, ir = _ray_ridges(work, int(fcx), int(fcy),
                         int(r * _IRIS_RP_MIN), int(r * _IRIS_SEARCH_MAX))
    iris_pts = _points_from_radii(fcx, fcy, ia, ir)
    iris_cover = len(iris_pts) / float(_RAD_N_ANGLES)
    if iris_cover >= _IRIS_MIN_COVER and len(pts) >= 5:
        con = _fit_concentric(pts, iris_pts)
        if con is not None:
            ncx, ncy, nrp, nri = con
            ratio = nri / nrp if nrp > 0 else 0.0
            moved = np.hypot(ncx - fcx, ncy - fcy)
            # Only trust the iris ring if the points actually lie ON a circle
            # (tight radial spread).  In these tight macro frames the limbus is
            # not a clean circle, so the "iris" ridges are scattered eyelid/glow
            # noise -> high residual -> rejected, and the pupil-only fit stands.
            d = np.hypot(iris_pts[:, 0] - ncx, iris_pts[:, 1] - ncy)
            iris_rms = float(np.std(d - nri)) if len(d) else 1e9
            if (_IRIS_RP_MIN <= ratio <= _IRIS_RP_MAX
                    and nrp >= rmin * 0.6 and moved < r
                    and iris_rms < _IRIS_MAX_RMS_FRAC * nri):
                fcx, fcy, r = ncx, ncy, nrp
                iris = (ncx, ncy, nri)
    if PUPIL_DEBUG_STAGES is not None:
        irisvis = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        for px, py in iris_pts:
            cv2.circle(irisvis, (int(px), int(py)), 3, (0, 165, 255), -1)
        if iris is not None:
            cv2.circle(irisvis, (int(iris[0]), int(iris[1])), int(iris[2]),
                       (0, 165, 255), 1)
        cv2.circle(irisvis, (int(fcx), int(fcy)), int(r), (0, 255, 0), 1)
        cv2.circle(irisvis, (int(fcx), int(fcy)), 4, (0, 255, 0), -1)
        _dbg(f"5_iris_concentric cov={iris_cover:.2f}", irisvis)

    # ── Stage 6: validate the fitted circle ───────────────────────────────
    angular_cover = len(pts) / float(_RAD_N_ANGLES)   # fraction of rays with an edge
    too_big = r > rmax * 1.5
    too_small = r < rmin * 0.6
    too_partial = angular_cover < _RAD_MIN_COVER
    bad = too_big or too_small or too_partial
    ell_full = ((float(fcx), float(fcy)), (float(2 * r), float(2 * r)), 0.0)

    # Stash for ambient/flash fusion: confidence = coverage x inlier tightness.
    d = np.hypot(pts[:, 0] - fcx, pts[:, 1] - fcy)
    inlier_frac = float(np.mean(np.abs(d - r) < 0.18 * r)) if len(pts) else 0.0
    _LAST_CONF = 0.0 if bad else angular_cover * inlier_frac
    _LAST_PUPIL = None if bad else (float(fcx), float(fcy), float(r))

    if not SWIRSKI_DEBUG:
        if bad:
            if not live:
                print("No pupil detected (implausible circle).")
            return []
        if not live:
            print(f"Pupil detected at ({int(fcx)}, {int(fcy)}), r={r:.0f}.")
        overlays = [{"ellipse": ell_full, "label": "", "color": (0, 255, 0)}]
    else:
        col = (0, 0, 255) if bad else (0, 255, 255)
        if not live:
            print(f"[debug] circle r={r:.0f} cover={angular_cover:.2f} from "
                  f"{len(pts)} pts at ({int(fcx)}, {int(fcy)})"
                  + ("  REJECTED" if bad else ""))
        overlays = [{
            "ellipse": ell_full,
            "label": f"r={r:.0f} cov={angular_cover:.2f}",
            "color": col,
        }, {
            "ellipse": ((float(ax), float(ay)), (16.0, 16.0), 0.0),
            "label": "anchor",
            "color": (255, 0, 255),
        }]

    if PUPIL_DEBUG_STAGES is not None:
        finalvis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _draw_overlays(finalvis, overlays)
        _dbg("9_final", finalvis)
    return overlays


def _draw_overlays(bgr, overlays):
    """Draw a list of candidate overlays (ellipse + centre dot + optional
    label) on a BGR image."""
    for ov in overlays:
        (ex, ey), (MA, ma), ang = ov["ellipse"]
        col = ov.get("color", (0, 255, 0))
        box = ((float(ex), float(ey)), (float(MA), float(ma)), float(ang))
        cv2.ellipse(bgr, box, col, 2)
        cv2.circle(bgr, (int(round(ex)), int(round(ey))), 3, col, -1)
        label = ov.get("label", "")
        if label:
            ty = int(ey) - int(max(MA, ma) / 2) - 6
            cv2.putText(bgr, label, (int(ex) - 45, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return bgr


def process_image(array, overlays=None):

    img = Image.fromarray(array)

    # Brightness
    img = ImageEnhance.Brightness(img).enhance(
        BRIGHTNESS
    )

    # Contrast
    img = ImageEnhance.Contrast(img).enhance(
        CONTRAST
    )

    # Rotate to match live feed orientation
    angle = _PIL_ROTATIONS[LIVE_ROTATION]
    if angle:
        img = img.rotate(angle, expand=True)

    # Draw the pupil candidate overlay(s) after rotation — they are already in
    # the rotated (display) orientation, the same as the rotated image, so no
    # coordinate transform is needed.
    if overlays and CV2_AVAILABLE:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _draw_overlays(arr, overlays)
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    return img


def _fuse_pupil(amb_pupil, amb_conf, redeye):
    """Fuse the ambient dark-disc fit and the flash red-eye into one decision.

    Both are in the same (rotated/display) orientation.  Strategy, given that
    the two cues are complementary (the red-eye is strong only when well aligned,
    the ambient dark-disc only when the pupil is unoccluded):

    - A strongly-trusted red-eye that *disagrees* with the ambient fit (or when
      there is no ambient fit) wins — this is the occluded case (e.g. 145755)
      where the ambient drifts onto the occluder boundary but the retroreflection
      sits squarely on the pupil.
    - Otherwise the ambient circle stands (it gives a proper boundary radius); a
      red-eye that *agrees* just corroborates it.
    - If neither is usable, returns None so the caller can flag the frame.

    Returns (cx, cy, r, source, confident) or None.
    """
    strong_redeye = redeye is not None and redeye[3] >= _REDEYE_TRUST
    if strong_redeye:
        rx, ry, rr, _ = redeye
        if amb_pupil is not None:
            ax, ay, ar = amb_pupil
            if np.hypot(rx - ax, ry - ay) <= ar:          # agree -> keep ambient
                return ax, ay, ar, "ambient+redeye", True
        return float(rx), float(ry), float(rr), "redeye", True   # override / sole
    if amb_pupil is not None:
        ax, ay, ar = amb_pupil
        return ax, ay, ar, "ambient", amb_conf >= _AMBIENT_MIN_CONF
    if redeye is not None:                                 # weak red-eye, last resort
        rx, ry, rr, _ = redeye
        return float(rx), float(ry), float(rr), "redeye?", False
    return None


def detect_and_annotate(ambient_array, flash_array):
    """Detect the pupil and annotate the flash image, fusing two cues.

    Detection runs in the display (rotated) orientation so its result lines up
    with the rotated flash photo that process_image() produces.  The ambient
    dark-disc fit (detect_pupil) and the flash red-eye retroreflection
    (detect_redeye) are fused by _fuse_pupil().  Returns
    (annotated_PIL_image, overlays).  Shared by capture_image() and the offline
    test harness so detection is identical in both.
    """
    overlays = []
    if CV2_AVAILABLE:
        rot = _CV2_ROTATIONS[LIVE_ROTATION]
        amb_bgr = cv2.cvtColor(ambient_array, cv2.COLOR_RGB2BGR)
        flash_bgr = cv2.cvtColor(flash_array, cv2.COLOR_RGB2BGR)
        if rot is not None:
            amb_bgr = cv2.rotate(amb_bgr, rot)
            flash_bgr = cv2.rotate(flash_bgr, rot)

        # Ambient dark-disc fit (uses the green channel — robust to skin/eye
        # colour because under violet light green suppresses the skin glow).
        detect_pupil(amb_bgr)
        amb_pupil, amb_conf, anchor = _LAST_PUPIL, _LAST_CONF, _LAST_ANCHOR

        # Flash red-eye, searched around the ambient reflex anchor.
        redeye = None
        if anchor is not None:
            h, w = amb_bgr.shape[:2]
            rmin = _SW_MIN_R_FRAC * min(h, w)
            rmax = _SW_MAX_R_FRAC * min(h, w)
            redeye = detect_redeye(flash_bgr, anchor[0], anchor[1], rmin, rmax)
        if PUPIL_DEBUG_STAGES is not None:
            rv = cv2.convertScaleAbs(flash_bgr, alpha=3.0)   # brighten the dark flash
            if anchor is not None:
                cv2.circle(rv, (anchor[0], anchor[1]), 5, (255, 0, 255), -1)
            if redeye is not None:
                cv2.circle(rv, (int(redeye[0]), int(redeye[1])),
                           max(6, int(redeye[2])), (0, 255, 255), 2)
            tag = f"peak={redeye[3]:.0f}" if redeye is not None else "none"
            _dbg(f"7_flash_redeye {tag}", rv)

        fused = _fuse_pupil(amb_pupil, amb_conf, redeye)
        if fused is not None:
            cx, cy, r, source, confident = fused
            ell = ((cx, cy), (2 * r, 2 * r), 0.0)
            colour = (0, 255, 0) if confident else (0, 165, 255)   # amber = low-conf
            label = source if SWIRSKI_DEBUG else ""
            if not confident:
                label = (label + " low-conf").strip()
            overlays = [{"ellipse": ell, "label": label, "color": colour}]
        if PUPIL_DEBUG_STAGES is not None:
            fv = cv2.convertScaleAbs(flash_bgr, alpha=3.0)
            _draw_overlays(fv, overlays)
            _dbg(f"8_fused {fused[3] if fused else 'NONE'}", fv)
    img = process_image(flash_array, overlays)
    return img, overlays


def transfer_capture(ambient_array, flash_array):
    """Stage the raw capture pair locally and POST it to the dev machine.

    Writes LOCAL_STAGING_DIR/capture_<timestamp>/{ambient.jpg, flash.jpg,
    meta.json}, then uploads each file via HTTP POST to receiver.py on the dev
    machine.  The raw images are saved (no overlay) so the test harness can
    re-run detection cleanly; meta.json records LIVE_ROTATION so the test
    rotates exactly as the Pi did.

    Any failure is reported on the Pi but never raised — a transfer problem
    must not interrupt capture or streaming.
    """
    if not TRANSFER_ENABLED:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(LOCAL_STAGING_DIR, f"capture_{stamp}")

    # ── stage the pair locally ──
    try:
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(ambient_array).save(os.path.join(folder, "ambient.jpg"))
        Image.fromarray(flash_array).save(os.path.join(folder, "flash.jpg"))
        meta = {
            "timestamp":     stamp,
            "live_rotation": LIVE_ROTATION,
            "exposure":      EXPOSURE_TIME,
            "flash_gain":    FLASH_GAIN,
            "red_gain":      RED_GAIN,
            "blue_gain":     BLUE_GAIN,
            "brightness":    BRIGHTNESS,
            "contrast":      CONTRAST,
            "swirski_debug": SWIRSKI_DEBUG,
        }
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except OSError as e:
        print(f"[TRANSFER ERROR] could not stage capture: {e}")
        return

    # ── upload each file to receiver.py on the dev machine ──
    folder_name = f"capture_{stamp}"
    base_url = f"http://{REMOTE_HOST}:{REMOTE_PORT}/upload/{folder_name}"
    failed = False
    for fname in ("ambient.jpg", "flash.jpg", "meta.json"):
        path = os.path.join(folder, fname)
        try:
            with open(path, "rb") as f:
                data = f.read()
            req = urllib.request.Request(
                f"{base_url}/{fname}",
                data=data, method="POST",
                headers={"Content-Type": "application/octet-stream"},
            )
            with urllib.request.urlopen(req, timeout=TRANSFER_TIMEOUT) as resp:
                if resp.status != 200:
                    print(f"[TRANSFER ERROR] {fname}: HTTP {resp.status}")
                    failed = True
        except (urllib.error.URLError, OSError) as e:
            # URLError covers HTTPError, connection refused, DNS, timeout.
            print(f"[TRANSFER ERROR] {fname}: {e}")
            failed = True

    if not failed:
        print(f"Transferred {folder_name} to {REMOTE_HOST}.")


def _drain_frames(picam2, n=FLUSH_FRAMES):
    """Discard the next n frames from the camera.

    picam2.capture_array() may return a frame that was already in the buffer,
    exposed under the *previous* state — wrong gain, or LED not yet on.  After
    flipping a GPIO LED or changing the gain, drain a couple of frames so the
    following capture is a fresh exposure under the new state.  Without this,
    the saved flash photo is often dark.
    """
    for _ in range(n):
        picam2.capture_array()


def capture_image(picam2):
    """Capture a flash + ambient image pair and detect the pupil.

    1.  GPIO 17 (flash) lit → capture the retina-flash image first, while the
        pupil is still maximally dilated.
    2.  GPIO 27 (ambient) lit → capture the ambient image.  The pupil reads as
        a dark disc here, suitable for Swirski detection.
    3.  Detect the pupil in the ambient image with the Swirski method and draw
        the resulting ellipse onto the flash photo.

    The raw ambient image is saved to AMBIENT_PATH; the annotated flash photo
    to PHOTO_PATH.
    """
    print("Capturing flash + ambient pair...")

    apply_camera_settings(picam2, FLASH_GAIN)
    time.sleep(0.02)
    _drain_frames(picam2)              # let the new gain take effect

    # ───────── FLASH IMAGE (GPIO 17) — first, pupil still dilated ─────────
    flash_on()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed before LED on
    flash_array = picam2.capture_array()
    time.sleep(FLASH_DURATION)
    flash_off()
    time.sleep(FLASH_POST_DELAY)

    # ───────── AMBIENT IMAGE (GPIO 27) — second ─────────
    ambient_on()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed before LED on
    ambient_array = picam2.capture_array()
    ambient_off()

    # Restore live settings
    apply_camera_settings(picam2, LIVE_GAIN)

    # Save the raw ambient image for inspection
    Image.fromarray(ambient_array).save(AMBIENT_PATH)

    # ───────── IMAGE TRANSFER ─────────
    # Push the raw pair to the dev machine for offline detection testing.
    transfer_capture(ambient_array, flash_array)

    # ───────── PUPIL DETECTION (Orlosky) ─────────
    if CV2_AVAILABLE:
        img, overlays = detect_and_annotate(ambient_array, flash_array)
        if not overlays:
            print("No pupil candidates detected.")
    else:
        print("cv2 not available - pupil detection skipped.")
        img = process_image(flash_array, [])

    # Save the annotated flash photo
    img.save(PHOTO_PATH)

    print("Captured.")


def streaming_mode(picam2):
    """Live video feed.

    SPACE (hold) / r — flash LED (GPIO 17), hold / lock
    a               — toggle ambient LED (GPIO 27)
    p               — toggle live Orlosky pupil detection
    ←/→             — rotate the live feed
    ENTER or e      — capture an ambient + flash still
    s               — exit streaming
    """

    global LIVE_ROTATION

    if not CV2_AVAILABLE:
        print("cv2 not available - cannot show live feed.")
        return

    print("\nStreaming mode ON")
    print("SPACE=flash | r=lock | a=ambient | p=pupil-detect | "
          "←/→=rotate | ENTER/e=capture | s=exit\n")

    apply_camera_settings(picam2, FLASH_GAIN)

    WINDOW_NAME = "Live Feed"
    SPACE_TIMEOUT = 0.15  # seconds before spacebar is considered released

    # Portrait dimensions (rotated 90°): 960w × 1280h → half res 480×640
    DISPLAY_W = 480
    DISPLAY_H = 640

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    # Capture and display the first frame before the LED is usable
    array = picam2.capture_array()
    frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    rot = _CV2_ROTATIONS[LIVE_ROTATION]
    if rot is not None:
        frame = cv2.rotate(frame, rot)
    if LIVE_ROTATION % 2 == 1:
        cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_W, DISPLAY_H)))
    else:
        cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_H, DISPLAY_W)))
    # Wait long enough for the window to fully render
    cv2.waitKey(500)

    last_space_time = 0.0
    led_on = False
    lock_on = False
    ambient_led_on = False
    swirski_live_on = SWIRSKI_LIVE_DEFAULT
    _detect_counter = 0
    _overlay_cache = None

    try:

        while True:

            array = picam2.capture_array()
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            rot = _CV2_ROTATIONS[LIVE_ROTATION]
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            # Portrait vs landscape: swap display dims if needed
            if LIVE_ROTATION % 2 == 1:  # 90° or 270°
                disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            else:
                disp = cv2.resize(frame, (DISPLAY_H, DISPLAY_W))

            # Live Swirski pupil detection — recompute every N frames
            if swirski_live_on:
                _detect_counter += 1
                if (_overlay_cache is None
                        or _detect_counter % _LIVE_DETECT_SKIP == 0):
                    gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
                    _overlay_cache = detect_pupil(gray, live=True)
                if _overlay_cache:
                    disp = _draw_overlays(disp.copy(), _overlay_cache)

            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKeyEx(1)
            now = time.time()
            space_held = (now - last_space_time <= SPACE_TIMEOUT)

            # Also exit if the window is closed via the X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('s'):
                break

            # Capture still: ENTER or e
            if key in (13, 10) or key == ord('e'):
                flash_off()
                led_on = False
                last_space_time = 0.0
                capture_image(picam2)
                # capture_image manages both LEDs itself; restore the ambient
                # LED to whatever state streaming had it in
                if ambient_led_on:
                    ambient_on()
                apply_camera_settings(picam2, FLASH_GAIN)

            # Track spacebar hold via auto-repeat
            elif key == 32:
                last_space_time = now
                space_held = True

            # r toggles flash LED lock
            elif key == ord('r'):
                lock_on = not lock_on

            # a toggles the ambient LED (GPIO 27)
            elif key == ord('a'):
                ambient_led_on = not ambient_led_on
                if ambient_led_on:
                    ambient_on()
                else:
                    ambient_off()
                print(f"Ambient LED: {'ON' if ambient_led_on else 'OFF'}")

            # p toggles live Swirski pupil detection
            elif key == ord('p'):
                swirski_live_on = not swirski_live_on
                _overlay_cache = None
                print(f"Live pupil detection: "
                      f"{'ON' if swirski_live_on else 'OFF'}")

            # Arrow keys: left=CCW step, right=CW step
            elif key == 65361:  # left arrow
                LIVE_ROTATION = (LIVE_ROTATION - 1) % 4
                print(f"Rotation: {LIVE_ROTATION * 90}°")
            elif key == 65363:  # right arrow
                LIVE_ROTATION = (LIVE_ROTATION + 1) % 4
                print(f"Rotation: {LIVE_ROTATION * 90}°")

            should_be_on = lock_on or space_held
            if should_be_on and not led_on:
                flash_on()
                led_on = True
            elif not should_be_on and led_on:
                flash_off()
                led_on = False

    finally:

        flash_off()
        ambient_off()
        apply_camera_settings(picam2, LIVE_GAIN)
        cv2.destroyWindow(WINDOW_NAME)
        print("Streaming mode OFF.\n")


def handle_parameter_command(cmd, picam2):

    global FLASH_GAIN
    global LIVE_GAIN
    global EXPOSURE_TIME

    global RED_GAIN
    global BLUE_GAIN

    global FLASH_PRE_DELAY
    global FLASH_POST_DELAY
    global FLASH_DURATION

    global BRIGHTNESS
    global CONTRAST
    global DETECT_EXCLUDE_BOTTOM
    global SWIRSKI_DEBUG

    parts = cmd.split()

    if len(parts) != 2:
        print("Invalid command.")
        return

    param = parts[0].lower()
    value = parts[1]

    try:

        # Camera

        if param == "exposure":
            EXPOSURE_TIME = int(value)

        elif param == "flash_gain":
            FLASH_GAIN = float(value)

        elif param == "live_gain":

            LIVE_GAIN = float(value)

            apply_camera_settings(
                picam2,
                LIVE_GAIN
            )

        # White balance

        elif param == "red_gain":

            RED_GAIN = float(value)

            apply_camera_settings(
                picam2,
                LIVE_GAIN
            )

        elif param == "blue_gain":

            BLUE_GAIN = float(value)

            apply_camera_settings(
                picam2,
                LIVE_GAIN
            )

        # Flash timing

        elif param == "pre_delay":
            FLASH_PRE_DELAY = float(value)

        elif param == "post_delay":
            FLASH_POST_DELAY = float(value)

        elif param == "flash_duration":
            FLASH_DURATION = float(value)

        # Image processing

        elif param == "brightness":
            BRIGHTNESS = float(value)

        elif param == "contrast":
            CONTRAST = float(value)

        # Pupil detection

        elif param == "exclude_bottom":
            DETECT_EXCLUDE_BOTTOM = min(0.9, max(0.0, float(value)))

        elif param == "debug":
            SWIRSKI_DEBUG = bool(int(value))

        else:
            print("Unknown parameter.")
            return

        print_settings()

    except ValueError:
        print("Invalid numeric value.")


# ───────── MAIN ─────────

def main():

    print("\n╔══════════════════════════════════════╗")
    print("║   Retina Flash Photography           ║")
    print("╚══════════════════════════════════════╝")

    setup_gpio()

    # ───────── CAMERA SETUP ─────────

    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={
            "size": (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            )
        }
    )

    picam2.configure(config)

    apply_camera_settings(
        picam2,
        LIVE_GAIN
    )

    picam2.start()

    time.sleep(2)

    # ───────── PREOPEN IMAGE VIEWER ─────────

    blank = Image.new(
        "RGB",
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        (0, 0, 0)
    )

    blank.save(PHOTO_PATH)

    os.system(
        f"xdg-open '{PHOTO_PATH}' >/dev/null 2>&1 &"
    )

    print("\nViewer opened.")
    print("Many viewers auto-refresh.\n")

    time.sleep(5)

    print_settings()

    # ───────── BOOT STRAIGHT INTO STREAMING MODE ─────────
    # The program always starts streaming; exiting streaming (s) drops into the
    # normal command loop below, from which streaming can be re-entered.

    try:

        streaming_mode(picam2)
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        except Exception:
            pass

        print("ENTER = capture (ambient + flash, with pupil detection)")
        print("s     = streaming mode (SPACE=flash, r=lock, a=ambient, "
              "p=pupil-detect, e/ENTER=capture)")
        print("q     = quit\n")

        # ───────── MAIN LOOP ─────────

        while True:

            cmd = input("> ").strip()

            # Quit

            if cmd == "q":
                break

            # Capture

            elif cmd == "":
                capture_image(picam2)

            # Streaming mode

            elif cmd == "s":
                streaming_mode(picam2)
                # Flush any keystrokes that leaked from the cv2 window into stdin
                try:
                    import termios
                    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                except Exception:
                    pass
                print("Normal mode. ENTER=capture | s=streaming | q=quit\n")

            # Parameter command

            else:
                handle_parameter_command(
                    cmd,
                    picam2
                )

    except KeyboardInterrupt:

        print("\nInterrupted.")

    finally:

        picam2.stop()

        cleanup_gpio()

        print("Done.")


if __name__ == "__main__":
    main()
