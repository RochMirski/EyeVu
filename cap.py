#!/usr/bin/env python3
"""
Retina Flash Photography
Raspberry Pi Zero + Camera Module V2

CONTROLS
────────
ENTER               Capture photo (ambient + flash pair)
s + ENTER           Streaming mode
q + ENTER           Quit

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

from picamera2 import Picamera2
import time
import os
import sys
from PIL import Image, ImageEnhance
import numpy as np

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

# ── Swirski detector parameters (commented out — using Orlosky instead) ──
# _SW_MIN_R_FRAC = 0.03 ; _SW_MAX_R_FRAC = 0.13 ; _SW_N_RADII = 6
# _SW_COARSE_DOWNSCALE = 4 ; _SW_RANSAC_ITERS = 40 ; _SW_RANSAC_ITERS_LIVE = 12
# _SW_INLIER_DIST = 2.0 ; _SW_MIN_SOLIDITY = 0.80 ; _SW_MAX_AXIS_RATIO = 2.2

# ── Orlosky detector parameters ──
_ORL_THRESHOLD_OFFSET = 15       # binary threshold = darkest pixel value + this
_ORL_MASK_FRAC = 0.5             # ROI square side, as a fraction of min(h, w)
_ORL_MIN_AREA_FRAC = 0.003       # min contour area, as a fraction of frame area
_ORL_MAX_RATIO = 3.0             # max contour bounding-box aspect ratio
_ORL_DARKEST_WIN = 20            # window size for the darkest-region search
_ORL_DILATE_KERNEL = 5           # dilation kernel side
_ORL_DILATE_ITERS = 2            # dilation iterations

# Flash-LED-cover exclusion — the LED's physical cover intrudes into the frame
# as a large near-black blob.  It is detected dynamically (a large,
# border-touching, very dark region); DETECT_EXCLUDE_BOTTOM is an optional
# manual fallback band (see handle_parameter_command's `exclude_bottom`).
_LED_DARK_THRESH = 30            # intensity below which a pixel is "near-black"
_LED_MIN_AREA_FRAC = 0.05        # min blob area (fraction of frame) to qualify
DETECT_EXCLUDE_BOTTOM = 0.0      # manual fallback: blank this bottom fraction

# Live mode: only recompute the detection every N frames
_LIVE_DETECT_SKIP = 5

# Debug — annotate EVERY detected pupil candidate (no plausibility filtering)
# instead of a single filtered result.  Toggle at runtime with `debug 0` / `debug 1`.
SWIRSKI_DEBUG = True


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
    # grow slightly — the out-of-focus cover has a soft, fuzzy edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, k)


def _apply_exclusions(gray):
    """Return a copy of `gray` with the flash-LED cover (detected dynamically)
    and the optional manual bottom band blanked to white (255), so they cannot
    be mistaken for a dark pupil."""
    h = gray.shape[0]
    g = gray.copy()
    cover = _find_led_cover_mask(g)
    if cover is not None:
        g[cover > 0] = 255
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
    darkest-region search → relative threshold → ROI mask → dilate → contours.

    Returns (contours, darkest_point)."""
    darkest = _orlosky_darkest_area(gray)
    if darkest is None:
        return [], None

    h, w = gray.shape[:2]
    dval = int(gray[darkest[1], darkest[0]])
    thresh = dval + _ORL_THRESHOLD_OFFSET

    # dark pupil → white
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    # keep only a square ROI around the darkest point
    binary = _mask_outside_square(binary, darkest,
                                  int(_ORL_MASK_FRAC * min(h, w)))
    # dilate to close the pupil blob
    kernel = np.ones((_ORL_DILATE_KERNEL, _ORL_DILATE_KERNEL), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=_ORL_DILATE_ITERS)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours, darkest


def detect_pupil(gray, live=False):
    """Detect the pupil with the Orlosky method.

    Returns a list of overlay dicts to draw — see _draw_overlays().
    Normal mode: the single largest plausible pupil contour (green).
    Debug mode (`SWIRSKI_DEBUG`): every contour found, unfiltered, each
    labelled with its area and aspect ratio (yellow), plus the darkest point.
    """
    if not CV2_AVAILABLE:
        return []
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Blank the flash-LED cover so the darkest-area search ignores it
    gray = _apply_exclusions(gray)
    h, w = gray.shape[:2]
    min_area = _ORL_MIN_AREA_FRAC * h * w

    contours, darkest = _orlosky_contours(gray)

    if not SWIRSKI_DEBUG:
        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or len(c) < 5:
                continue
            _x, _y, bw, bh = cv2.boundingRect(c)
            if min(bw, bh) == 0:
                continue
            if max(bw / bh, bh / bw) > _ORL_MAX_RATIO:
                continue
            if area > best_area:
                best_area = area
                best = c
        if best is None:
            return []
        try:
            ell = cv2.fitEllipse(best)
        except cv2.error:
            return []
        (ex, ey), _, _ = ell
        if not live:
            print(f"Pupil detected at ({int(ex)}, {int(ey)}).")
        return [{"ellipse": ell, "label": "", "color": (0, 255, 0)}]

    # ── debug: annotate every contour, discard nothing ────────────────────
    overlays = []
    n = 0
    for c in contours:
        if len(c) < 5:                       # cv2.fitEllipse needs >= 5 points
            continue
        try:
            ell = cv2.fitEllipse(c)
        except cv2.error:
            continue
        n += 1
        area = cv2.contourArea(c)
        _x, _y, bw, bh = cv2.boundingRect(c)
        ratio = max(bw / bh, bh / bw) if min(bw, bh) > 0 else 0.0
        overlays.append({
            "ellipse": ell,
            "label": f"{n} a={int(area)} ar={ratio:.1f}",
            "color": (0, 255, 255),          # yellow — candidate contour
        })
    if not live:
        print(f"[debug] {n} pupil candidate(s) (Orlosky).")
    if darkest is not None:
        overlays.append({
            "ellipse": ((float(darkest[0]), float(darkest[1])),
                        (16.0, 16.0), 0.0),
            "label": "darkest",
            "color": (255, 0, 255),          # magenta — darkest point
        })
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

    # ───────── FLASH IMAGE (GPIO 17) — first, pupil still dilated ─────────
    flash_on()
    time.sleep(FLASH_PRE_DELAY)
    flash_array = picam2.capture_array()
    time.sleep(FLASH_DURATION)
    flash_off()
    time.sleep(FLASH_POST_DELAY)

    # ───────── AMBIENT IMAGE (GPIO 27) — second ─────────
    ambient_on()
    time.sleep(FLASH_PRE_DELAY)
    ambient_array = picam2.capture_array()
    ambient_off()

    # Restore live settings
    apply_camera_settings(picam2, LIVE_GAIN)

    # Save the raw ambient image for inspection
    Image.fromarray(ambient_array).save(AMBIENT_PATH)

    # ───────── PUPIL DETECTION (Swirski) ─────────
    # Detection runs in the display (rotated) orientation so its result lines
    # up with the rotated flash photo that process_image() produces.
    overlays = []
    if CV2_AVAILABLE:
        amb_bgr = cv2.cvtColor(ambient_array, cv2.COLOR_RGB2BGR)
        rot = _CV2_ROTATIONS[LIVE_ROTATION]
        if rot is not None:
            amb_bgr = cv2.rotate(amb_bgr, rot)
        gray = cv2.cvtColor(amb_bgr, cv2.COLOR_BGR2GRAY)
        overlays = detect_pupil(gray)
        if not overlays:
            print("No pupil candidates detected.")
    else:
        print("cv2 not available - pupil detection skipped.")

    # Process the flash image, drawing the pupil candidate overlay(s)
    img = process_image(flash_array, overlays)
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

    print("ENTER = capture (ambient + flash, with pupil detection)")
    print("s     = streaming mode (SPACE=flash, r=lock, a=ambient, "
          "p=pupil-detect, e/ENTER=capture)")
    print("q     = quit\n")

    # ───────── MAIN LOOP ─────────

    try:

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
