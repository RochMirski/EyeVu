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
a                   Toggle ambient LED (GPIO 22/27/23/6/26/16) on/off
p                   Toggle live Orlosky pupil detection
t                   Toggle PC image transfer (HTTP-POST to receiver.py)
f                   Flip to the other eye (cover top<->bottom, 180°)
m                   Toggle RITnet: every capture <-> only when ridge weak
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
from dataclasses import dataclass
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

# ML pupil-segmentation backend.  Prefer torch RITnet (PC / aarch64); fall back to
# ncnn RITnet on boards where torch has no build (ARMv6, e.g. a Pi Zero W).  Both
# expose the same API: available(), locate(...), tangent_crop_center, RitnetResult.
_ML_BACKEND = None
ML_BACKEND_NAME = "none"
for _bk_name, _bk_label in (("ritnet_infer", "torch RITnet"),
                            ("ncnn_infer", "ncnn RITnet")):
    try:
        print(f"Checking {_bk_label} availability...")
        _bk = __import__(_bk_name)
        if _bk.available():
            _ML_BACKEND = _bk
            ML_BACKEND_NAME = _bk_label
            break
    except Exception as e:                       # noqa: BLE001
        print(f"  {_bk_name} not available ({e}).")
RITNET_AVAILABLE = _ML_BACKEND is not None

# ───────── CONFIG ─────────

LED_PIN   = 17     # flash LED   — retina retroreflection capture
# Ambient light = six LEDs driven together (diffuse light for pupil detection).
AMBIENT_PINS = [22, 27, 23, 6, 26, 16]

PHOTO_PATH   = "/tmp/retina_preview.jpg"   # annotated flash photo
AMBIENT_PATH = "/tmp/retina_ambient.jpg"   # raw ambient image (Swirski input)

# Open an external image viewer (xdg-open) on PHOTO_PATH at startup.  OFF by
# default — the alignment-guidance cv2 window is the live view now.  The annotated
# flash is still saved to PHOTO_PATH and the raw high-res frames are still
# transferred; only the auto-opened preview window is suppressed.  Toggle at
# runtime with `preview 1` / `preview 0`.
PREVIEW_VIEWER = False

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

    global GPIO_AVAILABLE
    if not GPIO_AVAILABLE:
        return

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        # Pass initial= so the rpi-lgpio backend (Raspberry Pi OS Bookworm/Trixie)
        # does NOT gpio_read to preserve the pin state on setup — that read raises
        # lgpio "GPIO not allocated" and used to crash the whole program at boot
        # (before streaming), which looked like "streaming immediately closes".
        GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        for _p in AMBIENT_PINS:
            GPIO.setup(_p, GPIO.OUT, initial=GPIO.LOW)
    except Exception as e:                       # noqa: BLE001 — never block streaming on GPIO
        GPIO_AVAILABLE = False
        print(f"Warning: GPIO setup failed ({e}); LEDs disabled, "
              "camera/streaming still work.")


def cleanup_gpio():

    if not GPIO_AVAILABLE:
        return

    GPIO.output(LED_PIN, GPIO.LOW)
    for _p in AMBIENT_PINS:
        GPIO.output(_p, GPIO.LOW)

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
    """Turn on the ambient LEDs (GPIO 22/27/23/6/26/16; diffuse light for pupil detection)."""
    if GPIO_AVAILABLE:
        for _p in AMBIENT_PINS:
            GPIO.output(_p, GPIO.HIGH)


def ambient_off():
    """Turn off the ambient LEDs (GPIO 22/27/23/6/26/16)."""
    if GPIO_AVAILABLE:
        for _p in AMBIENT_PINS:
            GPIO.output(_p, GPIO.LOW)


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

    print()

    print(f"transfer           = {'ON' if TRANSFER_ENABLED else 'OFF'}")
    print(f"preview            = {'ON' if PREVIEW_VIEWER else 'OFF'}")
    print(f"pi_detect          = {'ON' if PI_DETECT else 'OFF'}")
    print(f"use_cover          = {'ON' if USE_COVER_CALIB else 'OFF'}")
    print(f"cover_side         = {COVER_SIDE}  (rotation {LIVE_ROTATION * 90}°)")
    print(f"ritnet             = {'every capture' if RITNET_ALWAYS else f'when ridge<{RITNET_CONF_GATE:.2f}'}")

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

# ── LED-cover CALIBRATION ──
# The dynamic cover finder above is unreliable when the cover fuses with the
# eye-socket shadow.  A one-off calibration capture — the LED cover with NO eye
# in place (ideally the cover against a white background) — locates the cover
# once, as a fixed mask, so detection can treat that region as known-occluded
# (ray edges falling in it are the cover boundary, not the pupil, and are
# dropped).  The mask is stored in the display orientation it was calibrated in
# (alongside that LIVE_ROTATION in a .rot sidecar); load_cover_mask rotates it to
# the current orientation so it stays aligned after an eye/orientation flip.
# Capture it on the Pi with the `calibrate` command.
_HERE = os.path.dirname(os.path.abspath(__file__))
COVER_CALIB_DIR = os.path.join(_HERE, "calibration")
COVER_CALIB_MASK_PATH = os.path.join(COVER_CALIB_DIR, "led_cover_mask.png")
COVER_CALIB_IMAGE_PATH = os.path.join(COVER_CALIB_DIR, "led_cover_calib.jpg")
COVER_CALIB_ROT_PATH = os.path.join(COVER_CALIB_DIR, "led_cover_mask.rot")
USE_COVER_CALIB = True           # apply the calibrated cover mask if present
_COVER_CALIB_DARK = 20           # ALMOST-complete-black ceiling: the cover blocks
                                 # the LED so it reads ~0.  Kept low so only the
                                 # truly-black cover qualifies — a higher value let
                                 # dim background in and over-grew the mask
_COVER_CALIB_MERGE = 25          # close gaps up to ~this (px) so the cover's
                                 # pointed/curved-triangle blob and its nearby
                                 # "extra bits" on the same side join into one
_COVER_CALIB_DILATE = 9          # grow the final mask a little (soft fuzzy edge)
_COVER_SMOOTH = 15               # close kernel (px) — bridge small edge notches
_COVER_SMOOTH_OPEN = 45          # open kernel (px) — shave protrusions/peaks that
                                 # stick out from the main body (bigger = more
                                 # conservative, hugs the main blob more tightly)
_COVER_SMOOTH_EPS = 0.012        # contour-fit tolerance (fraction of perimeter) —
                                 # higher simplifies away small spikes
_COVER_MASK_CACHE = None         # lazily-loaded (mask_array, shape) cache
_COVER_CALIB_ROT_CACHE = None    # LIVE_ROTATION the stored mask was calibrated at

# Cover side: which edge of the working (display) image the LED cover intrudes
# from — "top" for the left eye (default), "bottom" for the right eye.  The two
# eyes differ by a 180° rig rotation, so switching eye just adds 180° to
# LIVE_ROTATION; the cover mask is rotated to match in load_cover_mask, and the
# detection geometry auto-detects the cover side from the mask, so it adapts
# either way.  Set at startup (prompt) and toggled live with `f` / `cover_side`.
COVER_SIDE = "top"

# Live mode: only recompute the detection every N frames
_LIVE_DETECT_SKIP = 5

# Debug — annotate EVERY detected pupil candidate (no plausibility filtering)
# instead of a single filtered result.  Toggle at runtime with `debug 0` / `debug 1`.
SWIRSKI_DEBUG = True

# Run pupil detection on the Pi at capture time.  OFF by default for now: the Pi
# only captures + transfers the raw pair, and detection is run on the dev machine
# by test_pupil_detection.py.  Toggle at runtime with `pi_detect 1` / `pi_detect 0`.
PI_DETECT = False

# Build the LED-cover mask on the Pi.  OFF by default: the `c` key / calibrate just
# captures the calibration IMAGE and uploads it, and the dev machine (receiver.py)
# builds the mask.  Toggle at runtime with `pi_build_calib 1` / `pi_build_calib 0`.
PI_BUILD_CALIB = False

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

# Alignment-guidance state for the LIVE feed (guidance is done live now, not at
# capture).  The tracker carries the previous frame's offset for relative phrasing;
# the prior is the last pupil centre (display orientation) that seeds the next search.
_GUIDANCE_TRACKER = None
_GUIDANCE_PRIOR = None
GUIDANCE_TARGET_MODE = "cover_top_mid" # drive the pupil to the LED-cover inner edge
                                       # ("centre" = image centre; toggle: target_mode)
GUIDANCE_USE_ML = True                 # let coarse_locate try RITnet (if present)
GUIDANCE_WINDOW = "Alignment Guidance" # Pi cv2 window the annotated capture is shown in

# Auto-capture: when live guidance (g) reports the pupil CENTRED and auto-capture is
# armed, turn the ambient LEDs off, wait AUTOCAP_DILATE s for the pupil to dilate,
# then run the standard flash procedure.  After a take it DISARMS — the operator
# re-authorises the next search-and-take with `k`.
AUTOCAP_DILATE = 1.5                    # seconds to dilate (ambient off) before flash
AUTOCAP_COOLDOWN = 3.0                  # min seconds between auto-captures (backstop)
AUTOCAP_CENTRED_FRAMES = 2             # consecutive centred detections before firing

# Show the red-pixel (red-eye) extraction on the Pi AFTER a capture's transfer.
REDEYE_PREVIEW = True

# RITnet is slow on the Pi (ncnn on ARMv6 ~ minutes/frame), so by default it runs
# only when the cheap ridge/red-eye cue is weak (missing or below RITNET_CONF_GATE)
# — the "two-tier" approach.  Flip RITNET_ALWAYS (command `ritnet_always 1` / key
# `m`) to run it on every capture.  The PC receiver sets RITNET_ALWAYS=True so its
# capture sorting always has both detectors.
RITNET_ALWAYS = False
RITNET_CONF_GATE = 0.35                 # run RITnet when ridge confidence < this


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


# ── LED-cover calibration: build / save / load a fixed cover mask ──

def _smooth_fill_cover(mask):
    """Fill holes inside the cover mask and smooth its outer edge to one contour.

    Morphologically close (bridge edge notches) then open (shave protrusions) to
    de-jag the boundary, take the largest external contour, fit it
    (approxPolyDP, `_COVER_SMOOTH_EPS`) and redraw it FILLED — so any black
    patches enclosed by the cover become part of the mask and the edge is a clean
    smooth contour rather than the ragged threshold output.
    """
    if mask is None or int(cv2.countNonZero(mask)) == 0:
        return mask
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (_COVER_SMOOTH, _COVER_SMOOTH))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (_COVER_SMOOTH_OPEN, _COVER_SMOOTH_OPEN))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)     # bridge small notches
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)         # shave sticking-out peaks
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, _COVER_SMOOTH_EPS * cv2.arcLength(c, True), True)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, thickness=-1)   # filled -> holes gone
    return out


def build_cover_mask(bgr_or_gray):
    """Build the LED-cover mask from a calibration frame (cover, no eye).

    The cover blocks the LED entirely, so it is an **almost completely black**
    region intruding from one frame edge — a pointed / curved-triangle blob with
    a few small "extra bits" on the same side.  We threshold only that near-black
    level (so dim background is NOT picked up), morphologically close so the
    triangle and its nearby bits merge, then keep the edge-touching component(s).
    Returns a uint8 mask (255 = cover) in the input orientation/size, or None.
    """
    if not CV2_AVAILABLE:
        return None
    gray = (bgr_or_gray if bgr_or_gray.ndim == 2
            else cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)        # tame background speckle
    h, w = gray.shape[:2]

    # Almost-complete-black only — this is the key against over-sensitivity.
    _, dark = cv2.threshold(gray, _COVER_CALIB_DARK, 255, cv2.THRESH_BINARY_INV)
    # Merge the triangle with its nearby extra bits before labelling.
    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (_COVER_CALIB_MERGE, _COVER_CALIB_MERGE))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, km)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return None

    # Keep the largest near-black blob that touches a frame edge — the cover
    # intrudes from an edge; stray dark specks elsewhere do not (and are small).
    min_area = _LED_MIN_AREA_FRAC * h * w
    best_lbl, best_area = 0, 0
    for lbl in range(1, n):
        x = stats[lbl, cv2.CC_STAT_LEFT]; y = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]; bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        touches = (x == 0 or y == 0 or x + bw == w or y + bh == h)
        if touches and area > min_area and area > best_area:
            best_lbl, best_area = lbl, area
    if best_lbl == 0:
        return None

    mask = np.where(labels == best_lbl, 255, 0).astype(np.uint8)
    mask = _smooth_fill_cover(mask)                  # fill holes + smooth the edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (_COVER_CALIB_DILATE, _COVER_CALIB_DILATE))
    return cv2.dilate(mask, k)                       # grow the soft fuzzy edge


def save_cover_calibration(mask, src_bgr=None):
    """Persist the cover mask (and optionally the source frame) to disk.

    Also records the LIVE_ROTATION the mask was calibrated at, so load_cover_mask
    can rotate it back into whatever orientation is active later (eye flip).
    """
    if not CV2_AVAILABLE or mask is None:
        return False
    os.makedirs(COVER_CALIB_DIR, exist_ok=True)
    cv2.imwrite(COVER_CALIB_MASK_PATH, mask)
    if src_bgr is not None:
        cv2.imwrite(COVER_CALIB_IMAGE_PATH, src_bgr)
    try:
        with open(COVER_CALIB_ROT_PATH, "w") as fh:
            fh.write(str(LIVE_ROTATION))
    except OSError:
        pass
    global _COVER_MASK_CACHE, _COVER_CALIB_ROT_CACHE
    _COVER_MASK_CACHE = None          # invalidate cache so the next load re-reads
    _COVER_CALIB_ROT_CACHE = None
    return True


def _cover_calib_rotation():
    """LIVE_ROTATION the stored mask was calibrated at (0 if no .rot sidecar)."""
    global _COVER_CALIB_ROT_CACHE
    if _COVER_CALIB_ROT_CACHE is None:
        rot = 0
        try:
            with open(COVER_CALIB_ROT_PATH) as fh:
                rot = int(fh.read().strip()) % 4
        except (OSError, ValueError):
            rot = 0                   # legacy mask, no sidecar -> assume no delta
        _COVER_CALIB_ROT_CACHE = rot
    return _COVER_CALIB_ROT_CACHE


def load_cover_mask(shape):
    """Load the calibrated cover mask, resized (nearest) to `shape` = (h, w).

    Returns a uint8 mask or None when calibration is disabled or absent.  Cached.
    """
    global _COVER_MASK_CACHE
    if not (USE_COVER_CALIB and CV2_AVAILABLE):
        return None
    if _COVER_MASK_CACHE is None:
        if not os.path.isfile(COVER_CALIB_MASK_PATH):
            return None
        m = cv2.imread(COVER_CALIB_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        _COVER_MASK_CACHE = m
    m = _COVER_MASK_CACHE
    if m is None:
        return None
    # Rotate the stored mask from its calibration orientation into the current
    # display orientation, so it stays aligned with frames after an eye flip.
    delta = (LIVE_ROTATION - _cover_calib_rotation()) % 4
    rot = _CV2_ROTATIONS[delta]
    if rot is not None:
        m = cv2.rotate(m, rot)
    if m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


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


def _ray_ridges(gray, cx, cy, r_lo, r_hi, gx=None, gy=None, cover_mask=None):
    """Cast _RAD_N_ANGLES rays from (cx, cy) over radii [r_lo, r_hi); return
    (angles, radii).

    radii[i] is the radius of the strongest OUTWARD intensity gradient (dark ->
    bright) along ray i — a circular-boundary ridge — or np.nan if that ray has
    no clear outward edge (e.g. a direction into the equally-dark occluder patch,
    where the radial gradient stays near zero, so it drops out).  Using the
    gradient ridge, not an intensity step, locks onto the true boundary and
    ignores the slow violet-glow ramp.  Pass precomputed gx/gy to avoid recompute.

    If `cover_mask` (calibrated LED-cover mask) is given, a ridge whose point
    falls inside the cover is discarded — that edge is the cover boundary, not
    the pupil, so it cannot drag the circle fit onto the occluder.
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
        for c in cand:
            if cover_mask is not None and cover_mask[ys[c], xs[c]] > 0:
                continue                                 # ridge is on the cover
            radii[i] = float(rs[c])
            break
    return angles, radii


def _ray_edges(gray, cx, cy, rmin, rmax, r_start=0, cover_mask=None):
    """Pupil-boundary ridges: _ray_ridges over the plausible-pupil radius band.

    `r_start` skips the innermost radii (set it past the inpainted-reflex blob).
    """
    r_lo = max(3, int(rmin * 0.5), int(r_start))
    r_hi = int(rmax * _RAD_SEARCH_MULT)
    return _ray_ridges(gray, cx, cy, r_lo, r_hi, cover_mask=cover_mask)


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

    # Calibrated LED-cover mask (known occluder region), if a calibration exists.
    cover_mask = load_cover_mask((h, w))
    if PUPIL_DEBUG_STAGES is not None and cover_mask is not None:
        cvis = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
        cvis[cover_mask > 0, 2] = 255           # cover tinted red
        _dbg("0_cover_calib", cvis)

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
        angles, radii = _ray_edges(work, cx, cy, rmin, rmax, r_start, cover_mask)
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
                         int(r * _IRIS_RP_MIN), int(r * _IRIS_SEARCH_MAX),
                         cover_mask=cover_mask)
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


def coarse_locate(ambient_bgr, flash_bgr, cover_mask=None, prior=None,
                  allow_ml=True):
    """Coarse pupil centre via a confidence-ranked cascade — for alignment guidance.

    Runs the cheap cues always (ambient dark-disc ridge fit + flash red-eye, fused
    by _fuse_pupil) and, when torch + RITnet weights are present, RITnet too; keeps
    the highest-confidence result.  `prior` (last known centre) seeds the red-eye
    window and the RITnet crop and is the fallback when every cue fails.  Inputs
    are BGR in display (rotated) orientation.  Returns
    (center_xy, radius_or_None, confidence, source) or None.

    Only a coarse centre is needed during approach (to steer the device); the exact
    outline doesn't matter until the pupil is centred and the fundus appears.
    """
    if not CV2_AVAILABLE:
        return None
    ridge, ritnet = detect_both(ambient_bgr, flash_bgr, cover_mask, prior, allow_ml)
    candidates = [c for c in (ridge, ritnet) if c is not None]
    if not candidates:
        if prior is not None:
            return ((float(prior[0]), float(prior[1])), None, 0.0, "prior")
        return None
    return max(candidates, key=lambda c: c[2])


# Two centres "agree" when within this fraction of the smaller frame dimension.
DETECT_AGREE_FRAC = 0.10


def detect_both(ambient_bgr, flash_bgr, cover_mask=None, prior=None, allow_ml=True):
    """Run the cheap ridge/red-eye cascade and RITnet SEPARATELY (not fused).

    Returns (ridge, ritnet); each is (center_xy, radius, confidence, source) or
    None.  `ridge` is the ambient dark-disc + flash red-eye cue fused by
    _fuse_pupil (the "traditional CV" result); `ritnet` is the ML segmentation
    (guarded — None where torch / weights are absent).  This is what
    coarse_locate ranks, and what classify_detection() uses to sort a capture by
    which detector(s) found the pupil.
    """
    if not CV2_AVAILABLE:
        return None, None
    h, w = ambient_bgr.shape[:2]
    rmin = _SW_MIN_R_FRAC * min(h, w)
    rmax = _SW_MAX_R_FRAC * min(h, w)

    # ── Ridge: ambient dark-disc fit (stashes _LAST_PUPIL/_LAST_CONF/_LAST_ANCHOR)
    #    fused with the flash red-eye, seeded by the reflex anchor or the prior.
    detect_pupil(ambient_bgr)
    amb_pupil, amb_conf, anchor = _LAST_PUPIL, _LAST_CONF, _LAST_ANCHOR
    seed = anchor if anchor is not None else (
        (int(prior[0]), int(prior[1])) if prior is not None else None)
    redeye = detect_redeye(flash_bgr, seed[0], seed[1], rmin, rmax) \
        if seed is not None else None
    fused = _fuse_pupil(amb_pupil, amb_conf, redeye)
    ridge = None
    if fused is not None:
        cx, cy, r, source, _confident = fused
        conf = amb_conf if source.startswith("ambient") else (
            min(1.0, redeye[3] / 255.0) if redeye is not None else amb_conf)
        ridge = ((float(cx), float(cy)), float(r), float(conf), source)

    # ── RITnet (optional; only where torch + weights are available) ──
    ritnet = None
    # Two-tier gate: run the (slow on Pi) ML rung only when the cheap ridge cue is
    # weak, unless RITNET_ALWAYS forces it on every capture.
    run_ml = allow_ml and _ML_BACKEND is not None
    if run_ml and not RITNET_ALWAYS:
        ridge_conf = ridge[2] if ridge is not None else 0.0
        run_ml = ridge_conf < RITNET_CONF_GATE
    if run_ml:
        ml = _ML_BACKEND                          # torch RITnet (PC) or ncnn (Pi)
        if ML_BACKEND_NAME == "ncnn RITnet":      # slow path — tell the operator
            print("  RITnet (ncnn) running - slow on the Pi, hold steady...")
        green = ambient_bgr[:, :, 1]
        if cover_mask is not None and int(cv2.countNonZero(cover_mask)):
            green = cv2.inpaint(green, cover_mask, 15, cv2.INPAINT_TELEA)
        reflex = _find_corneal_reflex(green)
        reflex_mask = reflex[3] if reflex is not None else None
        ranch = (reflex[0], reflex[1]) if reflex is not None else anchor
        # Crop centre: prior (tracking) -> tangent past the cover (initial)
        # -> the ridge centre -> the reflex anchor.
        if prior is not None:
            cc = prior
        elif cover_mask is not None and int(cv2.countNonZero(cover_mask)):
            cc = ml.tangent_crop_center(cover_mask, (h, w))
        elif ridge is not None:
            cc = ridge[0]
        else:
            cc = ranch
        rr = ml.locate(green, reflex_mask=reflex_mask, anchor=ranch, crop_center=cc)
        if rr.ok and rr.center is not None:
            ritnet = (rr.center, rr.radius, float(rr.confidence), "ritnet")
    return ridge, ritnet


def classify_detection(ridge, ritnet, shape, agree_frac=DETECT_AGREE_FRAC):
    """Categorise a capture by which detector(s) found the pupil.

    Returns (category, chosen) where category is one of
    ``"no_pupil" | "ridge_only" | "ritnet_only" | "both"`` and `chosen` is the
    (center, radius, conf, source) to use downstream (or None).  When both fire
    and AGREE (centres within agree_frac x min(h,w)) it is "both" (higher-conf of
    the two is chosen); when both fire but DISAGREE, the higher-confidence one
    wins and the category collapses to that detector's "*_only".
    """
    if ridge is None and ritnet is None:
        return "no_pupil", None
    if ritnet is None:
        return "ridge_only", ridge
    if ridge is None:
        return "ritnet_only", ritnet
    h, w = shape[:2]
    dist = float(np.hypot(ridge[0][0] - ritnet[0][0], ridge[0][1] - ritnet[0][1]))
    if dist <= agree_frac * min(h, w):
        return "both", (ridge if ridge[2] >= ritnet[2] else ritnet)
    return ("ridge_only", ridge) if ridge[2] >= ritnet[2] else ("ritnet_only", ritnet)


# ───────── RED-EYE / FUNDUS REGION EXTRACTION ─────────
_REDEYE_ROI_MULT = 1.8          # search radius = this x detected pupil radius
_REDEYE_ROI_MIN_FRAC = 0.06     # ...but at least this x min(h, w)
_REDEYE_MIN_SHIFT = 18          # floor on the red-shift score to select a pixel
_REDEYE_AMBIENT_DARK = 90       # ambient luma below this = "was dark" (soft prior)
_REDEYE_AMBIENT_BOOST = 12      # red-shift boost where the pixel read dark in ambient
_REDEYE_SPECULAR_DILATE = 9     # grow the specular / corneal-reflex exclusion


@dataclass
class RedeyeResult:
    valid: bool = False
    mask: object = None            # uint8 (H, W) selection, or None
    overlay: object = None         # BGR highlight image (always set)
    extract: object = None         # BGR isolated region on black, or None
    coverage: float = 0.0          # selected px / ROI px
    notes: str = ""


def redeye_extract(flash_bgr, dark_bgr, ambient_bgr, center, radius,
                   cover_mask=None):
    """Isolate the flash retroreflection (fundus red-eye) pixels near the pupil.

    Selects pixels that shifted most toward red vs the all-off `dark_bgr` (softly
    biased by having read dark in `ambient_bgr`), inside a generous region around
    the detected pupil `center`/`radius` (detection is imperfect, hence 'around' —
    the selection shape is arbitrary, not a circle).  Corneal reflections /
    speculars are EXCLUDED (kept raw, NOT inpainted).  Only runs when the pupil is
    validly located and not centred on the LED cover.  All frames must be in the
    same (display) orientation.  Returns a RedeyeResult; `overlay` always renders
    (highlight, or an INVALID banner) so the caller can save/show it.
    """
    res = RedeyeResult()
    base = cv2.convertScaleAbs(flash_bgr, alpha=3.0) if flash_bgr is not None else None

    def _banner(msg):
        vis = base.copy() if base is not None else np.zeros((16, 16, 3), np.uint8)
        for col, th in (((0, 0, 0), 4), ((0, 0, 255), 1)):
            cv2.putText(vis, f"REDEYE INVALID: {msg}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
        res.overlay = vis
        res.notes = msg
        return res

    if not CV2_AVAILABLE or flash_bgr is None or dark_bgr is None:
        return _banner("frames unavailable")
    h, w = flash_bgr.shape[:2]
    if center is None:
        return _banner("no pupil")
    cx, cy = int(center[0]), int(center[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return _banner("pupil out of frame")
    if (cover_mask is not None and cover_mask.shape[:2] == (h, w)
            and cover_mask[cy, cx] > 0):
        return _banner("pupil on LED cover")

    r = float(radius) if radius else _REDEYE_ROI_MIN_FRAC * min(h, w)
    roi_r = int(max(_REDEYE_ROI_MULT * r, _REDEYE_ROI_MIN_FRAC * min(h, w)))
    roi = np.zeros((h, w), np.uint8)
    cv2.circle(roi, (cx, cy), roi_r, 255, -1)

    f = flash_bgr.astype(np.int16)
    d = dark_bgr.astype(np.int16)
    # Red rose more than blue vs the all-off dark frame == "became reddish".
    redshift = (f[:, :, 2] - d[:, :, 2]) - (f[:, :, 0] - d[:, :, 0])
    warm = f[:, :, 2] > f[:, :, 0]                  # genuinely warm in the flash
    if ambient_bgr is not None and ambient_bgr.shape[:2] == (h, w):
        aluma = ambient_bgr.astype(np.int16).mean(axis=2)
        redshift = redshift + np.where(aluma < _REDEYE_AMBIENT_DARK,
                                       _REDEYE_AMBIENT_BOOST, 0)

    # Adaptive floor from the ROI's red-shift distribution (Otsu), with a fixed min.
    rs_pos = np.clip(redshift, 0, 255).astype(np.uint8)
    roi_vals = rs_pos[roi > 0]
    otsu = _REDEYE_MIN_SHIFT
    if roi_vals.size:
        otsu, _ = cv2.threshold(roi_vals.reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = max(_REDEYE_MIN_SHIFT, float(otsu))
    mask = ((redshift >= floor) & warm & (roi > 0)).astype(np.uint8) * 255

    # Exclude corneal reflections / speculars (do NOT inpaint them here).
    fgray = cv2.cvtColor(flash_bgr, cv2.COLOR_BGR2GRAY)
    _, spec = cv2.threshold(fgray, _INPAINT_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    reflex = _find_corneal_reflex(fgray)
    if reflex is not None:
        spec = cv2.bitwise_or(spec, reflex[3])
    spec = cv2.dilate(spec, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_REDEYE_SPECULAR_DILATE, _REDEYE_SPECULAR_DILATE)))
    mask[spec > 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    roi_area = int(cv2.countNonZero(roi))
    sel_area = int(cv2.countNonZero(mask))
    res.coverage = sel_area / roi_area if roi_area else 0.0

    extract = np.zeros_like(flash_bgr)
    extract[mask > 0] = flash_bgr[mask > 0]         # isolated region, original colour

    overlay = base.copy()
    if sel_area:
        overlay[mask > 0] = (0.35 * overlay[mask > 0]
                             + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 1)
    overlay[(spec > 0) & (roi > 0)] = (0, 0, 255)   # excluded speculars, in the ROI
    cv2.circle(overlay, (cx, cy), roi_r, (255, 255, 0), 1)
    cv2.circle(overlay, (cx, cy), 3, (255, 0, 255), -1)
    for col, th in (((0, 0, 0), 3), ((0, 255, 0), 1)):
        cv2.putText(overlay, f"redeye px={sel_area} cov={res.coverage:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)

    res.valid = True
    res.mask = mask
    res.overlay = overlay
    res.extract = extract
    res.notes = f"{sel_area}px in ROI r={roi_r}"
    return res


def _post_to_receiver(folder, filename, data):
    """POST raw bytes to receiver.py as /upload/<folder>/<filename>.

    Returns True on HTTP 200.  Never raises — a transfer problem must not
    interrupt capture, streaming or calibration.
    """
    url = f"http://{REMOTE_HOST}:{REMOTE_PORT}/upload/{folder}/{filename}"
    try:
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=TRANSFER_TIMEOUT) as resp:
            if resp.status != 200:
                print(f"[TRANSFER ERROR] {folder}/{filename}: HTTP {resp.status}")
                return False
        return True
    except (urllib.error.URLError, OSError) as e:
        print(f"[TRANSFER ERROR] {folder}/{filename}: {e}")
        return False


def upload_calibration_image(bgr):
    """POST the (rotated) calibration frame to the dev machine via receiver.py.

    receiver.py routes the special `calibration` folder into its own
    calibration/ directory and BUILDS the cover mask there (build only on the dev
    machine).  Returns True on a successful upload.
    """
    if not (TRANSFER_ENABLED and CV2_AVAILABLE) or bgr is None:
        return False
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        return False
    fname = os.path.basename(COVER_CALIB_IMAGE_PATH)        # led_cover_calib.jpg
    if _post_to_receiver("calibration", fname, buf.tobytes()):
        print(f"Calibration image uploaded to {REMOTE_HOST}.")
        return True
    return False


def transfer_capture(ambient_array, flash_array, both_array=None, detect_bgr=None,
                     dark_array=None, redeye=None, pupil_center=None):
    """Stage the raw capture frames locally and POST them to the dev machine.

    Writes LOCAL_STAGING_DIR/capture_<timestamp>/{ambient.jpg, flash.jpg, both.jpg,
    dark.jpg, detect.jpg, redeye_overlay.jpg, redeye_extract.jpg, redeye_mask.png,
    meta.json}, then uploads each via HTTP POST to receiver.py.  The raw frames
    (ambient/flash/both/dark) are saved un-annotated so the harness can re-run
    detection + red-eye extraction cleanly; `detect.jpg` and the `redeye_*` files
    are the Pi-side overlays/outputs.  meta.json is uploaded LAST so the receiver
    only triggers once the whole folder has arrived.

    Any failure is reported on the Pi but never raised — a transfer problem
    must not interrupt capture or streaming.
    """
    if not TRANSFER_ENABLED:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(LOCAL_STAGING_DIR, f"capture_{stamp}")
    fnames = ["ambient.jpg", "flash.jpg"]

    # ── stage the frames locally ──
    try:
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(ambient_array).save(os.path.join(folder, "ambient.jpg"))
        Image.fromarray(flash_array).save(os.path.join(folder, "flash.jpg"))
        if both_array is not None:
            Image.fromarray(both_array).save(os.path.join(folder, "both.jpg"))
            fnames.append("both.jpg")
        if dark_array is not None:
            Image.fromarray(dark_array).save(os.path.join(folder, "dark.jpg"))
            fnames.append("dark.jpg")
        if detect_bgr is not None and CV2_AVAILABLE:
            cv2.imwrite(os.path.join(folder, "detect.jpg"), detect_bgr)
            fnames.append("detect.jpg")
        if redeye is not None and redeye.valid and CV2_AVAILABLE:
            cv2.imwrite(os.path.join(folder, "redeye_overlay.jpg"), redeye.overlay)
            cv2.imwrite(os.path.join(folder, "redeye_extract.jpg"), redeye.extract)
            cv2.imwrite(os.path.join(folder, "redeye_mask.png"), redeye.mask)
            fnames += ["redeye_overlay.jpg", "redeye_extract.jpg", "redeye_mask.png"]
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
            "has_both":      both_array is not None,
            "has_dark":      dark_array is not None,
            "pupil_center":  ([round(float(pupil_center[0]), 1),
                               round(float(pupil_center[1]), 1)]
                              if pupil_center is not None else None),
        }
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        fnames.append("meta.json")             # reordered FIRST for the upload below
    except OSError as e:
        print(f"[TRANSFER ERROR] could not stage capture: {e}")
        return

    # ── upload to receiver.py: meta.json FIRST (it carries the centred pupil
    #    coords), then the rest; abort on the FIRST failure (don't bother with the
    #    others) so a dropped file doesn't waste time on the remainder. ──
    order = ["meta.json"] + [f for f in fnames if f != "meta.json"]
    folder_name = f"capture_{stamp}"
    ok = True
    for fname in order:
        try:
            with open(os.path.join(folder, fname), "rb") as f:
                data = f.read()
        except OSError as e:
            print(f"[TRANSFER ERROR] {fname}: {e}")
            ok = False
            break
        if not _post_to_receiver(folder_name, fname, data):
            print(f"[TRANSFER] {fname} failed - aborting the rest.")
            ok = False
            break

    print(f"Transferred {folder_name} to {REMOTE_HOST}." if ok
          else f"Transfer of {folder_name} aborted (a file failed).")


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


def capture_image(picam2, live_center_frac=None, live_radius_frac=None):
    """Capture a dark / flash / flash+ambient / ambient set and extract the red-eye.

    `live_center_frac` = (fx, fy) and `live_radius_frac` = fr are the pupil region
    (fractions of frame width/height) carried over from the last centred live frame.
    They locate the red-eye extraction — there is NO pupil detection at capture time.


    Three frames in one short LED sequence (the eye barely moves between them):
      1.  flash only      — retina retroreflection, pupil still dilated
      2.  flash + ambient — eye structure lit AND the pupil retroreflecting
      3.  ambient only     — the pupil reads as a dark disc

    All four raw frames are transferred to the dev machine.  The raw ambient is
    saved to AMBIENT_PATH; PHOTO_PATH gets the red-eye highlight overlay when a live
    pupil region is known, else the plain processed flash.
    """
    print("Capturing flash / flash+ambient / ambient triple...")

    apply_camera_settings(picam2, FLASH_GAIN)
    time.sleep(0.02)
    _drain_frames(picam2)              # let the new gain take effect

    # Four frames in one short LED sequence (the eye barely moves between them);
    # dark+flash are captured first, while the pupil is still dilated, so the
    # flash-minus-dark red-eye extraction is pupil-size-matched:
    #   0) dark (all off)   — near-black reference for the red-eye extraction
    #   1) flash only        — retina retroreflection, pupil still dilated
    #   2) flash + ambient   — eye structure lit AND the pupil retroreflecting
    #   3) ambient only      — the pupil reads as a dark disc (constricted)

    # ── 0) DARK (all LEDs off) — near-black reference, pupil still dilated ──
    flash_off()
    ambient_off()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed under the previous state
    dark_array = picam2.capture_array()

    # ── 1) FLASH ONLY (GPIO 17) ──
    flash_on()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed before LED on
    flash_array = picam2.capture_array()

    # ── 2) FLASH + AMBIENT (both LEDs) ──
    ambient_on()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed before ambient on
    both_array = picam2.capture_array()

    # ── 3) AMBIENT ONLY (GPIO 22/27/23/6/26/16) ──
    flash_off()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed before flash off
    ambient_array = picam2.capture_array()
    ambient_off()

    # Restore live settings
    apply_camera_settings(picam2, LIVE_GAIN)

    # Save the raw ambient image for inspection
    Image.fromarray(ambient_array).save(AMBIENT_PATH)

    # ───────── RED-EYE EXTRACTION (no pupil detection) ─────────
    # Alignment is done live; here we just isolate the red-eye pixels using the pupil
    # region carried over from the last centred live frame (live_center/radius_frac).
    # Produces the red-eye highlight overlay (detect.jpg) + outputs, transferred with
    # the raw frames.  Wrapped so an extraction error never aborts capture/transfer.
    detect_bgr = None
    redeye = None
    pupil_center = None
    img = None
    if CV2_AVAILABLE:
        try:
            detect_bgr, redeye, pupil_center = _redeye_capture(
                ambient_array, flash_array, dark_array,
                live_center_frac, live_radius_frac)
            img = Image.fromarray(cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:                     # noqa: BLE001
            import traceback
            print(f"[REDEYE ERROR] {e!r}")
            traceback.print_exc()
    if img is None:
        img = process_image(flash_array, [])

    # ───────── IMAGE TRANSFER ─────────
    # Push the raw 4 frames (+ detect.jpg + red-eye outputs); meta.json (with the
    # centred pupil coords) is uploaded FIRST and the rest abort on any failure.
    transfer_capture(ambient_array, flash_array, both_array, detect_bgr,
                     dark_array, redeye, pupil_center)

    # Save the annotated flash photo locally
    img.save(PHOTO_PATH)

    # Red-pixel preview on the Pi, AFTER the transfer attempt (if enabled).
    if REDEYE_PREVIEW and CV2_AVAILABLE and redeye is not None \
            and redeye.overlay is not None:
        ro = redeye.overlay
        if ro.shape[0] > 720:
            s = 720.0 / ro.shape[0]
            ro = cv2.resize(ro, (int(ro.shape[1] * s), 720))
        cv2.imshow("Red-eye extraction", ro)
        cv2.waitKey(1)

    print("Captured.")


def _draw_detections(vis, ridge, ritnet, category):
    """Overlay BOTH detectors on the guidance image: ridge (traditional) in cyan,
    RITnet (NN) in magenta, plus a legend with the category."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = vis.shape[0]
    if ridge is not None:
        (rx, ry), rr = ridge[0], ridge[1]
        cv2.circle(vis, (int(rx), int(ry)), max(3, int(rr)), (255, 255, 0), 2)  # cyan
        cv2.putText(vis, f"ridge {ridge[2]:.2f}",
                    (int(rx) - 36, int(ry) - max(3, int(rr)) - 6),
                    font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if ritnet is not None:
        (nx, ny), nr = ritnet[0], (ritnet[1] or 8)
        cv2.circle(vis, (int(nx), int(ny)), max(3, int(nr)), (255, 0, 255), 2)  # magenta
        cv2.putText(vis, f"nn {ritnet[2]:.2f}",
                    (int(nx) - 24, int(ny) + max(3, int(nr)) + 16),
                    font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    legend = f"[{category}]  cyan=ridge  magenta=nn"
    cv2.putText(vis, legend, (10, h - 12), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, legend, (10, h - 12), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _redeye_capture(ambient_array, flash_array, dark_array=None,
                    live_center_frac=None, live_radius_frac=None):
    """Extract the red-eye (fundus) pixels on a capture — no pupil detection here.

    The pupil region is carried over from the last centred LIVE frame:
    `live_center_frac`=(fx, fy) and `live_radius_frac`=fr are fractions of the
    captured frame's width/height, giving the red-eye centre and ROI size.  All
    alignment/guidance work happens live; this only isolates the red pixels.

    Returns (overlay_bgr, redeye, pupil_center) — the red-eye highlight overlay (or
    a brightened flash if no region is known), the RedeyeResult (or None), and the
    full-frame pupil coords used.
    """
    rot = _CV2_ROTATIONS[LIVE_ROTATION]
    amb_bgr = cv2.cvtColor(ambient_array, cv2.COLOR_RGB2BGR)
    flash_bgr = cv2.cvtColor(flash_array, cv2.COLOR_RGB2BGR)
    dark_bgr = (cv2.cvtColor(dark_array, cv2.COLOR_RGB2BGR)
                if dark_array is not None else None)
    if rot is not None:
        amb_bgr = cv2.rotate(amb_bgr, rot)
        flash_bgr = cv2.rotate(flash_bgr, rot)
        if dark_bgr is not None:
            dark_bgr = cv2.rotate(dark_bgr, rot)

    h, w = flash_bgr.shape[:2]
    center = radius = None
    if live_center_frac is not None:
        center = (float(live_center_frac[0]) * w, float(live_center_frac[1]) * h)
        if live_radius_frac:
            radius = float(live_radius_frac) * w

    cover_mask = load_cover_mask((h, w))
    redeye = None
    if dark_bgr is not None and center is not None:
        redeye = redeye_extract(flash_bgr, dark_bgr, amb_bgr, center, radius, cover_mask)
        print(f"  REDEYE: {'valid' if redeye.valid else 'skipped'} - {redeye.notes}")
    elif center is None:
        print("  REDEYE: skipped - no live pupil region (align in live mode first)")

    vis = (redeye.overlay if (redeye is not None and redeye.overlay is not None)
           else cv2.convertScaleAbs(flash_bgr, alpha=3.0))
    return vis, redeye, center


def _ship_calibration(bgr):
    """Upload the calibration image; optionally build the mask locally on the Pi.

    By default (`PI_BUILD_CALIB` off) the Pi only ships the image and the dev
    machine (receiver.py) builds the mask.  With it on, the Pi also builds, saves
    and starts using the mask locally.  Returns False only if `bgr` is unusable.
    """
    if bgr is None or not CV2_AVAILABLE:
        print("cv2 not available - cannot calibrate.")
        return False
    upload_calibration_image(bgr)
    if PI_BUILD_CALIB:
        mask = build_cover_mask(bgr)
        if mask is not None and save_cover_calibration(mask, bgr):
            frac = 100.0 * float((mask > 0).sum()) / mask.size
            print(f"Cover mask built on Pi ({frac:.1f}% of frame).")
        else:
            print("Pi cover-mask build FAILED (no dark edge-touching region).")
    return True


def calibrate_cover(picam2):
    """Capture a calibration frame under the CURRENT lighting (NO eye in place)
    and ship it to the dev machine, which builds the cover mask.  LEDs are left
    untouched (whatever is locked/on stays on)."""
    print("Cover calibration: ensure NO eye is in place...")
    _drain_frames(picam2)
    frame = picam2.capture_array()
    if not CV2_AVAILABLE:
        print("cv2 not available - cannot calibrate.")
        return
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rot = _CV2_ROTATIONS[LIVE_ROTATION]
    if rot is not None:
        bgr = cv2.rotate(bgr, rot)
    _ship_calibration(bgr)


def _set_cover_side(side):
    """Set which image edge the LED cover sits on: "top" (left eye) / "bottom"
    (right eye).  The two eyes differ by a 180° rig flip, so switching side adds
    180° to LIVE_ROTATION; the cover mask follows (load_cover_mask) and the
    detection geometry auto-detects the side, so everything stays consistent.
    """
    global COVER_SIDE, LIVE_ROTATION
    side = "bottom" if str(side).lower().startswith("b") else "top"
    if side != COVER_SIDE:
        LIVE_ROTATION = (LIVE_ROTATION + 2) % 4
        COVER_SIDE = side
    return COVER_SIDE


def _prompt_cover_orientation():
    """Ask which edge the LED cover sits on before streaming (default top = left
    eye).  Choosing the opposite side rotates the feed 180°.  A missing terminal
    or blank input keeps the default.  Switch later live with `f` / `cover_side`.
    """
    try:
        ans = input("LED cover at [t]op (left eye, default) or "
                    "[b]ottom (right eye)? ").strip().lower()
    except EOFError:
        ans = ""
    _set_cover_side("bottom" if ans.startswith("b") else "top")
    print(f"Cover side: {COVER_SIDE}  (rotation {LIVE_ROTATION * 90}°).\n")


def streaming_mode(picam2):
    """Live video feed.

    SPACE (hold) / r — flash LED (GPIO 17), hold / lock
    a               — toggle ambient LED (GPIO 22/27/23/6/26/16)
    p               — toggle live Orlosky pupil detection
    g               — toggle the live guidance arrow (arms auto-capture when centred)
    k               — authorise the next auto-capture (after one fires)
    t               — toggle PC image transfer (HTTP-POST to receiver.py)
    f               — flip to the other eye (cover top<->bottom, 180°)
    m               — toggle RITnet: every capture <-> only when ridge weak
    c               — snap an LED-cover calibration (no eye), instant; ships the
                      current frame (LEDs left as-is) for the dev machine to build
    ←/→             — rotate the live feed
    ENTER or e      — capture an ambient + flash still
    s               — exit streaming
    """

    global LIVE_ROTATION
    global TRANSFER_ENABLED

    if not CV2_AVAILABLE:
        print("cv2 not available - cannot show live feed.")
        return

    print("\nStreaming mode ON")
    print("SPACE=flash | r=lock | a=ambient | p=pupil-detect | g=guide-arrow | "
          "k=authorise-autocap | t=pc-transfer | f=flip-eye | "
          "c=calibrate-cover | ←/→=rotate | ENTER/e=capture | s=exit\n")

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
    import guidance
    swirski_live_on = SWIRSKI_LIVE_DEFAULT
    live_guide_on = False                  # 'g': draw the guidance arrow live
    _live_guide = guidance.GuidanceTracker()
    _live_center = None
    _live_radius = None                    # live pupil radius (disp px) -> red-eye ROI
    _autocap_armed = False                 # auto-capture when centred (armed by 'g'/'k')
    _centred_count = 0
    _last_autocap = 0.0
    _detect_counter = 0
    _overlay_cache = None
    exit_reason = "unknown"

    # WND_PROP_VISIBLE is unsupported on some backends (this Pi's GTK build returns
    # -1 even for a visible window), which made the close-on-X check exit streaming
    # instantly.  Probe it now; only use the check when the backend reports a valid
    # value (>= 0).  Otherwise rely on 's' / Ctrl-C to exit.
    _win_close_check = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 0
    if not _win_close_check:
        print("(window close-detection off: WND_PROP_VISIBLE unsupported on this backend)")

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

            # Live pupil detection (ridge) — recompute every N frames; runs when
            # the pupil overlay (p) OR live guidance (g) is on.
            if swirski_live_on or live_guide_on:
                _detect_counter += 1
                fresh = (_overlay_cache is None
                         or _detect_counter % _LIVE_DETECT_SKIP == 0)
                if fresh:
                    gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
                    _overlay_cache = detect_pupil(gray, live=True)
                    if _LAST_PUPIL is not None:
                        _live_center = (_LAST_PUPIL[0], _LAST_PUPIL[1])
                        _live_radius = _LAST_PUPIL[2]
                    else:
                        _live_center = None
                        _live_radius = None
                if swirski_live_on and _overlay_cache:
                    disp = _draw_overlays(disp.copy(), _overlay_cache)
                # Live guidance arrow: target the LED-cover edge, arrow from it to
                # the pupil, plain-language instruction.  Everything in disp coords
                # (cover mask resized to the display), so it lines up with the feed.
                if live_guide_on:
                    cover_disp = load_cover_mask(disp.shape[:2])
                    tgt = guidance.target_point(disp.shape, cover_disp,
                                                GUIDANCE_TARGET_MODE)
                    lg = _live_guide.update(_live_center, disp.shape, tgt, 1.0, "live")
                    disp = guidance.annotate(disp, lg, tgt, _live_center)

                    # Auto-capture when centred (only on fresh detections; debounced,
                    # armed, cooled-down).  Disarms after firing -> 'k' re-authorises.
                    if fresh:
                        _centred_count = _centred_count + 1 if lg.state == "centred" else 0
                        if (_autocap_armed and _centred_count >= AUTOCAP_CENTRED_FRAMES
                                and time.time() - _last_autocap > AUTOCAP_COOLDOWN):
                            _autocap_armed = False
                            _last_autocap = time.time()
                            _centred_count = 0
                            # Fractional pupil region from THIS centred live frame
                            # (disp space) -> used as the red-eye location + ROI size.
                            frac = ((_live_center[0] / disp.shape[1],
                                     _live_center[1] / disp.shape[0])
                                    if _live_center is not None else None)
                            rad_frac = (_live_radius / disp.shape[1]
                                        if _live_radius else None)
                            print(f"Centred in live feed -> auto-capture: ambient off, "
                                  f"dilating {AUTOCAP_DILATE:.1f}s, then flash. "
                                  f"(press 'k' to authorise the next)")
                            if ambient_led_on:
                                ambient_off()
                            flash_off(); led_on = False
                            cv2.imshow(WINDOW_NAME, disp); cv2.waitKey(1)
                            time.sleep(AUTOCAP_DILATE)     # pupil dilation
                            capture_image(picam2, frac, rad_frac)
                            if ambient_led_on:
                                ambient_on()
                            apply_camera_settings(picam2, FLASH_GAIN)
                            _live_guide.reset()

            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKeyEx(1)
            now = time.time()
            space_held = (now - last_space_time <= SPACE_TIMEOUT)

            # Also exit if the window is closed via the X button (only where the
            # backend actually supports the visibility property).
            if _win_close_check and \
                    cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                exit_reason = "window closed (WND_PROP_VISIBLE < 1)"
                break

            if key == ord('s'):
                exit_reason = "'s' pressed"
                break

            # Capture still: ENTER or e.  Uses the current live pupil region for the
            # red-eye extraction (no capture-time detection); needs the live overlay
            # ('p') or guidance ('g') on so a pupil is being tracked.
            if key in (13, 10) or key == ord('e'):
                flash_off()
                led_on = False
                last_space_time = 0.0
                cfrac = ((_live_center[0] / disp.shape[1],
                          _live_center[1] / disp.shape[0])
                         if _live_center is not None else None)
                rfrac = (_live_radius / disp.shape[1]
                         if (cfrac is not None and _live_radius) else None)
                capture_image(picam2, cfrac, rfrac)
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

            # a toggles the ambient LED (GPIO 22/27/23/6/26/16)
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

            # g toggles the live guidance arrow (target + arrow + instruction).
            # Turning it on ARMS the first auto-capture (fires when centred).
            elif key == ord('g'):
                live_guide_on = not live_guide_on
                _live_guide.reset()
                _overlay_cache = None
                _centred_count = 0
                _autocap_armed = live_guide_on
                print(f"Live guidance arrow: {'ON (auto-capture armed)' if live_guide_on else 'OFF'}")

            # k re-authorises the next auto-capture (after one has fired)
            elif key == ord('k'):
                if live_guide_on:
                    _autocap_armed = True
                    _centred_count = 0
                    print("Auto-capture authorised - align to centre for the next take.")
                else:
                    print("Turn on live guidance ('g') first.")

            # t toggles PC image transfer (HTTP-POST captures to receiver.py)
            elif key == ord('t'):
                TRANSFER_ENABLED = not TRANSFER_ENABLED
                print(f"PC image transfer: "
                      f"{'ON' if TRANSFER_ENABLED else 'OFF'}")

            # f flips to the other eye: cover top<->bottom (180°, mask follows)
            elif key == ord('f'):
                _set_cover_side("bottom" if COVER_SIDE == "top" else "top")
                _overlay_cache = None          # orientation changed -> redetect
                print(f"Cover side: {COVER_SIDE} (rotation {LIVE_ROTATION * 90}°)")

            # c snaps the current frame as an LED-cover calibration (no eye in
            # place) and ships it — instant, no confirm, LEDs left as locked/on.
            elif key == ord('c'):
                _ship_calibration(frame.copy())
                _overlay_cache = None          # cover may change -> redetect
                print("Calibration captured.")

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

    except Exception as e:                     # noqa: BLE001 — surface WHY streaming ended
        import traceback
        exit_reason = f"exception: {e!r}"
        traceback.print_exc()

    finally:

        flash_off()
        ambient_off()
        apply_camera_settings(picam2, LIVE_GAIN)
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:                      # noqa: BLE001
            pass
        print(f"Streaming mode OFF - reason: {exit_reason}\n")


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
    global PI_DETECT
    global PI_BUILD_CALIB
    global TRANSFER_ENABLED
    global PREVIEW_VIEWER
    global USE_COVER_CALIB
    global GUIDANCE_TARGET_MODE, GUIDANCE_USE_ML
    global RITNET_ALWAYS, RITNET_CONF_GATE
    global REDEYE_PREVIEW
    global _GUIDANCE_TRACKER, _GUIDANCE_PRIOR

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

        elif param == "pi_detect":
            PI_DETECT = bool(int(value))

        elif param == "pi_build_calib":
            PI_BUILD_CALIB = bool(int(value))

        elif param == "redeye_preview":
            REDEYE_PREVIEW = bool(int(value))

        # PC image transfer (HTTP-POST capture triples to receiver.py)
        elif param == "transfer":
            TRANSFER_ENABLED = bool(int(value))

        # External retina_preview.jpg viewer (xdg-open); off by default
        elif param == "preview":
            PREVIEW_VIEWER = bool(int(value))
            if PREVIEW_VIEWER:
                os.system(f"xdg-open '{PHOTO_PATH}' >/dev/null 2>&1 &")
                print("Preview viewer opened.")
            else:
                print("Preview viewer off (any open window stays).")

        # Cover side / eye: "top" (left eye) | "bottom" (right eye); flips 180°
        elif param == "cover_side":
            if value.lower()[:1] in ("t", "b"):
                _set_cover_side(value)
                print(f"Cover side: {COVER_SIDE} (rotation {LIVE_ROTATION * 90}°).")
            else:
                print("cover_side must be 'top' or 'bottom'.")
                return

        # Cover masking + alignment guidance
        elif param == "use_cover":
            USE_COVER_CALIB = bool(int(value))

        elif param == "guide_ml":
            GUIDANCE_USE_ML = bool(int(value))

        # RITnet gating: run it every capture, or only when ridge is weak
        elif param == "ritnet_always":
            RITNET_ALWAYS = bool(int(value))

        elif param == "ritnet_gate":          # ridge conf below which RITnet fires
            RITNET_CONF_GATE = max(0.0, min(1.0, float(value)))

        elif param == "guide_reset":          # start a new alignment session
            _GUIDANCE_TRACKER = None
            _GUIDANCE_PRIOR = None
            print("Guidance session reset.")

        elif param == "target_mode":          # "centre" | "cover_top_mid"
            if value in ("centre", "cover_top_mid"):
                GUIDANCE_TARGET_MODE = value
            else:
                print("target_mode must be 'centre' or 'cover_top_mid'.")
                return

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

    # Check ML availability at boot
    if RITNET_AVAILABLE:
        print(f"✓ RITnet ML pupil detection available ({ML_BACKEND_NAME})")
    else:
        print("✗ RITnet ML pupil detection NOT available (ridge + red-eye only)")

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

    # ───────── PREOPEN IMAGE VIEWER (off by default) ─────────
    # The alignment-guidance cv2 window is the live view now; the external
    # retina_preview.jpg viewer is only opened when PREVIEW_VIEWER is on.
    if PREVIEW_VIEWER:
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

    # Which eye / cover side are we starting on? (top = left eye, default.)
    _prompt_cover_orientation()

    # ───────── BOOT STRAIGHT INTO STREAMING MODE ─────────
    # The program always starts streaming; exiting streaming (s) drops into the
    # normal command loop below, from which streaming can be re-entered.

    try:

        streaming_mode(picam2)
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        except Exception:
            print("Warning: could not flush stdin; stray keystrokes may appear in the command loop.")
            pass

        print("ENTER     = capture (ambient + flash, with pupil detection)")
        print("s         = streaming mode (SPACE=flash, r=lock, a=ambient, "
              "p=pupil-detect, e/ENTER=capture)")
        print("calibrate = snap an LED-cover calibration (no eye); ships it to build here")
        print("q         = quit\n")

        # ───────── MAIN LOOP ─────────

        while True:

            cmd = input("> ").strip()

            # Quit

            if cmd == "q":
                break

            # Capture

            elif cmd == "":
                capture_image(picam2)

            # LED-cover calibration (no eye in place) — captures under the current
            # lighting and ships the image; the dev machine builds the mask.

            elif cmd == "calibrate":
                calibrate_cover(picam2)

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
