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
f                   Flip the cover side (top<->bottom, rotates the feed 180°)
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
import math
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

# guidance.py has no cap.py import, so this is safe at module level — but it does
# import cv2 unconditionally, so it goes behind the same guard: cap.py is expected
# to stay importable (with CV2_AVAILABLE False) on a machine without OpenCV.
try:
    import guidance
except ImportError:
    guidance = None

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
# from — "top" or "bottom".  The two rig orientations differ by a 180° rotation,
# so switching side just adds 180° to LIVE_ROTATION; the cover mask is rotated to
# match in load_cover_mask, and the detection geometry reads the cover side back
# off the mask, so it adapts either way.  Set at startup (prompt) and toggled live
# with `f` / `cover_side`.
#
# The pairing below is what LIVE_ROTATION's default (1 = 90° CCW) actually shows:
# at rotation 1 the cover comes in at the BOTTOM.  It used to be declared as "top",
# which inverted every side choice — asking for a top cover produced a bottom one.
# Do not "tidy" this back to "top" without re-checking the feed.
COVER_SIDE = "bottom"

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
GUIDANCE_TARGET_MODE = "centre"        # drive the pupil to the CAMERA CENTRE
                                       # ("cover_top_mid" = the LED-cover inner edge,
                                       #  which needs a cover mask; toggle: target_mode)
GUIDANCE_USE_ML = True                 # let coarse_locate try RITnet (if present)
GUIDANCE_WINDOW = "Alignment Guidance" # Pi cv2 window the annotated capture is shown in

# Auto-capture: the staged session fires the take itself at the gaze-sweep peak and
# then DISARMS — the operator re-authorises the next search-and-take with `k`.
# Before that final shot everything goes dark for AUTOCAP_DILATE s so the pupil can
# dilate (the sweep has just held the flash on the eye, constricting it).
AUTOCAP_DILATE = 1.5                    # seconds all-dark to dilate before the flash
AUTOCAP_COOLDOWN = 3.0                  # min seconds between auto-captures (backstop)
AUTOCAP_CENTRED_FRAMES = 2             # consecutive centred detections before firing

# Centring stage of the live staged session: a symmetric box the pupil must sit
# inside before the approach begins.  Wider than tall — the operator still has to
# close the vertical gap to the cover by hand in the next stage, so y need not be
# pinned here, while x wants to be reasonably settled.
CENTRE_DEAD_X_FRAC = 0.06               # half-width
CENTRE_DEAD_Y_FRAC = 0.09               # half-height
CENTRE_HOLD_FRAMES = 3                  # consecutive centred detections to hand over

# Show the red-pixel (red-eye) extraction on the Pi AFTER a capture's transfer.
REDEYE_PREVIEW = True

# Tint the detected red-eye pixels onto the LIVE feed whenever the flash is lit and
# an all-off reference frame is available.  On by default; SHIFT toggles it live.
REDEYE_HIGHLIGHT_DEFAULT = True

# Blank the camera picture on the LIVE feed, leaving only the guidance drawing on
# black ('i' toggles).  Detection, red-eye extraction, the sweep and the session
# saving all carry on exactly as before — this hides the picture, it does not stop
# any processing, and it applies ONLY to the live window: the capture previews, the
# session browser and the mosaic still show real imagery.
LIVE_IMAGE_HIDDEN_DEFAULT = False

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


# ── LIVE (calibration-free) LED-cover detection ──
# CURRENTLY OFF (USE_LIVE_COVER below).  The staged session runs
# centre -> sweep -> capture, driving the pupil to the CAMERA CENTRE
# (GUIDANCE_TARGET_MODE = "centre") and taking the sweep direction from the
# COVER_SIDE prompt rather than from a detected cover.  Everything below still
# works and is kept for when cover-relative targeting is wanted again — flip
# USE_LIVE_COVER to True and set GUIDANCE_TARGET_MODE = "cover_top_mid".
USE_LIVE_COVER = False
#
# The detector itself locates the cover on the working frame every few frames
# instead of relying on a stored calibration.
#
# Connected components (as _find_led_cover_mask does) does NOT work here: on an
# ambient-lit frame with an eye in place the cover touches the eye-socket and brow
# shadow, so the near-black blob either swallows 40% of the frame or trips an area
# cap and yields nothing.  Measured over the Transfers/ captures, that approach
# found a cover on barely a third of frames.
#
# What IS reliable is the cover's defining property: it is an unbroken dark band
# entering from ONE edge of the frame.  So instead of labelling blobs, we walk
# inward from the top edge and from the bottom edge, column by column, and record
# how deep the dark run goes before the image turns durably bright.  That depth
# profile IS the cover's outline; the side with the deeper, wider band wins.  A
# column whose run never ends is the socket shadow, not the cover, and is capped.
_COVER_LIVE_DARK = 8             # near-black ceiling.  MUCH lower than
                                 # _LED_DARK_THRESH (30): the cover blocks the LED
                                 # outright and reads ~0, while the eye-socket and
                                 # brow shadow sit around 10-30.  Measured over the
                                 # Transfers/ captures, dropping 30 -> 8 took the
                                 # mask from 38% of the frame to 23%, raised the
                                 # detection rate from 29/46 frames to 43/46, and
                                 # cut the cases where the band swallows the pupil
                                 # from 13/29 to 6/43.  Do not raise this.
_COVER_LIVE_GAP = 6              # consecutive bright rows that end a dark run (so a
                                 # speck of glare on the cover doesn't cut it short)
_COVER_LIVE_MIN_DEPTH_FRAC = 0.02  # a column counts as covered past this depth
_COVER_LIVE_MAX_DEPTH_FRAC = 0.60  # a band deeper than this has fused with the
                                   # socket shadow — reject rather than mask the eye
_COVER_LIVE_MIN_COLS_FRAC = 0.20   # need this fraction of columns covered to call it
_COVER_LIVE_PROFILE_SMOOTH = 15    # median window (px) over the depth profile
_COVER_LIVE_SPIKE_WIN = 91         # 1-D opening window (px) that shaves narrow deep
                                   # spikes off the profile.  These happen where the
                                   # dark PUPIL joins the band through an eyelash, and
                                   # they matter: guidance.target_point takes the
                                   # mask's deepest row, so one spike would put the
                                   # alignment target on the spike tip, not the cover
_COVER_LIVE_STABLE_FRAMES = 3    # consecutive stable detections before "found"
_COVER_LIVE_JITTER_FRAC = 0.03   # max centroid movement (frac of min(h,w)) to count
                                 # as the same, settled cover
_COVER_LIVE_HOLD_FRAMES = 10     # keep the last good mask this many misses before
                                 # falling back to the stored calibration


def _cover_depth_profile(dark_from_edge):
    """Per-column depth of the dark run entering from row 0.

    `dark_from_edge` is a boolean (h, w) array already oriented so that row 0 is
    the frame edge being probed.  A run ends at the first block of
    `_COVER_LIVE_GAP` consecutive bright rows, so glare specks on the cover do not
    truncate it.  Returns an int array of length w (depth in px, h if never ends).
    """
    h, w = dark_from_edge.shape
    gap = min(_COVER_LIVE_GAP, h)
    bright = (~dark_from_edge).astype(np.int32)
    csum = np.cumsum(bright, axis=0)
    # Sum over each window of `gap` rows starting at y: rows [y, y+gap).
    win = csum[gap - 1:, :] - np.vstack([np.zeros((1, w), np.int32),
                                         csum[:-gap, :]])
    all_bright = (win == gap)                      # (h-gap+1, w)
    ends = all_bright.any(axis=0)
    return np.where(ends, all_bright.argmax(axis=0), h).astype(np.int32)


def _open_profile(prof, win):
    """1-D greyscale opening (rolling min, then rolling max) of a depth profile.

    Removes narrow *deep* excursions — a spike where the dark pupil joins the
    cover band through an eyelash — while leaving the band's broad shape intact.
    """
    w = prof.shape[0]
    k = max(3, int(win) | 1)
    if w < k:
        return prof

    def _roll(a, fn):
        pad = np.pad(a, k // 2, mode="edge")
        return fn(np.lib.stride_tricks.sliding_window_view(pad, k), axis=-1)

    return _roll(_roll(prof, np.min), np.max).astype(np.int32)


def detect_cover_live(img):
    """Locate the LED cover on a live frame, without any calibration.

    `img` is BGR or grayscale in DISPLAY orientation.  Returns
    (mask, side, centroid) — a uint8 mask (255 = cover), "top"/"bottom" for the
    edge it intrudes from, and its (cx, cy) — or None when nothing qualifies.
    """
    if not CV2_AVAILABLE or img is None:
        return None
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    dark = gray < _COVER_LIVE_DARK

    min_depth = max(2, int(_COVER_LIVE_MIN_DEPTH_FRAC * h))
    max_depth = int(_COVER_LIVE_MAX_DEPTH_FRAC * h)

    best = None
    for side in ("top", "bottom"):
        prof = _cover_depth_profile(dark if side == "top" else dark[::-1, :])
        # Smooth the profile so a single stray column cannot spike the outline.
        # (Rolling median in numpy — cv2.medianBlur only takes 8-bit above k=5, and
        # these depths run to the frame height.)
        k = _COVER_LIVE_PROFILE_SMOOTH | 1                 # median needs odd
        if w >= k:
            pad = np.pad(prof, k // 2, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, k)
            prof = np.median(win, axis=-1).astype(np.int32)
        prof = _open_profile(prof, _COVER_LIVE_SPIKE_WIN)  # shave pupil spikes
        covered = prof >= min_depth
        if covered.mean() < _COVER_LIVE_MIN_COLS_FRAC:
            continue
        depths = prof[covered]
        if float(np.median(depths)) > max_depth:
            continue                               # fused with the socket shadow
        score = float(prof.clip(0, max_depth).sum())
        if best is None or score > best[0]:
            best = (score, side, prof)
    if best is None:
        return None

    _, side, prof = best
    prof = np.clip(prof, 0, max_depth)
    # Paint the band: rows [0, depth) from whichever edge it entered.
    rows = np.arange(h, dtype=np.int32).reshape(-1, 1)
    band = rows < prof.reshape(1, -1)
    if side == "bottom":
        band = band[::-1, :]
    mask = (band.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (_COVER_CALIB_DILATE, _COVER_CALIB_DILATE))
    mask = cv2.dilate(mask, k)                     # grow the soft fuzzy edge
    if int(cv2.countNonZero(mask)) == 0:
        return None

    ys, xs = np.where(mask > 0)
    return mask, side, (float(xs.mean()), float(ys.mean()))


class CoverTracker:
    """Debounce detect_cover_live across frames; fall back to the calibration.

    A single-frame cover mask flickers (the threshold sits right on the cover's
    soft edge), and the guidance target is derived from it, so the target would
    jump.  Require the detection to repeat with a settled centroid before
    reporting `found`, then hold the last good mask through short dropouts.  Only
    once the hold expires does it fall back to the stored calibration mask, so the
    calibration-free path is what normally drives the session.
    """

    def __init__(self):
        self.mask = None            # last good mask (display coords)
        self.side = None            # "top" / "bottom"
        self.centroid = None
        self.stable = 0             # consecutive settled detections
        self.misses = 0             # consecutive frames with no detection
        self.vetoed = 0             # detections thrown out for covering the pupil
        self.from_calibration = False

    def reset(self):
        self.__init__()

    @property
    def found(self):
        """Cover located and settled (or supplied by the calibration fallback)."""
        return self.mask is not None and (self.from_calibration
                                          or self.stable >= _COVER_LIVE_STABLE_FRAMES)

    def update(self, img, pupil_center=None):
        """Feed one display-orientation frame.  Returns the current mask or None.

        `pupil_center`, when known, is used as a veto: the pupil is dark and can be
        contiguous with the cover through the lashes, and a band that has run
        through it is wrong by construction (redeye_extract refuses a pupil sitting
        on the cover).  Such a detection is treated as a miss, so the session waits
        for a cleaner frame instead of aligning to a bogus edge.
        """
        shape = img.shape[:2]
        det = detect_cover_live(img)
        if det is not None and pupil_center is not None:
            px = int(np.clip(pupil_center[0], 0, shape[1] - 1))
            py = int(np.clip(pupil_center[1], 0, shape[0] - 1))
            if det[0][py, px] > 0:
                det = None
                self.vetoed += 1
        if det is not None:
            mask, side, centroid = det
            jitter = _COVER_LIVE_JITTER_FRAC * min(shape)
            settled = (self.centroid is not None
                       and not self.from_calibration
                       and float(np.hypot(centroid[0] - self.centroid[0],
                                          centroid[1] - self.centroid[1])) <= jitter)
            self.stable = self.stable + 1 if settled else 1
            self.mask, self.side, self.centroid = mask, side, centroid
            self.misses = 0
            self.from_calibration = False
            return self.mask

        self.misses += 1
        if self.mask is not None and self.misses <= _COVER_LIVE_HOLD_FRAMES:
            return self.mask                          # ride out a short dropout

        # Hold expired — fall back to the stored calibration, if there is one.
        fallback = load_cover_mask(shape)
        if fallback is not None and int(cv2.countNonZero(fallback)):
            ys, xs = np.where(fallback > 0)
            self.mask = fallback
            self.centroid = (float(xs.mean()), float(ys.mean()))
            self.side = "top" if self.centroid[1] < shape[0] * 0.5 else "bottom"
            self.from_calibration = True
            self.stable = 0
            return self.mask

        self.mask = self.side = self.centroid = None
        self.stable = 0
        self.from_calibration = False
        return None

    def status(self):
        """Short status string for the on-screen sub-line."""
        if self.mask is None:
            return "cover: searching" + (f" (vetoed {self.vetoed})"
                                         if self.vetoed else "")
        src = "calib" if self.from_calibration else "live"
        return f"cover: {self.side}/{src}" + ("" if self.found
                                              else f" ({self.stable}/"
                                                   f"{_COVER_LIVE_STABLE_FRAMES})")


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


def draw_redeye_highlight(bgr, mask):
    """Tint the detected red-eye pixels on a live frame and outline them.

    Cheap enough to run every frame (a boolean tint + one contour pass) — the
    expensive selection is done by redeye_extract and cached by the caller, so a
    slightly stale mask is drawn between recomputes rather than recomputing here.
    """
    if mask is None or bgr is None or mask.shape[:2] != bgr.shape[:2]:
        return bgr
    n = int(cv2.countNonZero(mask))
    if n == 0:
        return bgr
    bgr[mask > 0] = (0.35 * bgr[mask > 0]
                     + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cnts, -1, (0, 255, 0), 1)
    for col, th in (((0, 0, 0), 3), ((0, 255, 0), 1)):
        cv2.putText(bgr, f"redeye {n}px", (10, bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, th, cv2.LINE_AA)
    return bgr


def draw_hidden_banner(bgr, redeye_mask=None, extra=""):
    """Label a blanked live frame and keep the numbers the picture would have shown.

    With the camera image hidden ('i') the red-eye tint is gone, so its pixel count
    — the one number the sweep is actually steering on — is printed instead.
    """
    if bgr is None:
        return bgr
    h, w = bgr.shape[:2]
    lab = "IMAGE HIDDEN (i)"
    tw = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    for col, th in (((0, 0, 0), 3), ((160, 160, 160), 1)):
        cv2.putText(bgr, lab, (max(6, w - tw - 10), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
    n = int(cv2.countNonZero(redeye_mask)) if redeye_mask is not None else 0
    line = f"redeye {n}px" if n else ""
    if extra:
        line = f"{line}  {extra}" if line else extra
    if line:
        for col, th in (((0, 0, 0), 3), ((0, 255, 0), 1)):
            cv2.putText(bgr, line, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, th, cv2.LINE_AA)
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
_REDEYE_MIN_SHIFT = 24          # floor on the red-shift score to select a pixel.
                                # Raised from 18.  Measured over the Transfers/
                                # captures: on-eye selection drops to 68% of its
                                # former size (harmless — the sweep's peak-finding
                                # reads relative change, and no capture falls under
                                # REDEYE_MIN_PX) while the stray red picked up
                                # with the ROI parked on lit skin goes to ZERO.
                                # Past ~27 real readings start reading as lost.
_REDEYE_AMBIENT_DARK = 90       # ambient luma below this = "was dark" (soft prior)
_REDEYE_AMBIENT_BOOST = 12      # red-shift boost where the pixel read dark in ambient
_REDEYE_AMBIENT_MAX = 110       # STRICT gate (strict_ambient=True): a fundus pixel is
                                # inside the pupil, so it MUST read dark in ambient.
                                # Anything brighter is lit skin/sclera/iris, however
                                # red it went under the flash.  Speculars (the corneal
                                # reflection, the one bright thing legitimately inside
                                # the pupil) are already excluded from the mask, so
                                # this can be a hard veto rather than a prior.
                                # This absolute cap is only the backstop — measured on
                                # the Transfers/ captures it never fires, because the
                                # ambient frames are dim enough that even lit skin sits
                                # under it.  The Otsu split below is what discriminates.
_REDEYE_AMBIENT_OTSU_MIN = 30   # Adaptive veto: split the ambient luma INSIDE the ROI
                                # (Otsu) and reject the bright side — the pupil is the
                                # dark population, everything else is lit tissue.  Only
                                # applied when the split lands above this, i.e. the ROI
                                # really does contain both populations; a ROI sitting
                                # wholly inside a dark pupil splits on noise alone and
                                # would otherwise cull half the genuine fundus.
_REDEYE_SPECULAR_DILATE = 9     # grow the specular / corneal-reflex exclusion

# ── Pupil-ellipse gating: which connected red group is the real fundus ──
# Red-shift alone will happily select warm skin, a lid margin or a sclera glint.
# The fundus reflex, though, can only come back through the pupil — so the kept
# group must OVERLAP the detected pupil circle, and exactly one group survives.
_REDEYE_GROUP_OUTER_MULT = 1.6  # keep a qualifying group's pixels out to this x the
                                # pupil radius: the glow spills a little past the
                                # detected circle (the fit is imperfect and the pupil
                                # dilates between frames), but not indefinitely
_REDEYE_GROUP_MIN_OVERLAP = 8   # px of a group that must fall INSIDE the pupil circle
                                # for it to qualify at all
_REDEYE_GROUP_SIMILAR = 0.6     # two groups whose areas are within this ratio count as
                                # "of similar size", so shape breaks the tie
_REDEYE_GROUP_DISK_MARGIN = 0.12  # diskness gap that settles a similar-size tie; below
                                  # it, distance from the pupil centre decides

# ── C-shape / ring rejection ──
# A real reflex FILLS THE PUPIL IN from one side as the alignment improves, so it
# is always a solid region: a disc, a half-disc, a D.  A crescent that curls round
# into a C, or a ring tracing the pupil rim, is the limbus or the cover edge
# catching the flash, never the fundus.  Two independent tests, because neither
# catches both failures (measured on reference shapes):
#
#     shape                              solidity   contour-fill
#     full disc                            0.99        1.00
#     half disc (fills from one side)      0.99        1.00
#     3/4 disc                             0.82        1.00
#     thick C (270 deg arc)                0.57        0.99   <- solidity catches
#     thin C  (240 deg arc)                0.35        0.99   <- solidity catches
#     full ring                            0.99        0.30   <- fill catches
#
# solidity     = area / convex-hull area.  A C's hull bridges its opening.
# contour-fill = area / area enclosed by the outer contour.  A ring encloses its
#                own hole, which the hull test cannot see.
_REDEYE_MIN_SOLIDITY = 0.60     # below this the group has curled into a C.  On the
                                # Transfers/ captures this rejects 1 of 14 (the
                                # worst-aligned, solidity 0.58) and clears both
                                # reference C shapes; 0.55 rejects none of the 14
_REDEYE_MIN_CONTOUR_FILL = 0.55  # below this the group is a ring/annulus


def _group_diskness(xs, ys, area):
    """How much a pixel group looks like a filled-in disc rather than an arc.

    area / (area of its minimum enclosing circle).  A partially coloured-in circle
    — what a real fundus reflex looks like — scores high; a highlight running along
    part of a circumference (an iris/limbus rim or a lid margin catching the flash)
    encloses a big circle with few pixels in it, so it scores low.
    """
    if area <= 0 or len(xs) < 3:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), rad = cv2.minEnclosingCircle(pts)
    if rad <= 0:
        return 0.0
    return float(area) / (math.pi * float(rad) ** 2)


def _group_shape(comp_mask, area):
    """(solidity, contour_fill) for one connected group.

    Both are 1.0 for a solid blob.  Solidity falls when the group curls into a C
    (its convex hull bridges the opening); contour-fill falls when it is a ring
    (the outer contour encloses a hole the hull test cannot see).
    """
    cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or area <= 0:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    enclosed = float(cv2.contourArea(c))
    hull = float(cv2.contourArea(cv2.convexHull(c)))
    solidity = (enclosed / hull) if hull > 0 else 0.0
    fill = (float(area) / enclosed) if enclosed > 0 else 0.0
    return solidity, min(1.0, fill)


def select_redeye_group(mask, center, radius):
    """Reduce a red-shift mask to the ONE connected group that is the fundus reflex.

    A group qualifies only if it overlaps the pupil circle (`center`, `radius`);
    a qualifying group keeps its pixels out to `_REDEYE_GROUP_OUTER_MULT` x the
    radius, so a reflex that spills slightly past the fitted circle survives whole
    while a blob merely touching the pupil cannot drag half the frame in with it.

    Among qualifying groups the choice is, in order:
      1.  area — the biggest group wins outright unless another is of similar size
      2.  diskness — of similarly-sized groups, the filled-in disc beats the arc
      3.  distance — if those are close too, the group centred nearest the pupil

    Returns (mask, note).  The mask is all-zero when nothing qualifies.
    """
    if mask is None or center is None or not radius:
        return mask, ""
    h, w = mask.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)

    # Trim anything well beyond the pupil before grouping, so a qualifying group
    # cannot reach out into unrelated warm tissue through a thin bridge.
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    outer = d2 <= (_REDEYE_GROUP_OUTER_MULT * r) ** 2
    work = mask.copy()
    work[~outer] = 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(work)
    if n <= 1:
        return np.zeros_like(mask), "no red group"

    inside = d2 <= r * r                      # the pupil circle itself
    cands = []
    rejected = []
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        sel = labels == lbl
        overlap = int(np.count_nonzero(sel & inside))
        if overlap < _REDEYE_GROUP_MIN_OVERLAP:
            continue                          # never reaches into the pupil
        comp = sel.astype(np.uint8) * 255
        solidity, fill = _group_shape(comp, area)
        # A genuine reflex fills the pupil in from one side, so it is solid.  A C
        # or a ring is the limbus/cover rim catching the flash, not the fundus.
        if solidity < _REDEYE_MIN_SOLIDITY:
            rejected.append(f"C-shape(sol={solidity:.2f})")
            continue
        if fill < _REDEYE_MIN_CONTOUR_FILL:
            rejected.append(f"ring(fill={fill:.2f})")
            continue
        ys, xs = np.nonzero(sel)
        cands.append({
            "lbl": lbl, "area": area, "overlap": overlap,
            "disk": _group_diskness(xs, ys, area),
            "sol": solidity, "fill": fill,
            "dist": float(np.hypot(cents[lbl][0] - cx, cents[lbl][1] - cy)),
        })

    if not cands:
        why = (" - rejected " + ", ".join(rejected[:3])) if rejected else ""
        return np.zeros_like(mask), f"no valid red group{why}"

    biggest = max(c["area"] for c in cands)
    top = [c for c in cands if c["area"] >= _REDEYE_GROUP_SIMILAR * biggest]
    if len(top) == 1:
        best = top[0]
        why = "largest"
    else:
        top.sort(key=lambda c: -c["disk"])
        if top[0]["disk"] - top[1]["disk"] > _REDEYE_GROUP_DISK_MARGIN:
            best = top[0]
            why = f"diskness {top[0]['disk']:.2f}"
        else:
            best = min(top, key=lambda c: c["dist"])
            why = f"nearest ({best['dist']:.0f}px)"

    out = np.where(labels == best["lbl"], 255, 0).astype(np.uint8)
    note = (f"group {best['area']}px disk={best['disk']:.2f} "
            f"sol={best['sol']:.2f} off={best['dist']:.0f}px [{why}]"
            + (f" of {len(cands)}" if len(cands) > 1 else "")
            + (f", {len(rejected)} shape-rejected" if rejected else ""))
    return out, note


@dataclass
class RedeyeResult:
    valid: bool = False
    mask: object = None            # uint8 (H, W) selection, or None
    overlay: object = None         # BGR highlight image (always set)
    extract: object = None         # BGR isolated region on black, or None
    coverage: float = 0.0          # selected px / ROI px
    notes: str = ""


def redeye_extract(flash_bgr, dark_bgr, ambient_bgr, center, radius,
                   cover_mask=None, strict_ambient=False, gate_to_pupil=True):
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
    ambient_ok = None
    if ambient_bgr is not None and ambient_bgr.shape[:2] == (h, w):
        aluma = ambient_bgr.astype(np.int16).mean(axis=2)
        redshift = redshift + np.where(aluma < _REDEYE_AMBIENT_DARK,
                                       _REDEYE_AMBIENT_BOOST, 0)
        if strict_ambient:
            # Hard veto: the fundus glow can only come through the pupil, which is
            # dark under ambient light.  Pairing each flash frame with a FRESH
            # ambient one is what makes this trustworthy — a stale ambient frame
            # would be misregistered against a moving eye.
            #
            # The cut is adaptive: Otsu over the ambient luma inside the ROI, so it
            # tracks however bright this particular ambient exposure came out, with
            # the absolute cap as a backstop.
            a_thr = float(_REDEYE_AMBIENT_MAX)
            a_roi = np.clip(aluma[roi > 0], 0, 255).astype(np.uint8)
            if a_roi.size:
                a_otsu, _ = cv2.threshold(a_roi.reshape(-1, 1), 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if float(a_otsu) >= _REDEYE_AMBIENT_OTSU_MIN:
                    a_thr = min(a_thr, float(a_otsu))
            ambient_ok = aluma < a_thr

    # Adaptive floor from the ROI's red-shift distribution (Otsu), with a fixed min.
    rs_pos = np.clip(redshift, 0, 255).astype(np.uint8)
    roi_vals = rs_pos[roi > 0]
    otsu = _REDEYE_MIN_SHIFT
    if roi_vals.size:
        otsu, _ = cv2.threshold(roi_vals.reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = max(_REDEYE_MIN_SHIFT, float(otsu))
    sel = (redshift >= floor) & warm & (roi > 0)
    if ambient_ok is not None:
        sel &= ambient_ok
    mask = sel.astype(np.uint8) * 255

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

    # Keep only the one connected group that belongs to the pupil.  Done AFTER the
    # morphology so the open/close has already bridged the reflex into one blob.
    group_note = ""
    if gate_to_pupil and radius:
        mask, group_note = select_redeye_group(mask, (cx, cy), r)

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
    cv2.circle(overlay, (cx, cy), roi_r, (255, 255, 0), 1)          # search ROI
    if radius:                                                       # the gate itself
        cv2.circle(overlay, (cx, cy), int(round(r)), (0, 255, 255), 1)   # pupil
        cv2.circle(overlay, (cx, cy), int(round(_REDEYE_GROUP_OUTER_MULT * r)),
                   (0, 140, 200), 1)                                 # spill allowance
    cv2.circle(overlay, (cx, cy), 3, (255, 0, 255), -1)
    for col, th in (((0, 0, 0), 3), ((0, 255, 0), 1)):
        cv2.putText(overlay, f"redeye px={sel_area} cov={res.coverage:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)

    res.valid = True
    res.mask = mask
    res.overlay = overlay
    res.extract = extract
    res.notes = (f"{sel_area}px in ROI r={roi_r}"
                 + (f"; {group_note}" if group_note else ""))
    return res


# ───────── APPROACH + FIXATION-TARGET SEARCH ─────────
# Two stages share one idea: the red reflex is the only trustworthy signal, so
# nothing is predicted — things are moved and the red is measured.
#
#   APPROACH  the OPERATOR closes the vertical gap between the LED cover and the
#             pupil by hand.  We just watch the red pixel count and hand over once
#             there is a decent chunk of it.
#   TARGET    the FIXATION TARGET is then walked around the screen by coordinate
#             hill-climbing (y, then x, then finer).  The patient keeps looking at
#             it, so their gaze follows wherever it goes.  No direction is guessed:
#             a probe that lowers the red is simply reversed.
REDEYE_MIN_PX = 40              # fewer red pixels than this = no usable reflex
REDEYE_PUPIL_CONF_GATE = 0.20   # min detect_pupil confidence to trust the re-detected
                                # disc over the ambient-stage radius

# ── how much red counts as "a decent chunk" ──
# This is a share of the PUPIL, not an absolute count: the same reflex fills a
# small pupil and barely marks a wide one, so a fixed bar is either unreachable on
# a dilated eye or trivially cleared on a constricted one.
#
# Which pupil, though, is the whole difficulty.  The live fit usually reads LARGER
# than the truth — corneal reflections drag the circle off the pupil — and it is
# wrong by a different amount from frame to frame, so tracking it would make the
# bar wander with detection noise rather than with the eye.  What the detector does
# do is occasionally lock on properly: rare frames where the rays nearly all find
# an edge and those edges sit tightly on one circle.  Confidence is exactly that
# product (angular coverage x inlier fraction), so those moments are identifiable,
# and only they are believed.  Measured over the 99 Transfers/ ambient frames:
# confidence runs a median of 0.30 and a 90th percentile of 0.575, so 0.60 picks
# the top ~6% — the "suddenly fits properly" frames and nothing else.
#
# The bar then follows the most RECENT such fit, not the best one ever seen: the
# pupil dilates and constricts through a run, and a good fit from a minute ago
# describes an eye that has since changed.  Poorly-fitted frames in between change
# nothing at all — the last confident measurement simply stands.
APPROACH_GOOD_PX = 650          # the bar before any confident fit has landed.
                                # Measured on the Transfers/ captures the gated
                                # reflex runs 349-2672px (median ~1220), so the
                                # original 3000 would almost never have been reached
REDEYE_PUPIL_FRAC = 1.0 / 3.0   # required red, as a share of the fitted pupil's area
REDEYE_SIZE_CONF = 0.60         # a "really confident" fit — see the percentiles above
# Clamps, because an unreachable bar is the stage's worst failure: it cannot be
# argued with, and the operator is left pushing the scope at a target that will
# never go green.  Bounded by what the optics actually deliver (that 349-2672px
# range): below the floor, noise clears the bar; above the ceiling, too few real
# reflexes do.  The ceiling binds once a confident fit reports r > ~34px, which on
# the measured captures is the top of the confident-fit range and about the median
# reflex — past that the fit is claiming most of the iris and the bar it implies
# would be cleared by well under half of genuine reflexes.
REDEYE_MIN_PX_FLOOR = 350
REDEYE_MIN_PX_CEIL = 1200
APPROACH_HOLD = 2               # consecutive measurements at/above it, so a single
                                # lucky frame does not advance the stage
APPROACH_TIMEOUT_S = 90.0       # give the operator this long before dropping back
# Once there IS reflex and the operator has been told to hold, STAY held.  The
# reflex flickers frame to frame, and dropping straight back to "move down and
# right" on one weak reading sends the scope hunting again — usually away from
# the spot that just worked.  Only a sustained collapse releases the latch.
APPROACH_RELEASE_FRAC = 0.35    # reflex below this share of APPROACH_GOOD_PX...
APPROACH_RELEASE_TRIES = 4      # ...for this many readings in a row, to release

# ── overshoot / lost-eye detection during the approach ──
# The operator walks the pupil toward the cover edge by hand with nothing to go on
# but the red count, and it is easy to sail straight past: the pupil ends up under
# the cover or off the frame, no reflex can ever appear, and the stage would sit
# there for the whole of APPROACH_TIMEOUT_S still saying "keep closing".  A lost
# pupil also reads as a blink, so the blink detector goes on excusing the cycles
# and nothing downstream ever complains.  Both failures show up the same way on the
# ambient frame — the pupil is not found, or is found jammed against the cover's
# edge — so both feed ONE counter, checked once per lighting cycle (~1.1s).
# Deliberately slow to react: a single missed detection is entirely normal, and an
# alignment that is merely difficult must not be torn down underneath the operator.
APPROACH_EDGE_FRAC = 0.08       # pupil within this fraction of the frame height of
                                # the cover's own edge = driven past the useful zone
APPROACH_WARN_TRIES = 3         # bad ambient frames before the on-screen "back off"
APPROACH_ABORT_TRIES = 7        # ...and before dropping back to centring (~8s)

# ── stepped gaze scan (the target stage) ──
# The fixation point walks its range in STEPS and pauses at each one, and the
# reflex is recorded only while it is stopped.  The best position is then chosen
# from that profile.  Lighting is what shaped this stage: probing needed an ambient
# blip per measurement to re-check the pupil, and every blip re-constricts it, so
# there is NO ambient or dark switching here at all — the flash is simply held.
#
# The steps exist because a gaze cannot do anything else.  A saccade to a new point
# takes ~200-300ms of latency plus flight time, and only then is the eye pointing
# where the dot is; measurements arrive every _LIVE_DETECT_SKIP-th frame (~0.4s).
# Swept continuously, the eye chases a point it never reaches and each reading is
# taken at an unknown gaze angle — the profile ends up recording lag, not fundus.
# So: glide one step (the point is never teleported, which would lose the gaze),
# stop, let the eye arrive, and only then measure.  Every recorded sample is
# therefore a real, settled gaze direction, and every saved mosaic frame is one
# picture of one gaze position rather than a smear between two.
SCAN_SPEED_FRAC_S = 0.20        # travel per second BETWEEN stops, as a fraction of
                                # min(h, w) — fast enough not to drag out the scan,
                                # slow enough for the eye to track rather than jump
# Each stop is a CAPTURE, not a sample off a continuously-lit eye: the LEDs stay
# off while the point travels and the gaze settles, then one flash frame is taken
# and the LEDs go off again.  Two reasons.  The pupil re-dilates in the dark, and
# it re-dilates during time the scan has to spend anyway — waiting for the patient
# to find and fixate the new point — so the dilation is free.  And a single flash
# per gaze direction is the same exposure the final capture takes, which is what
# makes the scan's frames and the final frame comparable at all.
SCAN_DILATE_S = 0.6             # darkness required before a stop's flash fires.
                                # Runs from the moment the LEDs went off, so the
                                # travel and gaze-settle time counts toward it and
                                # usually satisfies it outright
SCAN_STEP_FRAC = 0.07           # distance covered between one stop and the next.
                                # Sets the whole scan's length: the vertical range
                                # is 0.35 of min(h, w) over two legs (0.61 of
                                # travel), so this gives ~9 stops at ~1.2s each —
                                # about 13s of held flash for the y axis, against
                                # ~5s for the old continuous sweep.  Widen it to
                                # trade profile resolution for time on the eye
SCAN_GAZE_LATENCY_S = 0.35      # time for the eye to actually reach a new point: a
                                # saccade's latency plus its flight, before which
                                # the gaze is not yet on this stop at all
SCAN_HOLD_MAX_S = 3.5           # watchdog on one stop.  The capture normally ends
                                # the stop; this covers a stop whose capture keeps
                                # coming back empty, which would otherwise park the
                                # scan on one point indefinitely
# The sweep is deliberately LOPSIDED.  The LED sits behind the cover, so reflex
# grows as the gaze turns that way and there is a lot of useful travel in that
# direction; turning away from the cover only ever loses reflex, so a short leg
# is enough to establish that the peak has been passed.  Side-to-side matters far
# less than either — set its range to 0 to skip that axis altogether.
SCAN_RANGE_Y_TOWARD_FRAC = 0.26   # vertical travel TOWARD the cover/LED
SCAN_RANGE_Y_AWAY_FRAC = 0.09     # ...and away from it
SCAN_RANGE_X_FRAC = 0.0           # side-to-side (0 = skip the axis entirely)
SCAN_SETTLE_S = 0.9             # pause before a sweep, and at the chosen best
SCAN_CONFIRM_FRAMES = 2         # captures taken at the chosen peak before the final
                                # shot.  Each is now a full dark-dilate-flash cycle
                                # rather than a frame off a lit eye, so this is the
                                # number of extra seconds it costs
SCAN_MIN_SAMPLES = 4            # too few readings on an axis to trust the profile.
                                # One capture per stop, so this is literally "at
                                # least four usable stops on this axis"
SCAN_SMOOTH = 3                 # median window over the recorded profile — kept
                                # narrow: the profile is now one entry per settled
                                # reading, not one per frame of a continuous sweep
SCAN_DROPOUT_TRIES = 12         # consecutive empty readings before giving up.
                                # Higher than the paced stages: readings now come
                                # every frame, so a blink alone is several of them

# Banking mosaic material DURING the search.  Every measurement whose reflex is
# confidently inside the pupil is worth keeping — the gaze is being walked around,
# so consecutive measurements see genuinely different parts of the fundus, which
# is precisely what the mosaic needs.  "Confidently in the pupil" means: the
# measurement was valid (so the group already overlapped the pupil and passed the
# C-shape/ring tests), it is a decent size, and the pupil disc was actually
# re-detected rather than falling back to the stale ambient radius.
SESSION_SAVE_MIN_PX = 500       # smaller than this is not worth stitching
SESSION_SAVE_GAP_S = 0.8        # min seconds between saved frames
SESSION_SAVE_REQUIRE_DISC = True  # require source == "redetect"


# ───────── BLINK DETECTION ─────────
# A blink ruins every measurement it touches: the lid covers the pupil, so the
# red reflex vanishes, pupil detection lands on a lid crease, and the guidance
# either chases a phantom or counts the frame as "reflex lost" and gives up.  The
# frames either side are just as bad — the lid is mid-sweep.
#
# Two cues, because neither alone is trustworthy here:
#
#  1. LOST PUPIL.  A lid across the pupil breaks the ambient dark-disc fit, and
#     that needs no photometric calibration at all — it is the same detector the
#     rest of the pipeline already relies on.  This is the primary cue.
#
#  2. NOTHING BLACK LEFT.  The pupil is dark in EVERY channel; lit skin is bright
#     in red.  So the darkest tenth of the per-pixel channel MAXIMUM, in a tight
#     ROI, jumps when a lid covers the pupil.  Measured over the Transfers/
#     frames with a synthetic lid: ~15 open, ~49 covered.  Grayscale is useless
#     for this — under violet/red light a saturated-red cheek maps to ~60 gray
#     and reads as "dark" alongside the pupil.
#
# Cue 2 is corroboration, not a gate: validated only against a SYNTHETIC lid (no
# real blink footage exists yet), it caught 28/46 full closures and no partial
# ones, largely because pasted "skin" is often shadow in these frames.  The live
# number is printed in the status line so it can be calibrated from one real
# session — raise BLINK_RISE_FACTOR if real blinks are being missed.
BLINK_DARK_PCTILE = 10          # percentile of the max-channel that tracks the
                                # blackest part of the pupil
BLINK_ROI_MULT = 1.2            # measure within this x the pupil radius
BLINK_ROI_MIN_FRAC = 0.05       # ...but at least this x min(h, w)
BLINK_RISE_FACTOR = 1.6         # darkest-tenth rising this much above baseline
BLINK_RISE_MIN = 6.0            # ...and by at least this many levels
BLINK_HOLD_FRAMES = 2           # keep distrusting cycles this long afterwards —
                                # the lid is still moving as it reopens
BLINK_WARMUP = 4                # samples before the baseline means anything
BLINK_BASELINE_N = 12           # running-median window over open frames
BLINK_MAX_CONSEC = 6            # after this many cycles blamed on blinking, stop
                                # excusing them: an eye that is shut, absent or
                                # simply lost is NOT a blink, and must be allowed
                                # to reach the normal lost-reflex path instead of
                                # stalling the search for ever


class BlinkDetector:
    """Flags cycles a blink may have spoiled, so the red-eye stages can skip them.

    `update()` takes the AMBIENT frame of a cycle plus the pupil-detection result
    already computed from it, and returns True while the cycle is suspect — during
    the blink and for BLINK_HOLD_FRAMES after.  The baseline only learns from
    frames judged open, so a long closure cannot drag it down and normalise itself.
    """

    def __init__(self):
        self.hist = []              # darkest-tenth values of recent OPEN frames
        self.hold = 0               # cycles still distrusted after a blink
        self.consec = 0             # consecutive cycles blamed on blinking
        self.baseline = None
        self.last = 0.0             # most recent darkest-tenth value
        self.blinks = 0             # completed blinks this session
        self.why = ""               # which cue fired, for the status line

    def reset(self):
        self.__init__()

    def _darkest(self, bgr, center, radius):
        """Darkest tenth of the per-pixel channel maximum, in a tight ROI."""
        h, w = bgr.shape[:2]
        r = int(max(BLINK_ROI_MULT * float(radius or 0),
                    BLINK_ROI_MIN_FRAC * min(h, w)))
        cx, cy = ((w // 2, h // 2) if center is None
                  else (int(center[0]), int(center[1])))
        roi = bgr[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
        if roi.size == 0:
            return 255.0
        chan_max = roi.max(axis=2) if roi.ndim == 3 else roi
        return float(np.percentile(chan_max, BLINK_DARK_PCTILE))

    def update(self, bgr, center=None, radius=None, pupil_found=True,
               pupil_conf=1.0):
        """Feed one cycle's ambient frame + its pupil result.  True => distrust."""
        if bgr is None:
            return self.hold > 0
        self.last = self._darkest(bgr, center, radius)
        lost = (not pupil_found) or pupil_conf < SWEEP_TRACK_CONF

        warming = self.baseline is None or len(self.hist) < BLINK_WARMUP
        risen = (not warming
                 and self.last > max(self.baseline * BLINK_RISE_FACTOR,
                                     self.baseline + BLINK_RISE_MIN))

        if (lost or risen) and self.consec < BLINK_MAX_CONSEC:
            if self.hold == 0:
                self.blinks += 1
            self.why = "pupil lost" if lost else "nothing black left"
            self.hold = BLINK_HOLD_FRAMES
            self.consec += 1
            return True

        if self.hold > 0 and self.consec < BLINK_MAX_CONSEC:
            self.hold -= 1          # reopening — still not trustworthy
            self.consec += 1
            return True

        # Open (or we have run out of patience and must stop blaming blinks).
        self.hold = 0
        self.consec = 0
        self.why = ""
        if not lost:
            self.hist.append(self.last)     # learn only from open frames
            if len(self.hist) > BLINK_BASELINE_N:
                self.hist.pop(0)
            self.baseline = float(np.median(self.hist))
        return False

    @property
    def suspect(self):
        return self.hold > 0

    def status(self):
        if self.baseline is None or len(self.hist) < BLINK_WARMUP:
            return f"blink: learning dark={self.last:.0f}"
        rel = self.last / self.baseline if self.baseline else 0.0
        tag = f"SUSPECT({self.why})" if self.hold else "ok"
        return (f"blink: {tag} dark={self.last:.0f}/{self.baseline:.0f} "
                f"x{rel:.2f} n={self.blinks}")
TARGET_MARGIN_FRAC = 0.12       # keep the target this far inside the frame edges —
                                # a fixation point in the corner is unusable
TARGET_INVERT_X = True          # The patient sees the fixation point MIRRORED
                                # left-to-right, so to send their gaze left the
                                # point has to be drawn to the right.  Applied only
                                # where the offset becomes a screen position — the
                                # search itself is a blind hill-climb, so the sign
                                # convention of its internal offset never matters,
                                # but where the dot is actually painted does.

# Stages that run the flash/ambient lighting cycle (rather than plain ambient).
# The streaming loop keys several decisions off this: it skips its own pupil
# detection, keeps the red-eye highlight alive across the ambient blip, and lets
# the stage rather than the operator drive both LEDs.
_LIT_STAGES = ((guidance.STAGE_APPROACH, guidance.STAGE_TARGET)
               if guidance is not None else ())

# Each measurement is a PAIR of frames: a brief ambient blip, then back to flash.
# The ambient companion is what proves a red patch is really fundus (it has to read
# dark there), and refreshing it every measurement keeps it registered against an
# eye that is deliberately moving.  The flash stays on the rest of the time — it is
# the patient's fixation target, and every ambient blip re-constricts the pupil.
SWEEP_LED_SETTLE_S = 0.15       # after flipping the LEDs, let the exposure catch up
SWEEP_DARK_S = 0.35             # ALL-OFF pause between the ambient blip and the
                                # flash frame.  The ambient light constricts the
                                # pupil, so measuring straight afterwards reads a
                                # smaller reflex than the alignment deserves; this
                                # lets it re-open.  The frame at the end of the
                                # pause also REFRESHES the dark reference, keeping
                                # the subtraction registered with a moving eye.
SWEEP_FLASH_S = 0.05            # flash-on time before the pipeline is flushed.
                                # Short on purpose: the pupil starts constricting
                                # the instant the flash reaches it, so the FIRST
                                # properly-lit frame carries the biggest reflex and
                                # every frame after it is a worse measurement of the
                                # same alignment.  What makes that first frame
                                # trustworthy is the flush below, not a long dwell —
                                # this only has to cover the LED itself coming up.

# Every frame the red measurement depends on is grabbed AFTER draining the camera
# pipeline, never straight off the loop.  picam2.capture_array() can hand back a
# frame exposed under the previous lighting state, and all three inputs are ruined
# by exactly that: a stale flash frame used as the dark reference cancels the very
# reflex being measured, a stale flash frame read as the ambient companion vetoes a
# real fundus as "too bright", and a stale dark frame measured as the flash frame
# reports no reflex at all.  Draining costs FLUSH_FRAMES frames per phase and buys
# the guarantee that each input was exposed under the light it claims.
SWEEP_FLUSH = True

# After each measurement the cycle PAUSES on the frame it just measured, with the
# selected pixels highlighted, before the next one starts.  Without it the operator
# never actually sees what is being counted: the measured frame is one frame out of
# a cycle that immediately flips to an ambient blip and washes it off the screen,
# so the red count in the corner is the only evidence of a selection that may well
# have landed on an eyelid.  Every LED is off through the hold — the picture is
# already captured, so there is nothing to gain by carrying on flashing the eye.
REDEYE_SHOW_S = 0.25            # hold the measured frame + its highlight this long
SWEEP_RETRY_S = 0.4             # spacing between retries after an empty measurement.
                                # Without this the retries fire on consecutive fresh
                                # frames and a whole budget is gone in ~2s
SWEEP_LOST_TRIES = 4            # consecutive empty measurements before giving up on
                                # the red reflex and dropping back to centring.  The
                                # approach stage does NOT use this — it has its own,
                                # far more patient budget (APPROACH_TIMEOUT_S), since
                                # at that point there is genuinely no reflex yet.

# Sanity check run on each fresh ambient frame: a CONFIDENT pupil found far from
# where the stage thinks it is means the red-eye ROI is parked on the wrong thing.
# Only a high-confidence ambient detection is trusted here — that is the reliable
# measurement (properly lit, dark-disc contrast), unlike the flash-lit re-detection.
SWEEP_AMBIENT_CONF = 0.45       # "very high confidence" ambient pupil
SWEEP_TRACK_CONF = 0.25         # lower bar for merely FOLLOWING the pupil with the
                                # ROI.  Both lit stages move the eye on purpose, so
                                # a slightly-uncertain new position beats holding a
                                # confidently stale one — and the group gate
                                # (pupil overlap + shape) rejects the ROI landing
                                # anywhere daft, so loose tracking is safe now.
# NOTE: ending a lit stage is RED-EYE-DRIVEN ONLY.  There used to be a second exit
# that aborted when the re-detected pupil drifted sideways, but under flash-only
# light that pupil position is the least trustworthy thing on the frame — it is
# exactly the measurement that falls back to the ambient radius — so it was ending
# runs the red signal said were fine.  The red reflex alone decides.


def measure_red_fraction(flash_bgr, dark_bgr, center, radius, cover_mask=None,
                         ambient_bgr=None, min_px=REDEYE_MIN_PX):
    """How much of the pupil is filled by the flash retroreflection, in [0, 1.5].

    `ambient_bgr` is the FRESH ambient companion frame grabbed moments before the
    flash frame.  When given it is applied as a hard veto (`strict_ambient`): a
    real fundus pixel lies inside the pupil, so it must read dark under ambient
    light; anything brighter is lit tissue that merely went red under the flash.

    Two steps, per the design:
      1.  ``redeye_extract`` selects the red (fundus) pixels around `center`.
      2.  Those pixels are painted BLACK on the flash frame and ``detect_pupil``
          is re-run on the result — with the glow removed the pupil is a plain
          dark disc again, so its area can be measured and the red pixels
          expressed as a fraction of it.

    Step 2 is the fragile half: under flash-only light the whole eye is near-black
    and the disc boundary may not survive.  When the re-detection fails or comes
    back unconfident we fall back to the `radius` measured during the ambient
    centring stage, and say so in the returned `source` so a sweep that never gets
    a real disc is visible rather than silently degraded.

    Returns a dict: fraction, red_px, center (possibly updated), radius, source,
    valid, notes, mask (the selected red pixels, for the live highlight).
    """
    out = {"fraction": 0.0, "red_px": 0, "center": center, "radius": radius,
           "source": "none", "valid": False, "notes": "", "mask": None,
           "extract": None}
    if not CV2_AVAILABLE or flash_bgr is None or dark_bgr is None or center is None:
        out["notes"] = "frames/centre unavailable"
        return out

    redeye = redeye_extract(flash_bgr, dark_bgr, ambient_bgr, center, radius,
                            cover_mask, strict_ambient=ambient_bgr is not None)
    if not redeye.valid or redeye.mask is None:
        out["notes"] = redeye.notes
        return out
    red_px = int(cv2.countNonZero(redeye.mask))
    out["red_px"] = red_px
    out["mask"] = redeye.mask                      # shown by the live highlight
    out["extract"] = redeye.extract                # kept for the session mosaic
    # `min_px` is a parameter because the gaze scan measures on full-resolution
    # frames, where the same reflex covers four times as many pixels.
    if red_px < min_px:
        out["notes"] = f"only {red_px}px red"
        return out

    # Red -> black, then re-detect the pupil as a dark disc.
    blanked = flash_bgr.copy()
    blanked[redeye.mask > 0] = 0
    gray = cv2.cvtColor(blanked, cv2.COLOR_BGR2GRAY)
    detect_pupil(gray, live=True)                  # stashes _LAST_PUPIL/_LAST_CONF
    if _LAST_PUPIL is not None and _LAST_CONF >= REDEYE_PUPIL_CONF_GATE:
        pcx, pcy, pr = _LAST_PUPIL
        out["center"] = (float(pcx), float(pcy))   # follow the rotating eye
        out["radius"] = float(pr)
        out["source"] = "redetect"
    elif radius:
        pr = float(radius)
        out["source"] = "ambient-r"                # fell back to the centring radius
    else:
        out["notes"] = "no pupil disc and no fallback radius"
        return out

    pupil_area = math.pi * float(pr) ** 2
    if pupil_area <= 0:
        out["notes"] = "degenerate pupil area"
        return out
    out["fraction"] = float(min(1.5, red_px / pupil_area))
    out["valid"] = True
    out["notes"] = f"{red_px}px / r={pr:.1f} ({out['source']})"
    return out


class PupilSizeRef:
    """The pupil size the red-eye bar is measured against — see APPROACH_GOOD_PX.

    Holds the radius from the most recent CONFIDENT pupil fit, and turns it into
    the number of red pixels that counts as a decent reflex.  Only ambient-lit
    detections are offered to it: under flash-only light the eye is near-black and
    the fitted circle is the least trustworthy thing on the frame.

    Nothing decays and nothing averages.  A confident fit replaces the stored one
    outright, and everything below the bar is ignored entirely, so a run of poor
    fits leaves the requirement exactly where the last good one put it.
    """

    def __init__(self):
        self.radius = None          # px, from the last confident fit
        self.conf = 0.0
        self.at = 0.0
        self.n = 0                  # confident fits accepted this run

    def reset(self):
        self.__init__()

    def update(self, radius, conf):
        """Offer one AMBIENT detection.  True if it was confident enough to take."""
        if not radius or conf < REDEYE_SIZE_CONF:
            return False
        self.radius = float(radius)
        self.conf = float(conf)
        self.at = time.time()
        self.n += 1
        return True

    def min_px(self):
        """Red pixels required to call the reflex good, for the pupil we last saw."""
        if self.radius is None:
            return APPROACH_GOOD_PX
        area = math.pi * self.radius * self.radius
        return int(max(REDEYE_MIN_PX_FLOOR,
                       min(REDEYE_MIN_PX_CEIL, REDEYE_PUPIL_FRAC * area)))

    def status(self):
        if self.radius is None:
            return f"need>={APPROACH_GOOD_PX}px (no confident pupil yet)"
        return (f"need>={self.min_px()}px (r={self.radius:.0f} "
                f"conf={self.conf:.2f} n={self.n})")


# One eye per streaming run, so one requirement: reset when a run starts.
_PUPIL_SIZE = PupilSizeRef()


def redeye_min_px():
    """Red pixels currently required for "a decent chunk of reflex"."""
    return _PUPIL_SIZE.min_px()


def note_pupil_size(radius, conf):
    """Offer an AMBIENT-lit pupil fit to the red-eye bar.  See PupilSizeRef."""
    return _PUPIL_SIZE.update(radius, conf)


class LitStage:
    """Shared lighting cycle for the flash-lit stages (approach and target search).

    Each measurement is an ambient / dark / flash TRIPLE, so a red patch can be
    checked against a FRESH ambient frame (the fundus must read dark there) and
    measured on a pupil the ambient light has not just constricted:

        DWELL   (flash lit — the fixation target; operator moving, or gaze settling)
          -> AMBIENT (brief blip; flush, then grab the ambient companion and read
                      the pupil off it)
          -> DARK    (all off; the pupil re-opens after the ambient blip, then
                      flush and take a fresh dark subtraction reference)
          -> FLASH   (flash on, flush, and measure the FIRST frame out of the
                      pipeline — see SWEEP_FLASH_S)
          -> next sample (AMBIENT) or, once the samples are in, DWELL

    Each of those three frames is taken straight after a pipeline flush, so it was
    definitely exposed under the light it is being trusted for.

    The cycle is paced by the work, not by a frame counter: the flash is turned on,
    held while the red-eye pass runs to completion, and only then handed back to
    the ambient blip.  Nothing can switch the lighting out from under a measurement
    in progress, because the phase does not advance until the pass returns.

    All waiting is against wall-clock deadlines polled from the streaming loop —
    never time.sleep — so the feed keeps rendering and 's' still aborts.
    """

    PHASE_DWELL = "dwell"
    PHASE_AMBIENT = "ambient"
    PHASE_DARK = "dark"
    PHASE_FLASH = "flash"
    PHASE_SHOW = "show"         # measured: hold the picture up before moving on

    def __init__(self, dwell_s):
        self.phase = self.PHASE_DWELL
        self.deadline = time.time() + dwell_s
        self.done = False               # advance to the next stage
        self.abort = ""                 # non-empty => drop back to centring
        self.spoiled = False            # a blink touched this cycle: throw it away
        self.spoiled_n = 0              # cycles discarded to blinks
        self.cv_s = 0.0                 # seconds the last red-eye pass took
        self.cv_max = 0.0               # ...and the worst so far this stage

    def note_cv(self, seconds):
        """Record how long the red-eye pass took.

        Surfaced in the status line because it sets the cycle's real pace: the
        flash is held until the pass returns, so a slow pass shows up as a slower
        cycle rather than as a truncated or stale measurement.
        """
        self.cv_s = float(seconds)
        self.cv_max = max(self.cv_max, self.cv_s)

    def lighting(self):
        """Which LEDs should be lit right now: "ambient", "off" or "flash"."""
        if self.phase == self.PHASE_AMBIENT:
            return "ambient"
        if self.phase in (self.PHASE_DARK, self.PHASE_SHOW):
            return "off"                # SHOW: the frame is already captured
        return "flash"                  # dwell and the measured frame alike

    def ready(self):
        return time.time() >= self.deadline

    def to_ambient(self, delay=SWEEP_LED_SETTLE_S):
        self.phase = self.PHASE_AMBIENT
        self.deadline = time.time() + delay

    def to_dark(self, delay=SWEEP_DARK_S):
        self.phase = self.PHASE_DARK
        self.deadline = time.time() + delay

    def to_flash(self, delay=SWEEP_FLASH_S):
        self.phase = self.PHASE_FLASH
        self.deadline = time.time() + delay

    def to_show(self, delay=REDEYE_SHOW_S):
        """Hold the just-measured frame on screen before the next cycle starts."""
        self.phase = self.PHASE_SHOW
        self.deadline = time.time() + delay

    def to_dwell(self, delay):
        self.phase = self.PHASE_DWELL
        self.deadline = time.time() + delay


class ApproachState(LitStage):
    """Watches the red reflex while the OPERATOR closes the gap by hand.

    Nothing is driven from here — the operator moves the scope so the pupil
    approaches the LED-cover edge, and this just reports the red pixel count and
    hands over once there is a decent chunk of it (`redeye_min_px()`, held for
    APPROACH_HOLD measurements so one lucky frame cannot advance the stage).  That
    bar is a share of the pupil's own area rather than a fixed count, and is re-read
    on every measurement, so a confident pupil fit part-way through the run takes
    effect at once — see PupilSizeRef.

    The camera instruction comes from the AMBIENT frames only.  Two positions are
    tracked, for two different jobs: `center` is the red-eye ROI and follows the
    eye through every phase of the cycle, while `amb_center` is only ever written
    from an ambient-lit frame and is the one the operator is steered by.  Under
    flash-only light the whole eye is near-black and the re-detected pupil is the
    least trustworthy measurement on the frame — the very one that falls back to a
    stale radius — so a lateral correction read off it sends the scope chasing
    detection noise sideways.  The instruction is latched between ambient readings
    for the same reason: recomputing it on the flash and dark phases made it flip
    in time with the lighting cycle, which is unreadable to move a scope by.
    """

    def __init__(self, cover_side, pupil_center, pupil_radius):
        super().__init__(SWEEP_LED_SETTLE_S)
        self.cover_side = cover_side
        self.center = pupil_center
        # Seeded from the centring stage, which is ambient-lit throughout.
        self.amb_center = pupil_center
        self.amb_fresh = True           # an unused ambient reading is waiting
        self._hint = ""                 # latched instruction + arrow, refreshed
        self._vec = None                # only when that reading arrives
        self.radius0 = pupil_radius
        self.red_px = 0
        self.best_px = 0
        self.good = 0                   # consecutive measurements at/above target
        self.held = False               # latched "hold steady" (see APPROACH_RELEASE_*)
        self.low = 0                    # consecutive readings well below target
        self.seen_reflex = False        # a respectable reflex has appeared at least once
        self.reverse = False            # vertical instruction is currently reversed
        self.flips = 0                  # direction reversals this stage
        self.tracked = 0                # times the ROI followed the ambient pupil
        self.lost = 0                   # consecutive ambient frames with no usable pupil
        self.recover = ""               # non-empty => tell the operator to back off
        self.need_px = redeye_min_px()  # bar this stage is currently working to
        self.started = time.time()

    def submit(self, meas):
        """Feed one measurement; sets `done` once the reflex is strong enough.

        The bar is read fresh every time rather than fixed at construction: a
        confident pupil fit can land at any point in the run and it should take
        effect immediately, not at the next stage.
        """
        need = redeye_min_px()
        self.need_px = need
        self.red_px = meas["red_px"]
        self.best_px = max(self.best_px, self.red_px)
        if meas["valid"] and meas["center"] is not None:
            self.center = meas["center"]
            if meas["radius"]:
                self.radius0 = meas["radius"]
        if self.red_px >= need:
            self.good += 1
            self.held = True            # latch: stop asking for camera movement
            self.low = 0
            self._reflex_here()
            if self.good >= APPROACH_HOLD:
                self.done = True
                return True
        else:
            self.good = 0
            if self.red_px < APPROACH_RELEASE_FRAC * need:
                self.low += 1
                if self.low >= APPROACH_RELEASE_TRIES:
                    self.held = False   # sustained collapse: guide again
                    self.low = 0
                    self._reflex_gone()
            else:
                self.low = 0            # still respectable; keep holding
                self._reflex_here()
        if time.time() - self.started > APPROACH_TIMEOUT_S:
            self.abort = (f"no decent red reflex in {APPROACH_TIMEOUT_S:.0f}s "
                          f"(best {self.best_px}px)")
            return True
        # Hold the measured picture up before the next cycle rather than going
        # straight back to the ambient blip, which would wipe it off the screen.
        self.to_show()
        return False

    def _reflex_here(self):
        """A respectable reflex is being measured: this direction is working."""
        self.seen_reflex = True
        if self.reverse:
            self.reverse = False        # got it back — resume closing on the cover
            self.amb_fresh = True       # re-render the instruction at once

    def _reflex_gone(self):
        """The reflex has collapsed and stayed collapsed — reverse the approach.

        Only once a reflex has actually been SEEN.  The reflex has a sweet spot in
        the cover/pupil gap, and the pupil looks fine on both sides of it, so one
        that appeared and then died is far more likely to be behind the scope than
        further ahead of it.  Before anything has ever appeared there is no
        evidence either way and the stage just keeps closing — which is also what
        stops this from sending the operator backwards the moment they start.

        If reversing does not bring it back either, the same collapse rule fires
        again and flips it once more: the operator is walked back and forth across
        the peak rather than being sent one way for ever.  (Driving past the eye
        ALTOGETHER is a different failure, caught by the lost-pupil check.)
        """
        if not self.seen_reflex:
            return
        self.reverse = not self.reverse
        self.flips += 1
        # A direction change is a decision, not a measurement, so it takes effect
        # without waiting for the next ambient frame — but it is re-rendered from
        # the SAME ambient pupil position, so the lateral half stays ambient-only.
        self.amb_fresh = True

    def check_ambient_pupil(self, center, conf, frame_shape, spoiled=False):
        """Follow the pupil on every ambient frame — and notice when it has gone.

        The operator is moving the scope, so the pupil is travelling across the
        frame the whole time; the ROI has to keep up or the next measurement is
        taken over stale coordinates.

        This is also the OVERSHOOT check (see APPROACH_EDGE_FRAC): a pupil that
        cannot be found, or that has been driven right up against the cover's own
        frame edge, means the scope has gone past the eye.  Called on EVERY ambient
        frame including ones a blink spoiled — a lost pupil reads as a blink, so
        skipping those is exactly how this failure used to stay invisible.  On a
        spoiled cycle the position is not adopted (a lid fits a confident-looking
        ellipse on a crease), but it still counts as the eye being there.
        """
        h = float(frame_shape[0])
        ok = center is not None and conf >= SWEEP_TRACK_CONF
        if ok:
            edge = APPROACH_EDGE_FRAC * h
            y = float(center[1])
            past = (y <= edge if guidance.cover_edge_word(self.cover_side) == "top"
                    else y >= h - edge)
            if past:
                ok = False
        if ok:
            self.lost = 0
            self.recover = ""
            if not spoiled:
                self.center = (float(center[0]), float(center[1]))
                # The one reading the camera instruction is allowed to move on:
                # ambient-lit, so the pupil is a properly exposed dark disc.
                self.amb_center = self.center
                self.amb_fresh = True
                self.tracked += 1
            return
        self.lost += 1
        if self.lost >= APPROACH_ABORT_TRIES:
            self.abort = (f"lost the pupil for {self.lost} cycles - the scope has "
                          "been moved past the eye")
        elif self.lost >= APPROACH_WARN_TRIES:
            self.recover = guidance.back_off_hint(self.cover_side)

    def hint(self):
        return guidance.approach_hint(self.cover_side, self.red_px, redeye_min_px())

    def _steer(self, frame_shape, anchor):
        """The latched (instruction, arrow), recomputed ONLY on a fresh ambient read.

        Both are produced together and cached together, so the words and the arrow
        can never be drawn from different readings.  Between ambient frames — that
        is, through the dark pause and the whole flash dwell — the operator keeps
        being shown exactly what the last ambient frame said.
        """
        if self.amb_fresh or not self._hint:
            self._hint = guidance.camera_hint(self.amb_center, anchor, frame_shape,
                                              close_to=self.cover_side,
                                              reverse=self.reverse)
            self._vec = guidance.camera_arrow(self.amb_center, anchor, frame_shape,
                                              close_to=self.cover_side,
                                              reverse=self.reverse)
            if self.reverse:
                # Say why, or a reversal after all that pushing reads as the
                # guidance changing its mind rather than as new information.
                # Kept short: this line is rendered small beside the fixation
                # point, where a long string crowds the target markers.
                self._hint += " - reflex lost"
            self.amb_fresh = False
        return self._hint, self._vec

    def camera_hint(self, frame_shape, anchor):
        """Which way to move the scope: keep closing on the cover edge, and
        correct any sideways drift off the target column.

        The sideways half is measured from the AMBIENT pupil only (see the class
        docstring) and holds between ambient frames.  The vertical half needs no
        measurement at all — during the approach it is simply "keep closing on the
        cover", which is why only the lateral term was ever at the mercy of the
        flash-lit re-detection.

        Once there IS enough reflex, stop asking for more travel and call for the
        scope to be held — that steadiness is what the gaze scan hands over into,
        and it must begin before the operator has drifted past the sweet spot.
        """
        hint, _ = self._steer(frame_shape, anchor)
        if self.recover:
            return self.recover         # overshot: reverse takes priority over all
        if self.held:
            return "Camera: hold steady - reflex found"
        return hint

    def camera_vector(self, frame_shape, anchor):
        """The same instruction as an arrow — where the camera centre should end up."""
        _, vec = self._steer(frame_shape, anchor)
        if self.recover:
            return guidance.back_off_vector(self.cover_side, frame_shape)
        if self.held:
            return None                 # holding: no arrow to chase
        return vec

    def worth_saving(self, meas):
        """Is this measurement good enough to bank for the mosaic?  (Never here.)

        The approach stage is the operator hunting for the reflex at all, so its
        frames are half-aligned by definition; banking them would fill the session
        with material the mosaic cannot use.  Only the target search saves.
        """
        return False

    def status(self):
        return (f"approach: red {self.red_px}px (best {self.best_px}) "
                f"{self.good}/{APPROACH_HOLD}  {_PUPIL_SIZE.status()}"
                + (f" HELD(-{self.low})" if self.held else "")
                + (" REVERSED" if self.reverse else "")
                + (f" flips={self.flips}" if self.flips else "")
                + f" cv={self.cv_s * 1000:.0f}ms"
                + f" trk={self.tracked}"
                + (f" LOST {self.lost}/{APPROACH_ABORT_TRIES}" if self.lost else ""))


class GazeScan(LitStage):
    """Walks the fixation point in steps and picks the best gaze direction.

    Each stop is ONE CAPTURE: dark while the point travels and the gaze settles,
    then a single flash frame, then dark again.  The eye is therefore never lit
    except at the instant of measurement, and the pupil spends the whole of the
    travel re-dilating — time the scan has to spend anyway waiting for the patient
    to find the new point, so the dilation costs nothing.  A stop's flash frame is
    the same kind of exposure the final capture takes, which is what makes the
    scan's frames comparable with it and with each other.

    There is no ambient blip anywhere in this stage: an ambient frame would
    re-constrict the pupil for no gain, since the veto it feeds is only worth
    having when the pupil position is in doubt, and here the eye is being held
    still by the operator.

    One axis at a time (y first, then a short x).  The point travels to one end of
    its range and back to the other, and the axis is then parked at the profile's
    peak.  Nothing is predicted — the peak is simply read off what was measured.

    Travel is STEP-AND-HOLD (see SCAN_STEP_FRAC): the point glides one step, stops,
    and the capture only fires once the gaze has had SCAN_GAZE_LATENCY_S to arrive
    AND the eye has been dark for SCAN_DILATE_S.  Sweeping continuously measured a
    gaze that was permanently chasing the point and never on it; stopping means
    every profile entry — and every frame banked for the mosaic — belongs to a
    known, settled gaze direction.  The glide between stops is still smooth,
    because a point that teleports loses the gaze entirely.

    The per-stop capture runs through the inherited phases: PHASE_DWELL (dark,
    travelling and settling) -> PHASE_DARK (take the subtraction reference)
    -> PHASE_FLASH (the one measured frame) -> PHASE_SHOW (hold it up) -> DWELL.
    Unlike the approach, the dark reference is refreshed at EVERY stop, so the
    subtraction is always registered against the eye where it is now.

    The vertical sweep is LOPSIDED and much longer than the horizontal one: the
    LED is behind the cover, so there is real reflex to be gained by turning the
    gaze that way, while turning away from it only loses reflex and needs just
    enough travel to show the peak has been passed.

    Consequences of taking no ambient frame, accepted deliberately:
      * no ambient companion, so redeye_extract's strict-ambient veto is off here;
        the pupil-overlap and C-shape/ring gates still apply
      * the pupil is tracked from the flash-frame re-detection alone
    """

    MODE_SETTLE = "settle"
    MODE_SWEEP = "sweep"        # gliding from one stop to the next
    MODE_HOLD = "hold"          # stopped: the gaze arrives and the reflex is read
    MODE_RETURN = "return"      # travelling BACK to the peak, never jumping to it
    MODE_CONFIRM = "confirm"

    def __init__(self, base_target, pupil_center, pupil_radius, frame_shape,
                 cover_side=None, px_scale=1.0):
        super().__init__(SCAN_SETTLE_S)
        # Area ratio between the frames this stage MEASURES on and the display
        # frames every threshold is expressed in.  See worth_saving.
        self.px_scale = float(px_scale)
        h, w = frame_shape[:2]
        self.shape = (h, w)
        self.cover_side = cover_side
        self.base = (float(base_target[0]), float(base_target[1]))
        self.offset = [0.0, 0.0]
        self.center = pupil_center
        self.hold = pupil_center if pupil_center is not None else base_target
        self.radius0 = pupil_radius
        scale = float(min(h, w))
        self.speed = SCAN_SPEED_FRAC_S * scale
        self.step = max(1.0, SCAN_STEP_FRAC * scale)
        self.rng_y_toward = SCAN_RANGE_Y_TOWARD_FRAC * scale
        self.rng_y_away = SCAN_RANGE_Y_AWAY_FRAC * scale
        self.rng_x = SCAN_RANGE_X_FRAC * scale
        self.margin = TARGET_MARGIN_FRAC * scale

        # y first: that is the axis the LED sits on, so it carries most of the
        # reflex variation.  Sweep toward the LED first (see cover_side).
        self.axes = [1, 0]
        self.axis_i = 0
        self.axis = self.axes[0]
        self.leg = 0                    # 0 = to the first end, 1 = to the other
        self.dir = -1 if guidance.cover_edge_word(cover_side) == "top" else 1
        self.origin = 0.0               # offset on this axis when the sweep began
        self.profile = []               # (offset_on_axis, red_px) this axis
        self.mode = self.MODE_SETTLE
        self.return_to = 0.0            # peak this axis is travelling back to
        self.t_last = time.time()
        self.step_end = 0.0             # offset the current glide is heading for
        self.hold_start = 0.0           # when the point stopped (gaze latency runs
                                        # from here)
        self.hold_hard = 0.0            # ...and when to give up waiting for samples
        self.hold_n = 0                 # settled measurements taken at this stop
        self.stops = 0                  # stops completed, for the status line
        self.dark_since = time.time()   # LEDs off since — drives SCAN_DILATE_S
        self.phase = self.PHASE_DWELL   # per-stop capture phase (dark until fired)

        self.best_px = 0
        self.best_offset = [0.0, 0.0]
        self.best_center = pupil_center
        self.best_radius = pupil_radius
        self.confirm = []
        self.dropouts = 0
        self.samples_n = 0
        self.saved = 0
        self.last_save = 0.0
        self.last_px = 0
        self.cam_settled = False
        self.tracked = 0

    # ── lighting: dark throughout, flash only for the measured frame ──
    def lighting(self):
        return "flash" if self.phase == self.PHASE_FLASH else "off"

    def dilated(self, now=None):
        """Has the eye been dark long enough for the pupil to re-open?"""
        now = time.time() if now is None else now
        return (now - self.dark_since) >= SCAN_DILATE_S

    def ready_to_capture(self, now=None):
        """Stopped, gaze arrived, pupil dilated — fire this stop's capture."""
        now = time.time() if now is None else now
        return (self.phase == self.PHASE_DWELL
                and self.mode in (self.MODE_HOLD, self.MODE_CONFIRM)
                and self._settled(now) and self.dilated(now))

    def to_dark_ref(self):
        """Take this stop's subtraction reference (the LEDs are already off)."""
        self.phase = self.PHASE_DARK
        self.deadline = time.time()

    def capture_done(self):
        """One stop's capture is over: back to darkness for the next glide."""
        self.phase = self.PHASE_DWELL
        self.dark_since = time.time()

    def target(self, offset=None):
        """Fixation point in display coords (x mirrored — see TARGET_INVERT_X)."""
        o = self.offset if offset is None else offset
        h, w = self.shape
        ox = -o[0] if TARGET_INVERT_X else o[0]
        return (int(round(min(w - self.margin, max(self.margin, self.base[0] + ox)))),
                int(round(min(h - self.margin, max(self.margin, self.base[1] + o[1])))))

    def _offset_limits(self, axis):
        """(lo, hi) offsets on `axis` whose fixation point is still on screen.

        Without this the offset could keep running past the frame while the drawn
        point sticks at the margin — the profile would then record several
        different offsets for one actual gaze position.
        """
        h, w = self.shape
        if axis == 1:
            return (self.margin - self.base[1], (h - self.margin) - self.base[1])
        if TARGET_INVERT_X:                 # drawn x = base - offset
            return (self.base[0] - (w - self.margin), self.base[0] - self.margin)
        return (self.margin - self.base[0], (w - self.margin) - self.base[0])

    def _range_for_axis(self, axis):
        """Largest travel this axis will ask for (0 => the axis is disabled)."""
        return max(self.rng_y_toward, self.rng_y_away) if axis == 1 else self.rng_x

    def _range_for_leg(self):
        """How far this leg travels — vertical is lopsided toward the cover."""
        if self.axis == 1:
            return self.rng_y_toward if self.leg == 0 else self.rng_y_away
        return self.rng_x

    def _end_for_leg(self):
        """Where this leg of the sweep finishes, on the active axis."""
        sign = self.dir if self.leg == 0 else -self.dir
        end = self.origin + sign * self._range_for_leg()
        lo, hi = self._offset_limits(self.axis)
        return max(lo, min(hi, end))

    def _advance_point(self, dt, end):
        """Glide the fixation point toward `end`; True when it arrives."""
        cur = self.offset[self.axis]
        step = self.speed * dt
        if abs(end - cur) <= step:
            self.offset[self.axis] = end
            return True
        self.offset[self.axis] = cur + math.copysign(step, end - cur)
        return False

    def _next_stop(self):
        """The offset one step further along this leg (clamped to the leg's end)."""
        end = self._end_for_leg()
        cur = self.offset[self.axis]
        if abs(end - cur) <= self.step:
            return end
        return cur + math.copysign(self.step, end - cur)

    def _begin_hold(self, now):
        """Stop at this position and let the patient's gaze catch up."""
        self.mode = self.MODE_HOLD
        self.hold_start = now
        self.hold_hard = now + SCAN_HOLD_MAX_S
        self.hold_n = 0

    def _end_hold(self):
        """Move on from a completed stop: next step, turn round, or finish the axis."""
        self.stops += 1
        if abs(self.offset[self.axis] - self._end_for_leg()) > 1e-6:
            self.step_end = self._next_stop()       # more of this leg to walk
            self.mode = self.MODE_SWEEP
        elif self.leg == 0:
            self.leg = 1                            # turn round, sweep the other way
            self.step_end = self._next_stop()
            self.mode = self.MODE_SWEEP
        else:
            self._finish_axis()

    def _settled(self, now=None):
        """Is the point stopped AND the gaze given time to arrive on it?"""
        now = time.time() if now is None else now
        return (self.mode in (self.MODE_HOLD, self.MODE_CONFIRM)
                and now >= self.hold_start + SCAN_GAZE_LATENCY_S)

    def _finish_axis(self):
        """Pick this axis's peak and start travelling BACK to it.

        The point is never teleported: a jump loses the patient's gaze, which is
        the one thing a smooth sweep exists to avoid.  MODE_RETURN walks it there
        at scan speed instead.
        """
        if len(self.profile) >= SCAN_MIN_SAMPLES:
            pos = np.array([p[0] for p in self.profile], float)
            val = np.array([p[1] for p in self.profile], float)
            k = min(SCAN_SMOOTH | 1, len(val) if len(val) % 2 else len(val) - 1)
            if k >= 3:                  # median-smooth: single frames are noisy
                pad = np.pad(val, k // 2, mode="edge")
                val = np.median(np.lib.stride_tricks.sliding_window_view(pad, k),
                                axis=-1)
            j = int(np.argmax(val))
            self.return_to = float(pos[j])
            self.best_px = max(self.best_px, int(val[j]))
        else:
            self.return_to = self.origin             # nothing usable; put it back

        self.profile = []
        self.mode = self.MODE_RETURN

    def _next_axis(self):
        """Called once the point has arrived back at this axis's peak."""
        self.best_offset = list(self.offset)
        self.axis_i += 1
        # Skip any axis given a zero range — that is how side-to-side is turned
        # off, and sweeping it would otherwise burn a settle for nothing.
        while (self.axis_i < len(self.axes)
               and self._range_for_axis(self.axes[self.axis_i]) <= 0.0):
            self.axis_i += 1
        if self.axis_i >= len(self.axes):
            self.mode = self.MODE_CONFIRM
            # The point has just walked back to the peak, so the gaze is still
            # moving: confirmation readings wait out the same settle as a stop.
            self.hold_start = time.time()
            self.deadline = self.hold_start + SCAN_SETTLE_S
            return
        self.axis = self.axes[self.axis_i]
        self.leg = 0
        self.dir = 1                    # x has no preferred side
        self.origin = self.offset[self.axis]
        self.mode = self.MODE_SETTLE
        self.hold_start = time.time()   # the gaze has just travelled; let it settle
        self.deadline = self.hold_start + SCAN_SETTLE_S

    def tick(self):
        """Advance the fixation point.  Called on EVERY loop iteration.

        Motion is deliberately separate from measurement: measurements only
        happen every _LIVE_DETECT_SKIP-th frame, and driving the point from those
        made it lurch ~6px at 6Hz instead of gliding.  The patient has to be able
        to follow it, so it moves at display frame rate.
        """
        now = time.time()
        dt = max(0.0, min(0.2, now - self.t_last))
        self.t_last = now

        if self.mode == self.MODE_SETTLE:
            if now >= self.deadline:
                self.origin = self.offset[self.axis]
                self.step_end = self._next_stop()
                self.mode = self.MODE_SWEEP
        elif self.mode == self.MODE_SWEEP:
            if self._advance_point(dt, self.step_end):
                self._begin_hold(now)
        elif self.mode == self.MODE_HOLD:
            # Leave once this stop's capture has been taken — but never wait past
            # SCAN_HOLD_MAX_S, or a stop whose capture keeps failing parks the scan
            # here for ever.  The capture itself is fired by the streaming loop,
            # which owns the camera; see ready_to_capture().
            if self.phase == self.PHASE_DWELL and (self.hold_n >= 1
                                                   or now >= self.hold_hard):
                self._end_hold()
        elif self.mode == self.MODE_RETURN:
            cur = self.offset[self.axis]
            step = self.speed * dt
            if abs(self.return_to - cur) <= step:
                self.offset[self.axis] = self.return_to
                self._next_axis()
            else:
                self.offset[self.axis] = cur + math.copysign(
                    step, self.return_to - cur)

    def submit(self, meas):
        """Record one stop's capture.  Called ONCE per stop; does not move the point."""
        now = time.time()

        if not meas["valid"]:
            self.dropouts += 1
            self.last_px = 0
            # Count the attempt even though nothing was recorded, or a stop whose
            # capture keeps coming back empty would be retried until the watchdog.
            self.hold_n += 1
            if self.dropouts >= SCAN_DROPOUT_TRIES:
                self.abort = f"red reflex lost ({meas['notes']})"
            return False
        self.dropouts = 0
        self.last_px = meas["red_px"]
        self.samples_n += 1
        if meas["center"] is not None:
            self.center = meas["center"]
            self.tracked += 1
        if meas["radius"]:
            self.radius0 = meas["radius"]

        if self.mode in (self.MODE_SETTLE, self.MODE_RETURN, self.MODE_SWEEP):
            # Point in motion (or about to be): the gaze is somewhere between two
            # positions, so this reading belongs to neither.  tick() handles the
            # motion; nothing is logged here.
            return False

        if self.mode == self.MODE_CONFIRM:
            self.confirm.append((meas["red_px"], meas["center"], meas["radius"]))
            if len(self.confirm) >= SCAN_CONFIRM_FRAMES:
                px = sorted(c[0] for c in self.confirm)
                med = px[len(px) // 2]
                pick = min(self.confirm, key=lambda c: abs(c[0] - med))
                self.best_px = max(self.best_px, int(med))
                self.best_center, self.best_radius = pick[1], pick[2]
                self.done = True
            return self.done

        # MODE_HOLD — one capture, at one settled gaze direction.
        self.hold_n += 1
        self.profile.append((self.offset[self.axis], float(meas["red_px"])))
        return False

    def worth_saving(self, meas):
        """Bank confident reflexes as the scan goes — see SESSION_SAVE_*.

        A stop produces exactly one capture, at one settled gaze direction, so this
        is really just the quality filter: the frame is already known to belong to a
        known gaze angle rather than to a smear between two.

        The size bar is scaled with the frame, because the scan captures at the full
        sensor resolution while SESSION_SAVE_MIN_PX is expressed in the display
        pixels every other stage counts in — four times the pixels for the same
        reflex, so an unscaled bar would let anything through.
        """
        if not meas["valid"] or meas["extract"] is None:
            return False
        if meas["red_px"] < SESSION_SAVE_MIN_PX * self.px_scale:
            return False
        if SESSION_SAVE_REQUIRE_DISC and meas["source"] != "redetect":
            return False
        now = time.time()
        if now - self.last_save < SESSION_SAVE_GAP_S:
            return False
        self.last_save = now
        self.saved += 1
        return True

    def check_ambient_pupil(self, center, conf, frame_shape, spoiled=False):
        """Unused here — this stage never lights the ambient LEDs."""
        return

    def camera_hint(self, frame_shape):
        text = guidance.camera_hint(self.center, self.hold, frame_shape,
                                    settled=self.cam_settled)
        self.cam_settled = "hold steady" in text
        return text

    def status(self):
        ax = "y" if self.axis else "x"
        if self.mode == self.MODE_SWEEP:
            span = self._end_for_leg() - self.offset[self.axis]
            where = f"moving {ax} leg{self.leg + 1} {span:+.0f}px to go"
        elif self.mode == self.MODE_HOLD:
            if self.phase != self.PHASE_DWELL:
                where = f"CAPTURING {ax} stop{self.stops + 1} ({self.phase})"
            elif not self._settled():
                where = f"stop{self.stops + 1} {ax}: gaze arriving"
            elif not self.dilated():
                left = max(0.0, SCAN_DILATE_S - (time.time() - self.dark_since))
                where = f"stop{self.stops + 1} {ax}: dilating {left:.1f}s"
            else:
                where = f"stop{self.stops + 1} {ax}: ready"
        elif self.mode == self.MODE_RETURN:
            where = f"returning {ax} to peak"
        elif self.mode == self.MODE_CONFIRM:
            where = f"confirming ({len(self.confirm)}/{SCAN_CONFIRM_FRAMES})"
        else:
            where = f"settling before {ax}"
        return (f"scan: {where} red={self.last_px}px best={self.best_px}px "
                f"n={self.samples_n} saved={self.saved}")


# ───────── SESSION GROUPING (for red-eye mosaicing) ─────────
# One streaming session = one eye alignment run = one group of captures whose
# red-eye extracts can be stitched into a wider fundus view.  Each take's RAW
# extract (isolated pixels on black, NOT the green highlight overlay) and its mask
# are written under LOCAL_SESSION_DIR/session_<stamp>/, and the same pair is
# uploaded so the receiver can group them too.
LOCAL_SESSION_DIR = "/tmp/eyevu_sessions"
SESSION_MIN_FOR_MOSAIC = 2       # takes needed before a mosaic can be built


def session_dir(session):
    """Local directory holding one session's red-eye extracts."""
    return os.path.join(LOCAL_SESSION_DIR, f"session_{session}")


def session_shots(session):
    """Paths of this session's saved extracts, in capture order."""
    d = session_dir(session)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith("redeye_") and f.endswith("_extract.png"))


def save_session_redeye(session, redeye):
    """Write one CAPTURE's raw red-eye extract + mask into the session folder."""
    if not (redeye is not None and redeye.valid):
        return None
    return save_session_frame(session, redeye.extract, redeye.mask)


def save_session_frame(session, extract, mask):
    """Write one red-eye extract + mask into the session folder.

    The EXTRACT is saved, not the overlay: the mosaic needs the actual fundus
    pixels on black, and the highlight tint would poison feature matching.

    Used both by the final capture and — during the target search — by every
    measurement whose reflex is confidently inside the pupil, so a session
    accumulates many overlapping views of the fundus instead of just one per
    take.  That is exactly the material the mosaic wants.
    Returns the extract's path, or None.
    """
    if not CV2_AVAILABLE or extract is None or mask is None:
        return None
    d = session_dir(session)
    try:
        os.makedirs(d, exist_ok=True)
        idx = len(session_shots(session)) + 1
        stamp = datetime.now().strftime("%H%M%S")
        base = os.path.join(d, f"redeye_{idx:02d}_{stamp}")
        cv2.imwrite(f"{base}_extract.png", extract)
        cv2.imwrite(f"{base}_mask.png", mask)
        print(f"  session {session}: saved shot {idx} "
              f"({int(cv2.countNonZero(mask))}px)")
        # Ship the same pair so the dev machine can mosaic the session too.
        if TRANSFER_ENABLED:
            for suffix, im in (("extract.png", extract),
                               ("mask.png", mask)):
                ok, buf = cv2.imencode(".png", im)
                if ok:
                    _post_to_receiver(f"session_{session}",
                                      f"redeye_{idx:02d}_{stamp}_{suffix}",
                                      buf.tobytes())
        return f"{base}_extract.png"
    except (OSError, cv2.error) as e:                # noqa: BLE001
        print(f"[SESSION ERROR] could not save red-eye extract: {e}")
        return None


def build_session_mosaic(session, show=True):
    """Stitch this session's red-eye extracts.  Returns (mosaic, info) or (None, {}).

    Runs the shared mosaic.py, so the Pi and the dev machine produce the same
    result from the same inputs.
    """
    try:
        import mosaic
    except ImportError as e:                         # noqa: BLE001
        print(f"[MOSAIC] unavailable: {e}")
        return None, {}
    d = session_dir(session)
    shots = session_shots(session)
    if len(shots) < SESSION_MIN_FOR_MOSAIC:
        print(f"[MOSAIC] need at least {SESSION_MIN_FOR_MOSAIC} captures, "
              f"have {len(shots)}.")
        return None, {}
    print(f"[MOSAIC] stitching {len(shots)} red-eye extract(s) from session {session}...")
    out = os.path.join(d, "mosaic.png")
    mos, info, paths = mosaic.build(d, out)
    if mos is None:
        print("[MOSAIC] failed - no reliable alignment between the captures.")
        return None, info
    print(f"[MOSAIC] used {info['used']}/{len(paths)} "
          f"(detector {info['detector']}); saved {out}")
    if show and CV2_AVAILABLE:
        vis = mos
        if vis.shape[0] > 720:
            s = 720.0 / vis.shape[0]
            vis = cv2.resize(vis, (int(vis.shape[1] * s), 720))
        cv2.imshow("Red-eye mosaic", vis)
        cv2.waitKey(1)
    if TRANSFER_ENABLED:
        try:
            ok, buf = cv2.imencode(".png", mos)
            if ok:
                _post_to_receiver(f"session_{session}", "mosaic.png", buf.tobytes())
        except cv2.error:
            pass
    return mos, info


def browse_session_shots(session, window="Session captures"):
    """Step through this session's captures ONE AT A TIME, at full size.

    Blocks the live feed while open — that is the point: a tiled sheet is too
    small to judge a reflex by, so this shows each frame as captured.

        SPACE / n / right   next        m   extract <-> mask
        p / left            previous    ESC / q / v   close
    """
    if not CV2_AVAILABLE:
        return False
    try:
        import mosaic
    except ImportError as e:                         # noqa: BLE001
        print(f"[VIEW] unavailable: {e}")
        return False
    imgs, masks, paths = mosaic.load_session(session_dir(session))
    if not imgs:
        print(f"[VIEW] session {session} has no captures yet.")
        return False

    i = 0
    show_mask = False
    print(f"[VIEW] {len(imgs)} capture(s) - SPACE/n/right=next, p/left=prev, "
          f"m=mask, ESC/q=close")
    try:
        while True:
            src = imgs[i]
            if show_mask and masks[i] is not None:
                vis = cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR)
            else:
                vis = src.copy()
            if vis.shape[0] > 900:                   # only shrink if it must
                s = 900.0 / vis.shape[0]
                vis = cv2.resize(vis, (int(vis.shape[1] * s), 900))
            px = int(cv2.countNonZero(masks[i])) if masks[i] is not None else -1
            label = (f"[{i + 1}/{len(imgs)}] "
                     f"{os.path.basename(paths[i]).replace('_extract.png', '')}"
                     + (f"  {px}px" if px >= 0 else "")
                     + ("  MASK" if show_mask else ""))
            for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(vis, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, th, cv2.LINE_AA)
            cv2.imshow(window, vis)
            k = cv2.waitKeyEx(0)
            if k in (27, ord('q'), ord('v')):
                break
            if k in (32, ord('n'), 65363):           # space / n / right
                i = (i + 1) % len(imgs)
            elif k in (ord('p'), 65361):             # p / left
                i = (i - 1) % len(imgs)
            elif k == ord('m'):
                show_mask = not show_mask
    finally:
        try:
            cv2.destroyWindow(window)
        except Exception:                            # noqa: BLE001
            pass
    return True


def show_session_keypoints(session):
    """Contact sheet of this session's extracts with their SIFT keypoints drawn.

    The diagnostic for a session that will not mosaic.  Each patch is shown as the
    detector sees it — cropped, enlarged, contrast-boosted — with a circle per
    keypoint at the scale it was found and a radius line for its orientation, and
    the count in the corner.  What usually explains a failed stitch is visible
    here: too few keypoints to match with, or plenty of them but all crowded on
    the patch's own outline, which is a different shape in every capture and so
    matches nothing.
    """
    try:
        import mosaic
    except ImportError as e:                         # noqa: BLE001
        print(f"[KEYPOINTS] unavailable: {e}")
        return False
    d = session_dir(session)
    out = os.path.join(d, "keypoints.png")
    sheet = mosaic.show_keypoints(d, out, show=CV2_AVAILABLE)
    if sheet is None:
        print(f"[KEYPOINTS] session {session} has no captures yet.")
        return False
    if TRANSFER_ENABLED:
        try:
            ok, buf = cv2.imencode(".png", sheet)
            if ok:
                _post_to_receiver(f"session_{session}", "keypoints.png",
                                  buf.tobytes())
        except cv2.error:
            pass
    return True


def show_session_shots(session):
    """Open a contact sheet of this session's constituent extracts."""
    try:
        import mosaic
    except ImportError as e:                         # noqa: BLE001
        print(f"[MOSAIC] unavailable: {e}")
        return False
    imgs, _, paths = mosaic.load_session(session_dir(session))
    if not imgs:
        print(f"[MOSAIC] session {session} has no captures yet.")
        return False
    sheet = mosaic.contact_sheet(
        imgs, labels=[os.path.basename(p).replace("_extract.png", "")
                      for p in paths])
    if sheet is None:
        return False
    if CV2_AVAILABLE:
        if sheet.shape[0] > 800:
            s = 800.0 / sheet.shape[0]
            sheet = cv2.resize(sheet, (int(sheet.shape[1] * s), 800))
        cv2.imshow("Session captures", sheet)
        cv2.waitKey(1)
    print(f"[MOSAIC] session {session}: {len(imgs)} capture(s) - "
          + ", ".join(os.path.basename(p) for p in paths))
    return True


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
                     dark_array=None, redeye=None, pupil_center=None, session=None):
    """Stage the raw capture frames locally and POST them to the dev machine.

    Writes LOCAL_STAGING_DIR/capture_<timestamp>/{ambient.jpg, flash.jpg, both.jpg,
    dark.jpg, detect.jpg, redeye_overlay.jpg, redeye_extract.jpg, redeye_mask.png,
    meta.json}, then uploads each via HTTP POST to receiver.py.  The raw frames
    (ambient/flash/both/dark) are saved un-annotated so the harness can re-run
    detection + red-eye extraction cleanly; `detect.jpg` and the `redeye_*` files
    are the Pi-side overlays/outputs.  meta.json is uploaded LAST so the receiver
    only triggers once the whole folder has arrived.

    `ambient_array` may be None (the sweep-triggered dark+flash capture): meta's
    `has_ambient` flag tells the receiver not to wait for ambient.jpg and to skip
    the detectors that need it.

    Any failure is reported on the Pi but never raised — a transfer problem
    must not interrupt capture or streaming.
    """
    if not TRANSFER_ENABLED:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(LOCAL_STAGING_DIR, f"capture_{stamp}")
    fnames = ["flash.jpg"]

    # ── stage the frames locally ──
    try:
        os.makedirs(folder, exist_ok=True)
        if ambient_array is not None:
            Image.fromarray(ambient_array).save(os.path.join(folder, "ambient.jpg"))
            fnames.append("ambient.jpg")
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
            "has_ambient":   ambient_array is not None,
            "session":       session,      # groups a streaming run for mosaicing
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
    
    _drain_frames(picam2)              # flush frames exposed before LED on
    flash_array = picam2.capture_array()
    time.sleep(FLASH_PRE_DELAY)

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


def capture_flash_pair(picam2, live_center_frac=None, live_radius_frac=None,
                       session=None):
    """Capture just the DARK + FLASH frames at the end of the target search.

    The search has already aligned the eye under a held flash and the ambient LEDs
    are off, so there is nothing left for the ambient / flash+ambient frames to
    contribute — and skipping them keeps the eye still and avoids re-lighting the
    ambient LEDs mid-alignment.  redeye_extract only needs the dark reference; the
    ambient frame is a soft prior it tolerates as None.

    Everything goes DARK for AUTOCAP_DILATE seconds first, so the pupil dilates
    before the shot that matters — the search has just held the flash on the eye,
    which constricts it, and a wider pupil passes more fundus light.  The dark
    reference frame is taken at the END of that window: it costs nothing extra and
    it is pupil-size-matched to the flash frame that follows immediately after.

    `session` groups this take with the others from the same streaming session, so
    their red-eye extracts can be mosaicked together later.

    The flash is expected to be ON on entry and is left OFF on return.
    """
    print(f"Capture -> dark {AUTOCAP_DILATE:.1f}s to dilate, then flash...")

    apply_camera_settings(picam2, FLASH_GAIN)
    time.sleep(0.02)

    # ── 0) DARK (all off) — dilate, then take the subtraction reference ──
    flash_off()
    ambient_off()
    time.sleep(AUTOCAP_DILATE)         # pupil dilation, in the dark
    _drain_frames(picam2)              # flush frames still exposed under the flash
    dark_array = picam2.capture_array()

    # ── 1) FLASH — the retroreflection, pupil still dilated ──
    flash_on()
    time.sleep(FLASH_PRE_DELAY)
    _drain_frames(picam2)              # flush frames exposed before the LED came on
    flash_array = picam2.capture_array()
    flash_off()

    apply_camera_settings(picam2, LIVE_GAIN)

    detect_bgr = None
    redeye = None
    pupil_center = None
    img = None
    if CV2_AVAILABLE:
        try:
            detect_bgr, redeye, pupil_center = _redeye_capture(
                None, flash_array, dark_array, live_center_frac, live_radius_frac)
            img = Image.fromarray(cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:                     # noqa: BLE001
            import traceback
            print(f"[REDEYE ERROR] {e!r}")
            traceback.print_exc()
    if img is None:
        img = process_image(flash_array, [])

    # Keep the raw red-eye extract for this session's mosaic BEFORE transferring,
    # so a network problem cannot cost us the material.
    if session and redeye is not None and redeye.valid:
        save_session_redeye(session, redeye)

    transfer_capture(None, flash_array, None, detect_bgr, dark_array,
                     redeye, pupil_center, session=session)
    img.save(PHOTO_PATH)

    if REDEYE_PREVIEW and CV2_AVAILABLE and redeye is not None \
            and redeye.overlay is not None:
        ro = redeye.overlay
        if ro.shape[0] > 720:
            s = 720.0 / ro.shape[0]
            ro = cv2.resize(ro, (int(ro.shape[1] * s), 720))
        cv2.imshow("Red-eye extraction", ro)
        cv2.waitKey(1)

    print("Captured (flash + dark).")


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

    `ambient_array` may be None — the sweep-triggered capture takes only the dark
    and flash frames, and redeye_extract treats the ambient frame as an optional
    soft prior.

    Returns (overlay_bgr, redeye, pupil_center) — the red-eye highlight overlay (or
    a brightened flash if no region is known), the RedeyeResult (or None), and the
    full-frame pupil coords used.
    """
    rot = _CV2_ROTATIONS[LIVE_ROTATION]
    amb_bgr = (cv2.cvtColor(ambient_array, cv2.COLOR_RGB2BGR)
               if ambient_array is not None else None)
    flash_bgr = cv2.cvtColor(flash_array, cv2.COLOR_RGB2BGR)
    dark_bgr = (cv2.cvtColor(dark_array, cv2.COLOR_RGB2BGR)
                if dark_array is not None else None)
    if rot is not None:
        if amb_bgr is not None:
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
    """Set which image edge the LED cover sits on: "top" or "bottom".

    The two rig orientations differ by a 180° flip, so switching side adds 180° to
    LIVE_ROTATION; the cover mask follows (load_cover_mask) and the detection
    geometry reads the side back off the mask, so everything stays consistent.
    Relative (not a fixed rotation per side) so any arrow-key rotation offset the
    operator has dialled in survives a side switch.
    """
    global COVER_SIDE, LIVE_ROTATION
    side = "bottom" if str(side).lower().startswith("b") else "top"
    if side != COVER_SIDE:
        LIVE_ROTATION = (LIVE_ROTATION + 2) % 4
        COVER_SIDE = side
    return COVER_SIDE


def _prompt_cover_orientation():
    """Ask which edge the LED cover sits on before streaming.

    Choosing the opposite side rotates the feed 180°.  A missing terminal or blank
    input keeps the default.  Switch later live with `f` / `cover_side`.
    """
    try:
        ans = input("LED cover at [t]op or [b]ottom (default)? ").strip().lower()
    except EOFError:
        ans = ""
    _set_cover_side("top" if ans.startswith("t") else "bottom")
    print(f"Cover side: {COVER_SIDE}  (rotation {LIVE_ROTATION * 90}°).\n")


def streaming_mode(picam2):
    """Live video feed.

    SPACE (hold) / r — flash LED (GPIO 17), hold / lock
    a               — toggle ambient LED (GPIO 22/27/23/6/26/16)
    p               — toggle live Orlosky pupil detection
    g               — start/stop the staged alignment session (arms auto-capture):
                        centre (pupil -> camera centre) -> sweep -> capture
                        (a find_cover stage runs first when USE_LIVE_COVER is on)
    k               — authorise the next auto-capture (after one fires)
    t               — toggle PC image transfer (HTTP-POST to receiver.py)
    f               — flip the cover side (top<->bottom, rotates the feed 180°)
    SHIFT (or h)    — toggle the live red-eye highlight (on by default)
    i               — hide/show the camera picture on the live feed: guidance aids
                      and instructions stay, the image, pupil overlay and red-eye
                      tint go.  Everything keeps processing underneath, and the
                      captures/mosaic/browser still show real imagery.
    o               — PAUSE the feed and mosaic this session's red-eye captures
    v               — view this session's constituent captures (contact sheet)
    x               — show every capture's SIFT keypoints (why a session will not
                      stitch: too few features, or all of them on the patch rim)
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
    print("SPACE=flash | r=lock | a=ambient | p=pupil-detect | g=staged-guidance | "
          "k=authorise-autocap | SHIFT/h=redeye-highlight | i=hide-image | "
          "o=mosaic | v=view-shots | x=keypoints | "
          "t=pc-transfer | f=flip-cover-side | c=calibrate-cover | ←/→=rotate | "
          "ENTER/e=capture | s=exit\n")

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

    # ── staged alignment session ('g') ──
    # _stage is the single source of truth for where the session is; None = off.
    #   find_cover — ambient on, locate the dark LED-cover outline live
    #                (SKIPPED while USE_LIVE_COVER is off)
    #   centre     — ambient on, drive the pupil onto the target (strict in y)
    #   sweep      — ambient OFF / flash HELD, step the gaze and track the red reflex
    #   capture    — fire the dark+flash pair at the peak
    START_STAGE = (guidance.STAGE_FIND_COVER if USE_LIVE_COVER
                   else guidance.STAGE_CENTRE)
    _stage = None
    _live_guide = guidance.GuidanceTracker(dead_x_frac=CENTRE_DEAD_X_FRAC,
                                           dead_y_frac=CENTRE_DEAD_Y_FRAC)
    _cover = CoverTracker()
    _sweep = None                          # ApproachState / TargetSearch while lit
    _show_frame = None                     # measured frame, held up during PHASE_SHOW
    _sweep_dark = None                     # all-off reference frame (display coords)
    _sweep_dark_full = None                # ...and the scan's full-resolution one
    _sweep_ambient = None                  # fresh ambient companion for the veto
    _sweep_amb_on = False                  # ambient LEDs currently driven BY the sweep
    # Blink detection runs ONLY in the red-eye stages, on their ambient frames.
    _blink = BlinkDetector()
    # New run, possibly a new eye: the red-eye bar goes back to its default until
    # this eye's pupil has been fitted confidently at least once.
    _PUPIL_SIZE.reset()
    _live_center = None
    _live_radius = None                    # live pupil radius (disp px) -> red-eye ROI
    _autocap_armed = False                 # auto-capture at the sweep peak ('g'/'k')
    _centred_count = 0
    _last_autocap = 0.0
    _detect_counter = 0
    _overlay_cache = None
    _pre_sweep_ambient = False             # ambient state to restore if the sweep aborts

    # ── capture session (for red-eye mosaicing) ──
    # One streaming run groups its takes under a single id, so their red-eye
    # extracts can be stitched together with 'o' without picking files by hand.
    _session = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Capture session: {_session}  "
          f"('o' = mosaic the session, 'v' = view its captures)")

    # ── live red-eye highlight (SHIFT toggles) ──
    # Whenever the flash is lit and an all-off reference frame is on hand, the
    # detected red-eye pixels are tinted straight onto the feed, so the operator
    # can see what the sweep is measuring instead of trusting the number.
    redeye_hl_on = REDEYE_HIGHLIGHT_DEFAULT

    # ── blanked feed ('i') ──
    # Guidance only, on black: the picture, the pupil overlay and the red-eye tint
    # are not DRAWN, but every measurement behind them still runs on `clean`, so the
    # sweep, the auto-capture and the session saving are bit-for-bit unaffected.
    image_hidden = LIVE_IMAGE_HIDDEN_DEFAULT
    _live_dark = None                      # most recent all-LEDs-off frame (disp)
    _redeye_mask = None                    # last computed red-eye selection
    _hl_counter = 0                        # ticks EVERY frame (unlike _detect_counter)
    exit_reason = "unknown"

    def _grab_full():
        """One fresh frame at FULL sensor resolution, in display orientation.

        The gaze scan measures on these, so its saved extracts are the same
        resolution as the final capture's — the mosaic stitches both together, and
        patches that differ in scale by 2x register far less reliably than patches
        that do not.  Everything else works on the resized copy below.
        """
        a = picam2.capture_array()
        f = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        r = _CV2_ROTATIONS[LIVE_ROTATION]
        if r is not None:
            f = cv2.rotate(f, r)
        return f

    def _to_disp(f):
        """Resize a full-resolution frame to the display size."""
        return cv2.resize(f, (DISPLAY_W, DISPLAY_H) if LIVE_ROTATION % 2 == 1
                          else (DISPLAY_H, DISPLAY_W))

    def _grab_disp():
        """One fresh frame in display orientation/size (same path as the loop's)."""
        return _to_disp(_grab_full())

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

            # The un-annotated frame.  The staged session measures on THIS, never on
            # `disp` — which picks up the pupil overlay and the guidance drawing as
            # the iteration goes on, and would poison the red-pixel count.
            clean = disp

            # 'i' blanks only what is DRAWN: `clean` still carries the real frame, so
            # everything measured below is identical either way.  Swap in a black
            # canvas of the same size and let the guidance draw onto that.
            if image_hidden:
                disp = np.zeros_like(clean)
            elif (_show_frame is not None and _sweep is not None
                    and _sweep.phase == _sweep.PHASE_SHOW):
                # Freeze on the frame the measurement was actually taken from, so
                # the highlight below marks the pixels that were counted rather than
                # floating over a later, differently-lit frame.  `clean` is left
                # live, so the dark-reference tracking above is unaffected.
                disp = _show_frame.copy()

            # Keep the most recent all-LEDs-off frame as the red-eye subtraction
            # reference, so the live highlight works outside the sweep too (e.g.
            # while holding SPACE).  Both LEDs off => this frame IS a dark frame.
            _hl_counter += 1
            # NB: test the ACTUAL lamp state, not the operator's `ambient_led_on`
            # toggle — during the sweep the phase machine drives the ambient LEDs
            # independently, and an ambient-lit frame stored as the "dark"
            # reference would wreck both the highlight and the red-shift subtraction.
            amb_lit = ambient_led_on or _sweep_amb_on
            if not led_on and not amb_lit:
                _live_dark = clean             # genuinely all-off => a dark frame
                if _stage not in _LIT_STAGES:
                    _redeye_mask = None
            elif not led_on and _stage not in _LIT_STAGES:
                _redeye_mask = None            # flash off => no retroreflection

            # Live pupil detection (ridge) — recompute every N frames; runs when
            # the pupil overlay (p) OR the staged session (g) is on.  Skipped during
            # the sweep: the frame is flash-lit and near-black there, and the sweep
            # runs its own detection inside measure_red_fraction.
            if swirski_live_on or _stage is not None:
                _detect_counter += 1
                fresh = (_overlay_cache is None
                         or _detect_counter % _LIVE_DETECT_SKIP == 0)
                if fresh and _stage not in _LIT_STAGES:
                    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
                    _overlay_cache = detect_pupil(gray, live=True)
                    if _LAST_PUPIL is not None:
                        _live_center = (_LAST_PUPIL[0], _LAST_PUPIL[1])
                        _live_radius = _LAST_PUPIL[2]
                        # Ambient light (no lit stage is running here), so this fit
                        # is a candidate for sizing the red-eye bar.
                        note_pupil_size(_LAST_PUPIL[2], _LAST_CONF)
                    else:
                        _live_center = None
                        _live_radius = None
                # Detection above always runs; only the drawing answers to 'i'.
                if swirski_live_on and _overlay_cache and not image_hidden:
                    disp = _draw_overlays(disp.copy(), _overlay_cache)

            # ───────── STAGED ALIGNMENT SESSION ─────────
            # Everything below works in DISPLAY coords (the cover mask, the target,
            # the pupil and the red-eye ROI all share the resized frame), so the
            # overlay lines up with what the operator sees.
            if _stage is not None:
                extra = _cover.status() if USE_LIVE_COVER else ""

                # ── find_cover: locate the dark cover outline, no calibration ──
                if _stage == guidance.STAGE_FIND_COVER:
                    if fresh:
                        _cover.update(clean, _live_center)
                    if _cover.found:
                        if (_cover.side != COVER_SIDE
                                and not _cover.from_calibration):
                            print(f"WARNING: cover detected on the {_cover.side} but "
                                  f"cover_side is '{COVER_SIDE}' - the rig is flipped "
                                  f"relative to the prompt.  Press 'f' to correct.")
                        _stage = guidance.STAGE_CENTRE
                        _live_guide.reset()
                        _centred_count = 0
                        if not ambient_led_on:
                            ambient_on(); ambient_led_on = True
                        print(f"Stage -> centre (cover on the {_cover.side})")

                # With live cover detection off there is no mask: the target is the
                # camera centre and nothing downstream vetoes on the cover.
                cover_disp = _cover.mask if USE_LIVE_COVER else None
                tgt = guidance.target_point(disp.shape, cover_disp,
                                            GUIDANCE_TARGET_MODE)

                # ── centre: pupil into the box; patient fixates the target ──
                if _stage == guidance.STAGE_CENTRE:
                    lg = _live_guide.update(_live_center, disp.shape, tgt, 1.0, "live")
                    # The patient's line never changes; the operator's camera move
                    # goes underneath it, so both read from the same place.
                    lg.hint = guidance.camera_hint(_live_center, tgt, disp.shape)
                    lg.instruction = guidance.INSTRUCTION_LOOK
                    if fresh:
                        _centred_count = (_centred_count + 1
                                          if lg.state == "centred" else 0)
                        if (_autocap_armed and _centred_count >= CENTRE_HOLD_FRAMES
                                and time.time() - _last_autocap > AUTOCAP_COOLDOWN):
                            # Hand over to the flash-lit approach.  Take the all-off
                            # dark reference FIRST (every red measurement subtracts
                            # it), then run the ambient/flash cycle from here on.
                            _centred_count = 0
                            _pre_sweep_ambient = ambient_led_on
                            ambient_off()
                            flash_off(); led_on = False
                            _drain_frames(picam2)
                            _sweep_dark = _grab_disp()
                            flash_on(); led_on = True
                            apply_camera_settings(picam2, FLASH_GAIN)
                            _drain_frames(picam2)
                            side = ((_cover.side or COVER_SIDE)
                                    if USE_LIVE_COVER else COVER_SIDE)
                            _sweep = ApproachState(side, _live_center, _live_radius)
                            _blink.reset()   # new eye/lighting: relearn "open"
                            _stage = guidance.STAGE_APPROACH
                            print(f"Stage -> approach: move the scope so the pupil "
                                  f"nears the {guidance.cover_edge_word(side)} edge "
                                  f"- red {_PUPIL_SIZE.status()}")
                    # In centring the two coincide — the pupil belongs on the same
                    # point the patient is looking at — but both markers are drawn
                    # anyway so the vocabulary is identical in every stage.
                    disp = guidance.annotate(
                        disp, lg, tgt, _live_center,
                        deadzone_box=_live_guide.deadzone_box(disp.shape),
                        extra=extra, cover_mask=cover_disp, fixation=True,
                        hold_point=tgt)

                # ── approach / target: both run the ambient/flash cycle ──
                elif _stage in (guidance.STAGE_APPROACH, guidance.STAGE_TARGET):
                    # The LEDs are driven at the bottom of the loop from
                    # _sweep.lighting(); here we just consume the frames it produces.
                    if _stage == guidance.STAGE_TARGET:
                        # Move the point EVERY frame so it glides between stops.
                        # Driving it from the measurement cadence made it lurch,
                        # which a gaze cannot follow.
                        _sweep.tick()
                        tgt = _sweep.target()      # the stepped fixation point

                        # ONE capture per stop, at FULL resolution, once the gaze
                        # has arrived and the pupil has re-dilated in the dark.
                        # Same shape as the approach's cycle and as the final
                        # capture: dark reference, then a single flash frame.
                        if _sweep.ready_to_capture():
                            _sweep.to_dark_ref()
                        elif _sweep.phase == _sweep.PHASE_DARK:
                            _drain_frames(picam2)
                            _sweep_dark_full = _grab_full()
                            _sweep.to_flash()
                        elif (_sweep.phase == _sweep.PHASE_FLASH
                                and _sweep.ready()):
                            _drain_frames(picam2)
                            flash_full = _grab_full()
                            # The stage tracks the pupil in DISPLAY coords, so the
                            # ROI has to be taken up to the full-resolution frame.
                            fs = flash_full.shape[1] / float(disp.shape[1])
                            ctr = (_sweep.center[0] * fs, _sweep.center[1] * fs)
                            rad = (_sweep.radius0 * fs) if _sweep.radius0 else None
                            cov = (cv2.resize(cover_disp,
                                              (flash_full.shape[1],
                                               flash_full.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                                   if cover_disp is not None else None)
                            _cv_t0 = time.time()
                            meas = measure_red_fraction(
                                flash_full, _sweep_dark_full, ctr, rad, cov,
                                min_px=REDEYE_MIN_PX * _sweep.px_scale)
                            _sweep.note_cv(time.time() - _cv_t0)
                            # Full-res mask; the overlay and the freeze are display
                            # sized, so keep a scaled copy for drawing only.
                            _redeye_mask = (_to_disp(meas["mask"])
                                            if meas["mask"] is not None else None)
                            _show_frame = _to_disp(flash_full)
                            if _sweep.worth_saving(meas):
                                save_session_frame(_session, meas["extract"],
                                                   meas["mask"])
                            # Put the stage's pupil back into display coords.
                            if meas["center"] is not None:
                                meas["center"] = (meas["center"][0] / fs,
                                                  meas["center"][1] / fs)
                            if meas["radius"]:
                                meas["radius"] = meas["radius"] / fs
                            meas["red_px"] = int(meas["red_px"] / _sweep.px_scale)
                            _sweep.submit(meas)
                            _sweep.to_show()
                        elif (_sweep.phase == _sweep.PHASE_SHOW
                                and _sweep.ready()):
                            _show_frame = None
                            _sweep.capture_done()
                    elif _sweep.ready():
                        # NOTE: deliberately NOT gated on `fresh`.  That counter
                        # throttles the live pupil OVERLAY, which does not run in a
                        # lit stage at all — so all it did here was stall each phase
                        # until the next 5th frame, stretching the cycle by ~0.3s a
                        # phase and leaving most of the flash period unscanned.  The
                        # per-phase deadlines and the flush below do the pacing.
                        if _sweep.phase == _sweep.PHASE_AMBIENT:
                            # Grab the companion AFTER a flush: a stale flash frame
                            # taken for an ambient one vetoes a real fundus.  Kept
                            # in its own name rather than reassigning `clean`, which
                            # the rest of the iteration still refers to.
                            amb_frame = clean
                            if SWEEP_FLUSH:
                                _drain_frames(picam2)
                                amb_frame = _grab_disp()
                            _sweep_ambient = amb_frame      # the veto companion
                            _amb_gray = cv2.cvtColor(amb_frame, cv2.COLOR_BGR2GRAY)
                            # Blink check FIRST, on the ambient frame — the only
                            # one in the cycle where the eye is lit well enough to
                            # tell an open pupil from a closed lid.  A spoiled
                            # cycle is thrown away whole rather than measured.
                            # Detection runs first: whether it found the pupil is
                            # the blink detector's primary cue, and needs no
                            # photometric calibration.
                            detect_pupil(_amb_gray, live=True)
                            _amb_spoiled = _blink.update(
                                amb_frame, _sweep.center, _sweep.radius0,
                                pupil_found=_LAST_PUPIL is not None,
                                pupil_conf=_LAST_CONF)
                            # Tell the stage what the ambient frame showed FIRST and
                            # unconditionally, spoiled or not.  A pupil driven out of
                            # the frame reads as a blink, so a check that skipped
                            # spoiled cycles could never see the overshoot it exists
                            # to catch — the stage decides what to do with it.
                            _sweep.check_ambient_pupil(
                                (_LAST_PUPIL[0], _LAST_PUPIL[1])
                                if _LAST_PUPIL is not None else None,
                                _LAST_CONF if _LAST_PUPIL is not None else 0.0,
                                amb_frame.shape, spoiled=_amb_spoiled)
                            if _amb_spoiled:
                                # Spoiled — and do NOT adopt this frame's pupil.
                                # A lid gives a confident-looking fit on a crease,
                                # and taking it would drag the ROI off the pupil
                                # for every cycle that follows.
                                _sweep.spoiled = True
                                _overlay_cache = None
                            elif _LAST_PUPIL is not None:
                                # The ambient blip is the one properly-lit frame in
                                # the cycle, so it is also where the red-eye bar
                                # gets its pupil size from.
                                note_pupil_size(_LAST_PUPIL[2], _LAST_CONF)
                                if _LAST_CONF >= SWEEP_TRACK_CONF:
                                    _sweep.radius0 = _LAST_PUPIL[2]
                                    # Keep the loop's own tracked pupil current
                                    # too, so the camera hint, a manual capture
                                    # and a drop back to centring all start from
                                    # where the eye actually is now.
                                    _live_center = (_LAST_PUPIL[0], _LAST_PUPIL[1])
                                    _live_radius = _LAST_PUPIL[2]
                            _sweep.to_dark()
                        elif _sweep.phase == _sweep.PHASE_DARK:
                            # All LEDs off: the pupil has re-opened after the
                            # ambient blip, and this frame is a fresh subtraction
                            # reference registered with the eye where it is NOW.
                            # Flushed for the same reason as the others — a stale
                            # flash frame here would subtract away the reflex.
                            if SWEEP_FLUSH:
                                _drain_frames(picam2)
                                _sweep_dark = _grab_disp()
                            else:
                                _sweep_dark = clean
                            _sweep.to_flash()
                        elif _sweep.phase == _sweep.PHASE_FLASH and _sweep.spoiled:
                            # Discarded: not measured, not submitted, and NOT
                            # counted as a lost reflex — a blink is not evidence
                            # that the alignment is wrong.  Just run the cycle
                            # again once the eye is open.
                            _sweep.spoiled = False
                            _sweep.spoiled_n += 1
                            _sweep.to_ambient()
                        elif _sweep.phase == _sweep.PHASE_FLASH:
                            # THE measured frame: the first one out of the pipeline
                            # after the flush, i.e. the earliest genuinely flash-lit
                            # exposure there is.  The reflex is at its largest here
                            # and shrinks from this moment on as the pupil
                            # constricts, so a later frame in the same flash period
                            # would under-report an alignment that was in fact fine.
                            flash_frame = clean
                            if SWEEP_FLUSH:
                                _drain_frames(picam2)
                                flash_frame = _grab_disp()
                            # The flash stays lit across this whole call — the LEDs
                            # are only driven at the bottom of the loop, and the
                            # phase does not advance until submit() below returns.
                            # So the detection can never be cut short by the ambient
                            # coming back on, however long it takes.
                            _cv_t0 = time.time()
                            meas = measure_red_fraction(
                                flash_frame, _sweep_dark, _sweep.center,
                                _sweep.radius0,
                                cover_disp, ambient_bgr=_sweep_ambient)
                            _sweep.note_cv(time.time() - _cv_t0)
                            _redeye_mask = meas["mask"]   # reuse for the highlight
                            # Bank confident reflexes as they happen: the gaze is
                            # being walked around, so successive measurements see
                            # different parts of the fundus — mosaic material the
                            # single end-of-search capture cannot provide.
                            if _sweep.worth_saving(meas):
                                save_session_frame(_session, meas["extract"],
                                                   meas["mask"])
                            _show_frame = flash_frame   # held up during PHASE_SHOW
                            _sweep.submit(meas)
                        elif _sweep.phase == _sweep.PHASE_SHOW:
                            # The operator has had REDEYE_SHOW_S looking at what was
                            # selected; start the next cycle.
                            _show_frame = None
                            _sweep.to_ambient()
                        else:                               # dwell elapsed
                            _sweep.to_ambient()

                    if _sweep.abort:
                        # Restore ambient light and go back to centring rather than
                        # carrying on against a signal we no longer trust.
                        print(f"{_stage} aborted: {_sweep.abort} -> back to centring")
                        flash_off(); led_on = False
                        ambient_on(); ambient_led_on = True; _sweep_amb_on = False
                        apply_camera_settings(picam2, FLASH_GAIN)
                        _sweep = None
                        _sweep_ambient = None
                        _show_frame = None
                        _redeye_mask = None
                        _live_guide.reset()
                        _centred_count = 0
                        _stage = guidance.STAGE_CENTRE
                    elif _sweep.done and _stage == guidance.STAGE_APPROACH:
                        # Enough red to work with — hand over to the target search,
                        # which now steers the eye by moving the fixation point.
                        # px_scale: the scan measures on full-resolution frames, so
                        # every pixel-count threshold it applies has to be taken up
                        # by the AREA ratio against the display frames the other
                        # stages count in.
                        _full_w = CAMERA_HEIGHT if LIVE_ROTATION % 2 == 1 \
                            else CAMERA_WIDTH
                        _pxs = (float(_full_w) / float(disp.shape[1])) ** 2
                        _sweep = GazeScan(tgt, _sweep.center, _sweep.radius0,
                                          disp.shape, _sweep.cover_side,
                                          px_scale=_pxs)
                        _stage = guidance.STAGE_TARGET
                        print(f"Stage -> gaze scan: one full-resolution flash per "
                              f"gaze stop, {SCAN_DILATE_S:.1f}s dark between them "
                              f"for the pupil to re-open (px x{_pxs:.0f})")
                    elif _sweep.done:
                        _stage = guidance.STAGE_CAPTURE
                    else:
                        # Always show the blink numbers: the photometric cue is
                        # only synthetically validated, so the live readout is
                        # what it gets calibrated from.
                        sub = _sweep.status() + "  " + _blink.status()
                        if _sweep.spoiled_n:
                            sub += f" skipped={_sweep.spoiled_n}"
                        hold = None
                        # The red count belongs next to the fixation point with the
                        # instructions, not only in the corner status line: that is
                        # where the operator is already looking, and it is the one
                        # number their scope movement is actually chasing.
                        if _stage == guidance.STAGE_APPROACH:
                            sub = _sweep.hint() + "  |  " + sub
                            cam = _sweep.camera_hint(disp.shape, tgt)
                            vec = _sweep.camera_vector(disp.shape, tgt)
                            need = redeye_min_px()
                            read = f"red {int(_sweep.red_px)} / {need} px"
                            read_ok = _sweep.red_px >= need
                            # Here the two markers genuinely pull apart: the gaze
                            # stays on the fixation point while the pupil has to
                            # travel to the cover edge.
                            hold = guidance.approach_destination(
                                disp.shape, _sweep.cover_side, tgt[0], cover_disp)
                        else:
                            cam = _sweep.camera_hint(disp.shape)
                            vec = None          # the gaze moves here, not the scope
                            hold = _sweep.hold  # ...so show where to keep the pupil
                            # No threshold to meet here — the scan is maximising —
                            # so show what it has against the best it has seen.
                            need = redeye_min_px()
                            read = (f"red {int(_sweep.last_px)} px "
                                    f"(best {int(_sweep.best_px)})")
                            read_ok = _sweep.last_px >= need
                        sg = guidance.Guidance(state=_stage,
                                               instruction=guidance.INSTRUCTION_LOOK,
                                               hint=cam, hint_vector=vec,
                                               readout=read, readout_ok=read_ok)
                        disp = guidance.annotate(disp, sg, tgt, _sweep.center,
                                                 extra=sub, cover_mask=cover_disp,
                                                 fixation=True, hold_point=hold)

                # ── capture: dilate in the dark, then the dark+flash pair ──
                if _stage == guidance.STAGE_CAPTURE:
                    _autocap_armed = False          # one take; 'k' re-authorises
                    _last_autocap = time.time()
                    frac = ((_sweep.best_center[0] / disp.shape[1],
                             _sweep.best_center[1] / disp.shape[0])
                            if _sweep.best_center is not None else None)
                    rad_frac = (_sweep.best_radius / disp.shape[1]
                                if _sweep.best_radius else None)
                    print(f"Gaze scan best red={_sweep.best_px}px from "
                          f"{_sweep.samples_n} sample(s) -> capturing "
                          f"(press 'k' to authorise the next)")
                    cv2.imshow(WINDOW_NAME, disp); cv2.waitKey(1)
                    capture_flash_pair(picam2, frac, rad_frac, session=_session)
                    led_on = False                  # capture_flash_pair leaves it off
                    _sweep_amb_on = False           # sweep no longer owns the ambient
                    if _pre_sweep_ambient:
                        ambient_on(); ambient_led_on = True
                    apply_camera_settings(picam2, FLASH_GAIN)
                    _sweep = None
                    _sweep_dark = None
                    _sweep_ambient = None
                    _redeye_mask = None
                    _live_guide.reset()
                    _cover.reset()
                    _centred_count = 0
                    _stage = START_STAGE

            # ── live red-eye highlight ──
            # The sweep already computed a mask above; outside it, compute one on
            # every _LIVE_DETECT_SKIP-th frame whenever the flash is lit and a dark
            # reference exists.  This runs off its OWN counter, not the detection
            # loop's `fresh`, so the highlight still works with pupil detection and
            # the staged session both off (just holding SPACE).
            # With the picture hidden the tint is skipped but the SELECTION is still
            # computed below — the mask feeds the sweep's numbers and the banner.
            if redeye_hl_on and _stage in _LIT_STAGES:
                # The sweep owns the mask; keep drawing it across the ambient blip
                # so the highlight doesn't strobe with the lighting cycle.
                if not image_hidden:
                    disp = draw_redeye_highlight(disp.copy(), _redeye_mask)
            elif redeye_hl_on and led_on and _live_dark is not None:
                if _hl_counter % _LIVE_DETECT_SKIP == 0:
                    # Prefer a tracked pupil; fall back to the frame centre, which
                    # is where the pupil is being driven anyway, so the ROI still
                    # lands on the eye when nothing is tracking it.
                    hl_center = _live_center
                    hl_radius = _live_radius
                    if hl_center is None:
                        hl_center = (disp.shape[1] // 2, disp.shape[0] // 2)
                        hl_radius = None
                    hl = redeye_extract(clean, _live_dark, None, hl_center,
                                        hl_radius,
                                        _cover.mask if USE_LIVE_COVER else None)
                    _redeye_mask = hl.mask if hl.valid else None
                if not image_hidden:
                    disp = draw_redeye_highlight(disp.copy(), _redeye_mask)

            if image_hidden:
                draw_hidden_banner(disp, _redeye_mask)

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

            # g starts/stops the staged alignment session and ARMS the first
            # auto-capture.  Stopping mid-sweep restores the ambient lighting.
            elif key == ord('g'):
                if _stage is None:
                    _stage = START_STAGE
                    _cover.reset()
                    _live_guide.reset()
                    _overlay_cache = None
                    _centred_count = 0
                    _sweep = None
                    _sweep_dark = None
                    _sweep_ambient = None
                    _sweep_amb_on = False
                    _autocap_armed = True
                    if not ambient_led_on:
                        ambient_on(); ambient_led_on = True
                    print(f"Staged guidance ON (auto-capture armed) -> {START_STAGE}"
                          f"  [target: {GUIDANCE_TARGET_MODE}]")
                else:
                    was_sweeping = (_stage in _LIT_STAGES)
                    _stage = None
                    _sweep = None
                    _sweep_dark = None
                    _sweep_ambient = None
                    _sweep_amb_on = False
                    _autocap_armed = False
                    _live_guide.reset()
                    _cover.reset()
                    if was_sweeping:
                        flash_off(); led_on = False
                        ambient_led_on = _pre_sweep_ambient
                        if ambient_led_on:
                            ambient_on()
                        else:
                            ambient_off()
                    print("Staged guidance OFF")

            # k re-authorises the next auto-capture (after one has fired)
            elif key == ord('k'):
                if _stage is not None:
                    _autocap_armed = True
                    _centred_count = 0
                    print("Auto-capture authorised - align for the next take.")
                else:
                    print("Turn on staged guidance ('g') first.")

            # t toggles PC image transfer (HTTP-POST captures to receiver.py)
            elif key == ord('t'):
                TRANSFER_ENABLED = not TRANSFER_ENABLED
                print(f"PC image transfer: "
                      f"{'ON' if TRANSFER_ENABLED else 'OFF'}")

            # f flips the cover side: top<->bottom (180°, mask follows)
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

            # SHIFT toggles the live red-eye highlight.  X11 keysyms, same family
            # as the arrow keys below (XK_Shift_L/R = 0xFFE1/0xFFE2); some cv2
            # backends never report a bare modifier, so 'h' does the same thing.
            elif key in (65505, 65506) or key == ord('h'):
                redeye_hl_on = not redeye_hl_on
                _redeye_mask = None
                print(f"Live red-eye highlight: "
                      f"{'ON' if redeye_hl_on else 'OFF'}")

            # i hides/shows the camera picture on the live feed.  Guidance only:
            # useful when the operator should read the instructions rather than
            # study the eye, and when the raw feed must not be on screen.  Nothing
            # downstream changes — detection, the sweep and the session saving all
            # run on `clean`, and the captures shown after the scan are unaffected.
            elif key == ord('i'):
                image_hidden = not image_hidden
                print(f"Live camera image: "
                      f"{'HIDDEN (guidance only)' if image_hidden else 'SHOWN'}")

            # o PAUSES the live feed and mosaics this session's red-eye extracts.
            # Stitching is slow (feature detection on several frames), hence the
            # pause: the LEDs go off and nothing else runs until it is done.
            elif key == ord('o'):
                n = len(session_shots(_session))
                if n < SESSION_MIN_FOR_MOSAIC:
                    print(f"[MOSAIC] session {_session} has {n} capture(s); "
                          f"need {SESSION_MIN_FOR_MOSAIC}.")
                else:
                    print(f"--- live feed PAUSED: mosaicing {n} capture(s) ---")
                    was_flash, was_amb = led_on, ambient_led_on
                    flash_off(); ambient_off(); led_on = False
                    cv2.putText(disp, "MOSAICING - paused", (10, disp.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow(WINDOW_NAME, disp); cv2.waitKey(1)
                    build_session_mosaic(_session)
                    show_session_shots(_session)
                    print("--- live feed RESUMED (any window stays open) ---")
                    if was_amb:
                        ambient_on()
                    if was_flash:
                        flash_on(); led_on = True
                    apply_camera_settings(picam2, FLASH_GAIN)
                    _live_dark = None       # lighting was disturbed; re-reference

            # x shows the SIFT keypoints of every capture in this session — the
            # diagnostic for "why will this session not stitch?".  Cheap (detection
            # only, no matching), so it does not pause the feed the way 'o' does.
            elif key == ord('x'):
                was_flash, was_amb = led_on, ambient_led_on
                flash_off(); ambient_off(); led_on = False
                show_session_keypoints(_session)
                if was_amb:
                    ambient_on()
                if was_flash:
                    flash_on(); led_on = True
                _live_dark = None          # lighting was disturbed; re-reference

            # v steps through this session's captures one at a time, full size.
            # Blocks the feed while open; LEDs off so nothing is left lit.
            elif key == ord('v'):
                was_flash, was_amb = led_on, ambient_led_on
                flash_off(); ambient_off(); led_on = False
                browse_session_shots(_session)
                if was_amb:
                    ambient_on()
                if was_flash:
                    flash_on(); led_on = True
                _live_dark = None          # lighting was disturbed; re-reference

            # Arrow keys: left=CCW step, right=CW step
            elif key == 65361:  # left arrow
                LIVE_ROTATION = (LIVE_ROTATION - 1) % 4
                print(f"Rotation: {LIVE_ROTATION * 90}°")
            elif key == 65363:  # right arrow
                LIVE_ROTATION = (LIVE_ROTATION + 1) % 4
                print(f"Rotation: {LIVE_ROTATION * 90}°")

            # ── LED drive ──
            # During the sweep the phase machine owns both LEDs: flash for settle
            # and measurement, a brief ambient blip to grab the veto companion.
            # It also counts as a reason for the flash to be lit, so a stray SPACE
            # press-and-release cannot switch it off underneath the sweep.
            sweeping = (_stage in _LIT_STAGES and _sweep is not None)
            lit = _sweep.lighting() if sweeping else None
            want_ambient = (lit == "ambient")
            # "off" is the dark pause between the ambient blip and the measured
            # flash frame — neither LED runs then.
            should_be_on = lock_on or space_held or (lit == "flash")
            if should_be_on and not led_on:
                flash_on()
                led_on = True
            elif not should_be_on and led_on:
                flash_off()
                led_on = False

            if sweeping:
                if want_ambient != _sweep_amb_on:
                    ambient_on() if want_ambient else ambient_off()
                    _sweep_amb_on = want_ambient
            elif _sweep_amb_on:                 # left the sweep — hand the ambient
                _sweep_amb_on = False           # LEDs back to the operator's toggle
                ambient_on() if ambient_led_on else ambient_off()

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

        # Cover side: "top" | "bottom"; switching flips the feed 180°
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

    # Which side is the LED cover on? (bottom = the default rig orientation.)
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
