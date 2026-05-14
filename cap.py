#!/usr/bin/env python3
"""
Retina Flash Photography
Raspberry Pi Zero + Camera Module V2

CONTROLS
────────
ENTER               Capture photo
s + ENTER           Streaming mode
q + ENTER           Quit

Streaming mode
──────────────
SPACE (hold)        LED on
r                   Toggle LED lock on/off
f                   Toggle focus metric overlay
←/→                 Rotate live feed
ENTER or e          Capture still
s                   Exit streaming

PARAMETERS
──────────
exposure 32000
flash_gain 1.6
live_gain 1.0

pre_delay 0.05
post_delay 0.05

flash1_duration 0.075
flash_gap 3
flash2_duration 0.075

brightness 2
contrast 1

red_gain 2.2
blue_gain 1.4
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

LED_PIN = 17

PHOTO_PATH = "/tmp/retina_preview.jpg"

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

FLASH1_DURATION = 0.075
FLASH_GAP = 3.0
FLASH2_DURATION = 0.075

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

    GPIO.setup(LED_PIN, GPIO.OUT)

    GPIO.output(LED_PIN, GPIO.LOW)


def cleanup_gpio():

    if not GPIO_AVAILABLE:
        return

    GPIO.output(LED_PIN, GPIO.LOW)

    GPIO.cleanup()


def flash_on():

    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH)


def flash_off():

    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.LOW)


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

    print(f"flash1_duration    = {FLASH1_DURATION}")
    print(f"flash_gap          = {FLASH_GAP}")
    print(f"flash2_duration    = {FLASH2_DURATION}")

    print()

    print(f"brightness         = {BRIGHTNESS}")
    print(f"contrast           = {CONTRAST}")

    print("────────────────────────\n")


def process_image(array):

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

    return img


def capture_image(picam2):
    # ─────────────────────────────────────────────────────────────────────────
    # ROI ZOOM / RESOLUTION ENHANCEMENT — DESIGN NOTES
    # ─────────────────────────────────────────────────────────────────────────
    # Goal: increase pixel density over the pupil/fundus disc, discard the
    # surrounding iris/eyelid area that carries no diagnostic information.
    #
    # WHY THIS IS FULLY ACHIEVABLE on the Pi Camera Module V2 (IMX219):
    #
    #   The sensor's native resolution is 3280 × 2464 px.  We currently
    #   capture at 1280 × 960, which means we are already throwing away
    #   (3280*2464) / (1280*960) ≈ 6.6× of the available sensor pixels by
    #   down-sampling the whole field of view uniformly.
    #
    #   picamera2 exposes the `ScalerCrop` control, which tells the ISP to
    #   read out only a rectangular sub-region of the full sensor area and
    #   then scale *that* region up to the requested output size.  This is a
    #   true optical crop — equivalent to moving the camera closer — and
    #   gives genuine sub-pixel resolution improvement over the pupil, not
    #   just a post-capture digital zoom.
    #
    # HOW IT WOULD WORK:
    #
    #   1.  From the live feed the pupil centre (cx, cy) and radius (r) are
    #       already known from _detect_reflex().  These are in display-frame
    #       coordinates (e.g. 480 × 640 after resize + rotation).
    #
    #   2.  Map those coordinates back through the rotation and the
    #       resize to get sensor-frame coordinates:
    #
    #         sensor_x = cx_live * (CAMERA_WIDTH  / DISPLAY_W)
    #         sensor_y = cy_live * (CAMERA_HEIGHT / DISPLAY_H)
    #         margin   = r_live  * (CAMERA_WIDTH  / DISPLAY_W) * 1.5
    #         crop_x   = max(0, int(sensor_x - margin))
    #         crop_y   = max(0, int(sensor_y - margin))
    #         crop_w   = min(CAMERA_WIDTH  - crop_x, int(margin * 2))
    #         crop_h   = min(CAMERA_HEIGHT - crop_y, int(margin * 2))
    #
    #   3.  picamera2 ScalerCrop takes the crop in *native full-sensor*
    #       coordinates.  Because we configure the camera at 1280 × 960
    #       (which itself corresponds to a centred crop of the 3280 × 2464
    #       sensor), we also need to apply the sensor-to-output scale factor
    #       and the ISP's centred-crop offset:
    #
    #         native_scale_x = 3280 / CAMERA_WIDTH   # ≈ 2.5625
    #         native_scale_y = 2464 / CAMERA_HEIGHT  # ≈ 2.5667
    #         native_offset_x = (3280 - CAMERA_WIDTH  * native_scale_x) / 2  # = 0
    #         native_offset_y = (2464 - CAMERA_HEIGHT * native_scale_y) / 2  # ≈ 0
    #         scaler_crop = (
    #             int(crop_x * native_scale_x + native_offset_x),
    #             int(crop_y * native_scale_y + native_offset_y),
    #             int(crop_w * native_scale_x),
    #             int(crop_h * native_scale_y),
    #         )
    #         picam2.set_controls({"ScalerCrop": scaler_crop})
    #
    #   4.  Capture as normal. The 1280 × 960 output now contains only the
    #       pupil region, so every pixel is worth 6.6× more information
    #       than in the full-frame capture. After capture, restore the
    #       ScalerCrop to the full sensor area:
    #         picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
    #
    # PRACTICAL CAVEATS:
    #
    #   a)  ScalerCrop must be set *before* the flash so there is time for
    #       the ISP to apply the new crop (a 1–2 frame settling delay, ~50 ms
    #       at 20 fps, is sufficient — the existing FLASH_PRE_DELAY covers it).
    #
    #   b)  The pupil centre from the live feed is measured at a different
    #       instant than the flash capture (patient may move slightly).
    #       Using a generous margin (1.5–2× the pupil radius) compensates.
    #
    #   c)  The rotation applied to the live display (LIVE_ROTATION) must be
    #       inverted when mapping back to raw sensor coordinates; the sensor
    #       itself is never rotated — only the display pipeline is.
    #
    #   d)  If _detect_reflex() returns None (no bright spot found yet),
    #       fall back to the full-frame capture as today.
    #
    # This approach is therefore entirely feasible and would be straightforward
    # to implement once the coordinate-mapping logic above is wired up and the
    # last detected pupil position is stored in a module-level variable that
    # capture_image() can read.
    # ─────────────────────────────────────────────────────────────────────────

    print("Flash + capture...")

    # Apply flash settings
    apply_camera_settings(
        picam2,
        FLASH_GAIN
    )

    time.sleep(0.02)

    # ───────── FIRST FLASH ─────────

    flash_on()

    time.sleep(FLASH1_DURATION)

    flash_off()

    # Gap between flashes
    time.sleep(FLASH_GAP)

    # ───────── SECOND FLASH ─────────

    flash_on()

    time.sleep(FLASH_PRE_DELAY)

    # Capture while flash ON
    array = picam2.capture_array()

    time.sleep(FLASH2_DURATION)

    flash_off()

    time.sleep(FLASH_POST_DELAY)

    # Restore live settings
    apply_camera_settings(
        picam2,
        LIVE_GAIN
    )

    # Process image
    img = process_image(array)

    # Update preview image
    img.save(PHOTO_PATH)

    print("Captured.")


# ── Volcano / concentric-ring detection ─────────────────────────────────────
_V_LOCAL_KERNEL   = 50
_V_MIN_CONTRAST   = 0.10
_V_MAX_RIM_R      = 220
_V_N_ANGLES       = 180
_V_RING_SEP_RATIO = 1.3
_V_CENTRE_TOL     = 55
_V_RING_SIZE_TOL  = 0.50
_V_SPECULAR_PCT   = 99.0
_V_INPAINT_R      = 7
# Live-feed performance: smaller search radius + only recompute every N frames
_V_LIVE_MAX_RIM_R = 100
_LIVE_OVERLAY_SKIP = 5

_V_PIT_COL   = (0,   255, 255)   # yellow  — pit edge
_V_RIM_COL   = (255,   0, 255)   # magenta — rim
_V_PEAK_COL  = (0,   165, 255)   # orange  — peak-volcano


def _v_radial_profile(smooth, cx, cy, max_r):
    angles = np.linspace(0, 2 * np.pi, _V_N_ANGLES, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    prof = np.zeros(max_r, dtype=np.float32)
    for r in range(max_r):
        xs = np.clip((cx + r * cos_a).astype(int), 0, smooth.shape[1] - 1)
        ys = np.clip((cy + r * sin_a).astype(int), 0, smooth.shape[0] - 1)
        prof[r] = smooth[ys, xs].mean()
    return prof


def _v_find_volcanos(channel_u8, img_h, img_w, max_rim_r=None):
    if max_rim_r is None:
        max_rim_r = _V_MAX_RIM_R
    smooth   = cv2.GaussianBlur(channel_u8.astype(np.float32), (31, 31), 0)
    ch_range = float(smooth.max() - smooth.min())
    if ch_range == 0:
        return []
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (_V_LOCAL_KERNEL, _V_LOCAL_KERNEL))
    eroded       = cv2.erode(smooth,  kernel)
    dilated      = cv2.dilate(smooth, kernel)
    contrast_map = dilated - eroded
    local_min    = ((smooth - eroded)  < 1.5) & (contrast_map > ch_range * _V_MIN_CONTRAST)
    local_max    = ((dilated - smooth) < 1.5) & (contrast_map > ch_range * _V_MIN_CONTRAST)

    results = []
    for mask, kind in ((local_min.astype(np.uint8), 'pit'),
                       (local_max.astype(np.uint8), 'peak')):
        n_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask)
        for lbl in range(1, n_labels):
            cx, cy  = int(centroids[lbl][0]), int(centroids[lbl][1])
            safe_r  = min(cx, cy, img_w - cx - 1, img_h - cy - 1, max_rim_r)
            if safe_r < 20:
                continue
            prof       = _v_radial_profile(smooth, cx, cy, safe_r)
            center_val = prof[0]
            search_end = min(safe_r - 1, max_rim_r)
            if kind == 'pit':
                rim_idx  = int(np.argmax(prof[5:search_end])) + 5
                contrast = (prof[rim_idx] - center_val) / ch_range
                if contrast < _V_MIN_CONTRAST:
                    continue
                thresh = center_val + 0.2 * (prof[rim_idx] - center_val)
                above  = np.where(prof[1:rim_idx] > thresh)[0]
                pit_r  = int(above[0]) + 1 if len(above) else 5
                results.append(dict(cx=cx, cy=cy, kind='pit',
                                    pit_r=pit_r, rim_r=rim_idx, contrast=contrast))
            else:
                moat_idx = int(np.argmin(prof[5:search_end])) + 5
                drop     = (center_val - prof[moat_idx]) / ch_range
                if drop < _V_MIN_CONTRAST or moat_idx + 5 >= search_end:
                    continue
                outer_idx = int(np.argmax(prof[moat_idx:search_end])) + moat_idx
                rise      = (prof[outer_idx] - prof[moat_idx]) / ch_range
                if rise < _V_MIN_CONTRAST * 0.5:
                    continue
                results.append(dict(cx=cx, cy=cy, kind='peak',
                                    pit_r=moat_idx, rim_r=outer_idx, contrast=drop))

    results.sort(key=lambda v: -v['contrast'])
    unique = []
    for v in results:
        if not any(np.hypot(v['cx'] - u['cx'], v['cy'] - u['cy']) < 30 for u in unique):
            unique.append(v)
    return unique


def _v_has_two_rings(v):
    return v['pit_r'] > 0 and v['rim_r'] > v['pit_r'] * _V_RING_SEP_RATIO


def _v_draw(canvas, volcanos):
    for v in volcanos:
        cx, cy  = v['cx'], v['cy']
        pit_col = _V_PIT_COL if v['kind'] == 'pit' else _V_PEAK_COL
        cv2.circle(canvas, (cx, cy), v['pit_r'], pit_col, 2)
        cv2.circle(canvas, (cx, cy), v['rim_r'], _V_RIM_COL, 2)
        cv2.circle(canvas, (cx, cy), 4, pit_col, -1)


def _volcano_overlay(frame, live=False):
    """Detect peak-type concentric rings (orange+magenta) in blue/red channels,
    match them, and fall back to green peaks if no match.  Returns annotated BGR
    frame.  Replaces _pupil_overlay / _focus_overlay."""
    mr = _V_LIVE_MAX_RIM_R if live else _V_MAX_RIM_R
    img_h, img_w = frame.shape[:2]

    # Inpaint specular (Purkinje) reflection
    spec_max  = np.max(frame, axis=2)
    thresh    = np.percentile(spec_max, _V_SPECULAR_PCT)
    spec_mask = (spec_max > thresh).astype(np.uint8)
    spec_mask = cv2.dilate(spec_mask,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img = cv2.inpaint(frame, spec_mask, _V_INPAINT_R, cv2.INPAINT_TELEA)

    canvas     = frame.copy()
    blue_vs    = _v_find_volcanos(img[:, :, 0], img_h, img_w, mr)
    red_vs     = _v_find_volcanos(img[:, :, 2], img_h, img_w, mr)
    blue_peaks = [v for v in blue_vs if _v_has_two_rings(v) and v['kind'] == 'peak']
    red_peaks  = [v for v in red_vs  if _v_has_two_rings(v) and v['kind'] == 'peak']

    matches = []
    for bv in blue_peaks:
        for rv in red_peaks:
            if np.hypot(bv['cx'] - rv['cx'], bv['cy'] - rv['cy']) > _V_CENTRE_TOL:
                continue
            for attr in ('pit_r', 'rim_r'):
                mean_r = (bv[attr] + rv[attr]) / 2
                if mean_r == 0 or abs(bv[attr] - rv[attr]) / mean_r > _V_RING_SIZE_TOL:
                    break
            else:
                matches.append((bv, rv))

    if matches:
        for bv, rv in matches:
            cx  = (bv['cx']    + rv['cx'])    // 2
            cy  = (bv['cy']    + rv['cy'])    // 2
            p_r = (bv['pit_r'] + rv['pit_r']) // 2
            r_r = (bv['rim_r'] + rv['rim_r']) // 2
            cv2.circle(canvas, (bv['cx'], bv['cy']), bv['pit_r'], (255, 100,   0), 1)
            cv2.circle(canvas, (bv['cx'], bv['cy']), bv['rim_r'], (255, 100,   0), 1)
            cv2.circle(canvas, (rv['cx'], rv['cy']), rv['pit_r'], (  0,  80, 255), 1)
            cv2.circle(canvas, (rv['cx'], rv['cy']), rv['rim_r'], (  0,  80, 255), 1)
            cv2.circle(canvas, (cx, cy), p_r, (255, 255, 255), 2)
            cv2.circle(canvas, (cx, cy), r_r, (255, 255, 255), 2)
            cv2.circle(canvas, (cx, cy), 5,   (255, 255, 255), -1)
            cv2.putText(canvas, f"match {p_r}/{r_r}px",
                        (cx - 40, cy - r_r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        _v_draw(canvas, blue_peaks)
        _v_draw(canvas, red_peaks)
        green_vs    = _v_find_volcanos(img[:, :, 1], img_h, img_w, mr)
        green_peaks = [v for v in green_vs
                       if _v_has_two_rings(v) and v['kind'] == 'peak']
        _v_draw(canvas, green_peaks)

    return canvas


def _detect_reflex(frame):
    """Locate the pupil centre from the corneal LED reflex (near-white bright
    spot) and estimate the pupil disc radius from the surrounding red/orange
    fundus glow.  Using the specular reflection is more reliable than colour
    alone because it is always at the pupil centre and is unaffected by the
    diffuse LED ring on surrounding skin.

    Returns (cx, cy, radius) in full-resolution coordinates, or None.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # ── 1. Corneal reflex: near-white, very bright ──────────────────────────
    # V > 190, S < 120 → excludes coloured regions while catching a slightly
    # pink-tinted LED reflection.
    bright = cv2.inRange(hsv,
                         np.array([0,   0, 190]),
                         np.array([179, 120, 255]))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k3)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k3)

    spot_cnts, _ = cv2.findContours(
        bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_spot = None
    best_sc   = 0.0
    for c in spot_cnts:
        a = cv2.contourArea(c)
        if a < 3:
            continue
        p = cv2.arcLength(c, True)
        if p < 1:
            continue
        circ = 4 * np.pi * a / (p * p)
        sc = a * circ          # favour large AND circular blobs
        if sc > best_sc:
            best_sc   = sc
            best_spot = c

    if best_spot is None:
        return None

    (sx, sy), spot_r = cv2.minEnclosingCircle(best_spot)

    # ── 2. Pupil disc from combined bright + fundus region ──────────────────
    # The fundus glow visible through the dilated pupil is red/orange.
    # Union with the bright spot then morphologically close to merge the two
    # regions into one connected pupil disc, bridging the dark ring between
    # the white specular spot and the surrounding reddish glow.
    fund_lo = cv2.inRange(hsv, np.array([0,   40, 30]),
                               np.array([25,  255, 255]))
    fund_hi = cv2.inRange(hsv, np.array([155, 40, 30]),
                               np.array([179, 255, 255]))
    fundus = cv2.bitwise_or(fund_lo, fund_hi)

    disc = cv2.bitwise_or(bright, fundus)
    k_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    disc = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, k_cl)

    disc_cnts, _ = cv2.findContours(
        disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pupil_r_h = None
    if disc_cnts:
        # Pick the disc contour whose enclosing-circle centre is closest to
        # the bright spot — rules out skin / optics artefacts far from centre.
        def _dist_sq(c):
            (x, y), _ = cv2.minEnclosingCircle(c)
            return (x - sx) ** 2 + (y - sy) ** 2
        c_disc = min(disc_cnts, key=_dist_sq)
        _, r_cand = cv2.minEnclosingCircle(c_disc)
        if r_cand > spot_r:   # disc must be larger than the spot itself
            pupil_r_h = r_cand

    if pupil_r_h is None:
        # Fallback: corneal reflex is typically ~5-10 % of pupil diameter
        pupil_r_h = max(spot_r * 5, 15)

    return (int(sx * 2), int(sy * 2), max(int(pupil_r_h * 2), 10))


def _pupil_overlay(frame):
    return _volcano_overlay(frame, live=True)


def _focus_overlay(frame):
    return _volcano_overlay(frame)


def streaming_mode(picam2):
    """Live video feed. Hold SPACE to turn LED on, r to lock/unlock.
    f toggles focus overlay on live feed. t toggles focus overlay on captures.
    Arrow keys rotate. Press ENTER or e to capture a still. Press s to exit."""

    global LIVE_ROTATION

    if not CV2_AVAILABLE:
        print("cv2 not available - cannot show live feed.")
        return

    print("\nStreaming mode ON")
    print("SPACE=LED | r=lock | f=pupil-overlay | ←/→=rotate | ENTER/e=capture | s=exit\n")

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
    lock_on = False  # Space+Shift toggle
    focus_overlay_on  = False
    _overlay_counter  = 0
    _overlay_cache    = None

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
            if focus_overlay_on:
                _overlay_counter += 1
                if (_overlay_cache is None
                        or _overlay_cache.shape != disp.shape
                        or _overlay_counter % _LIVE_OVERLAY_SKIP == 0):
                    _overlay_cache = _pupil_overlay(disp)
                disp = _overlay_cache
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
                # Always annotate the preview with pupil location + focus score
                _saved = cv2.imread(PHOTO_PATH)
                if _saved is not None:
                    cv2.imwrite(PHOTO_PATH, _focus_overlay(_saved))
                apply_camera_settings(picam2, FLASH_GAIN)

            # Track spacebar hold via auto-repeat
            elif key == 32:
                last_space_time = now
                space_held = True

            # r toggles LED lock
            elif key == ord('r'):
                lock_on = not lock_on

            # f toggles pupil overlay on live feed
            elif key == ord('f'):
                focus_overlay_on = not focus_overlay_on
                print(f"Pupil overlay (live): {'ON' if focus_overlay_on else 'OFF'}")

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

    global FLASH1_DURATION
    global FLASH_GAP
    global FLASH2_DURATION

    global BRIGHTNESS
    global CONTRAST

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

        elif param == "flash1_duration":
            FLASH1_DURATION = float(value)

        elif param == "flash_gap":
            FLASH_GAP = float(value)

        elif param == "flash2_duration":
            FLASH2_DURATION = float(value)

        # Image processing

        elif param == "brightness":
            BRIGHTNESS = float(value)

        elif param == "contrast":
            CONTRAST = float(value)

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

    print("ENTER = capture")
    print("s     = streaming mode (SPACE=LED, r=lock, f=focus, e/ENTER=capture)")
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