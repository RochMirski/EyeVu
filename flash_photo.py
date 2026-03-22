#!/usr/bin/env python3
"""
Retina Flash Photography - Raspberry Pi Zero

Modes:
  1) Manual  – Press SPACE to flash LED and capture.
  2) Auto    – Automatically captures when a pupil is detected near the
               centre of the frame (Orlosky detector), then pauses >=10 s.

The captured photo is displayed in a second window until the next capture or quit.
Photos are saved to ~/Desktop/retina_photos/ with timestamps.
Press 'q' to quit.
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available. LED flash will be simulated.")

# ──────────────── Configuration ────────────────
LED_PIN = 17              # BCM GPIO pin connected to the LED (via resistor)
FLASH_PRE_DELAY = 0.05   # Seconds to wait after LED on before capturing (lets LED stabilise)
FLASH_POST_DELAY = 0.02  # Seconds to keep LED on after capture
PHOTO_DIR = os.path.expanduser("~/Desktop/retina_photos")
CAMERA_INDEX = 0          # 0 for default camera; change if needed
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ──── Auto-capture (pupil centred) settings ────
AUTO_COOLDOWN_SECS = 10   # Minimum seconds between auto-captures
CENTRE_TOLERANCE_PX = 60  # Max distance from frame centre to count as "centred"
MIN_PUPIL_AREA = 1000     # Minimum contour area to accept as a pupil
MIN_SOLIDITY = 0.80       # Reject contours less solid than this (semicircles ~0.5-0.7)
MIN_ELLIPSE_FILL = 0.60   # Contour must fill at least 60% of its fitted ellipse
EDGE_MARGIN_PX = 8        # Reject contours whose bbox is within this many px of the frame edge
# ────────────────────────────────────────────────

LIVE_WINDOW = "Live Feed"
PHOTO_WINDOW = "Captured Photo"
SETTINGS_WINDOW = "Camera Settings"

# ──── Camera property defaults ────
# Values are 0–100 mapped to the camera's own range.
# Gain, brightness, exposure, contrast.
DEFAULT_LIVE_GAIN       = 100
DEFAULT_LIVE_BRIGHTNESS = 50
DEFAULT_LIVE_EXPOSURE   = 50
DEFAULT_LIVE_CONTRAST   = 50

DEFAULT_FLASH_GAIN       = 50
DEFAULT_FLASH_BRIGHTNESS = 50
DEFAULT_FLASH_EXPOSURE   = 50
DEFAULT_FLASH_CONTRAST   = 50


# ═══════════════ Orlosky Pupil Detector (inlined) ═══════════════

def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height
    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]
    return cv2.resize(cropped_img, (width, height))


def apply_binary_threshold(image, darkest_val, added_threshold):
    threshold = darkest_val + added_threshold
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded


def get_darkest_area(image):
    ignore = 20
    skip = 20
    area = 20
    internal_skip = 10
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest = None
    for y in range(ignore, gray.shape[0] - ignore, skip):
        for x in range(ignore, gray.shape[1] - ignore, skip):
            s = 0
            n = 0
            for dy in range(0, area, internal_skip):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, area, internal_skip):
                    if x + dx >= gray.shape[1]:
                        break
                    s += gray[y + dy][x + dx]
                    n += 1
            if s < min_sum and n > 0:
                min_sum = s
                darkest = (x + area // 2, y + area // 2)
    return darkest


def mask_outside_square(image, center, size):
    x, y = center
    half = size // 2
    mask = np.zeros_like(image)
    tl_x = max(0, x - half)
    tl_y = max(0, y - half)
    br_y = min(image.shape[0], y + half)
    mask[tl_y:br_y, tl_x:tl_x + size] = 255
    return cv2.bitwise_and(image, mask)


def filter_contours_largest(contours, pixel_thresh, ratio_thresh, img_shape=None):
    """Return the largest contour passing area, ratio, solidity, and edge checks."""
    max_area = 0
    largest = None
    img_h, img_w = img_shape[:2] if img_shape is not None else (9999, 9999)
    for c in contours:
        a = cv2.contourArea(c)
        if a < pixel_thresh:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ratio = max(w / h, h / w)
        if ratio > ratio_thresh:
            continue
        # Solidity: contour area / convex hull area
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = a / hull_area
            if solidity < MIN_SOLIDITY:
                continue
        # Edge rejection: skip contours touching the frame border
        if (x < EDGE_MARGIN_PX or y < EDGE_MARGIN_PX or
                x + w > img_w - EDGE_MARGIN_PX or
                y + h > img_h - EDGE_MARGIN_PX):
            continue
        if a > max_area:
            max_area = a
            largest = c
    return [largest] if largest is not None else []


def detect_pupil(frame):
    """Run Orlosky-style pupil detection on a frame.

    Returns (centre_x, centre_y, ellipse) or None if no pupil found.
    The frame is annotated in-place with the ellipse overlay.
    """
    work = crop_to_aspect_ratio(frame.copy())
    darkest = get_darkest_area(work)
    if darkest is None:
        return None

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    darkest_val = gray[darkest[1], darkest[0]]
    thresh = apply_binary_threshold(gray, darkest_val, 15)
    thresh = mask_outside_square(thresh, darkest, 250)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced = filter_contours_largest(contours, MIN_PUPIL_AREA, 3, img_shape=dilated.shape)

    if len(reduced) == 0 or len(reduced[0]) <= 5:
        return None

    ellipse = cv2.fitEllipse(reduced[0])

    # Ellipse-fill check: contour area vs fitted ellipse area
    contour_area = cv2.contourArea(reduced[0])
    ew, eh = ellipse[1]  # ellipse axes
    ellipse_area = np.pi * (ew / 2.0) * (eh / 2.0)
    if ellipse_area > 0 and (contour_area / ellipse_area) < MIN_ELLIPSE_FILL:
        return None

    cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
    # Scale detected coords back to original frame size
    h_orig, w_orig = frame.shape[:2]
    sx = w_orig / 640.0
    sy = h_orig / 480.0
    cx_scaled = int(cx * sx)
    cy_scaled = int(cy * sy)
    # Draw on the original frame
    scaled_ellipse = ((cx_scaled, cy_scaled),
                      (int(ellipse[1][0] * sx), int(ellipse[1][1] * sy)),
                      ellipse[2])
    cv2.ellipse(frame, scaled_ellipse, (0, 255, 0), 2)
    cv2.circle(frame, (cx_scaled, cy_scaled), 4, (255, 255, 0), -1)
    return (cx_scaled, cy_scaled, scaled_ellipse)


# ═══════════════════════════════════════════════════════════════════


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


def _nothing(_):
    """No-op callback for trackbar changes."""
    pass


def create_settings_trackbars():
    """Create a settings window with trackbars for live and flash camera params."""
    cv2.namedWindow(SETTINGS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SETTINGS_WINDOW, 400, 380)

    cv2.createTrackbar("Live Gain",       SETTINGS_WINDOW, DEFAULT_LIVE_GAIN,       100, _nothing)
    cv2.createTrackbar("Live Brightness", SETTINGS_WINDOW, DEFAULT_LIVE_BRIGHTNESS, 100, _nothing)
    cv2.createTrackbar("Live Exposure",   SETTINGS_WINDOW, DEFAULT_LIVE_EXPOSURE,   100, _nothing)
    cv2.createTrackbar("Live Contrast",   SETTINGS_WINDOW, DEFAULT_LIVE_CONTRAST,   100, _nothing)

    cv2.createTrackbar("Flash Gain",       SETTINGS_WINDOW, DEFAULT_FLASH_GAIN,       100, _nothing)
    cv2.createTrackbar("Flash Brightness", SETTINGS_WINDOW, DEFAULT_FLASH_BRIGHTNESS, 100, _nothing)
    cv2.createTrackbar("Flash Exposure",   SETTINGS_WINDOW, DEFAULT_FLASH_EXPOSURE,   100, _nothing)
    cv2.createTrackbar("Flash Contrast",   SETTINGS_WINDOW, DEFAULT_FLASH_CONTRAST,   100, _nothing)


def read_trackbar_settings(prefix):
    """Read the four trackbar values for a given prefix ('Live' or 'Flash')."""
    return {
        "gain":       cv2.getTrackbarPos(f"{prefix} Gain",       SETTINGS_WINDOW),
        "brightness": cv2.getTrackbarPos(f"{prefix} Brightness", SETTINGS_WINDOW),
        "exposure":   cv2.getTrackbarPos(f"{prefix} Exposure",   SETTINGS_WINDOW),
        "contrast":   cv2.getTrackbarPos(f"{prefix} Contrast",   SETTINGS_WINDOW),
    }


def apply_camera_settings(cap, settings):
    """Apply a settings dict (values 0-100) to the camera."""
    # Map 0-100 slider to real OpenCV property ranges.
    # Exact ranges vary per camera; these are reasonable defaults.
    cap.set(cv2.CAP_PROP_GAIN,       settings["gain"] * 2.55)        # 0 – 255
    cap.set(cv2.CAP_PROP_BRIGHTNESS,  settings["brightness"] * 2.55)  # 0 – 255
    cap.set(cv2.CAP_PROP_EXPOSURE,   (settings["exposure"] - 50) * 0.26)  # approx -13 … +13
    cap.set(cv2.CAP_PROP_CONTRAST,    settings["contrast"] * 2.55)   # 0 – 255


def flash_and_capture(cap):
    """Apply flash settings, turn on LED, capture a fresh frame, restore live settings."""
    # Switch to flash camera settings
    flash_settings = read_trackbar_settings("Flash")
    apply_camera_settings(cap, flash_settings)
    time.sleep(0.02)  # let camera adjust

    # Turn on LED
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH)

    # Let LED reach full brightness
    time.sleep(FLASH_PRE_DELAY)

    # Discard one buffered frame so we read a frame actually lit by the LED
    cap.grab()
    ret, frame = cap.read()

    # Brief hold then turn off
    time.sleep(FLASH_POST_DELAY)
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.LOW)

    # Restore live camera settings
    live_settings = read_trackbar_settings("Live")
    apply_camera_settings(cap, live_settings)

    if not ret:
        print("Error: Failed to capture frame during flash.")
        return None
    return frame


def save_photo(frame):
    """Save frame to the photos directory and return the file path."""
    os.makedirs(PHOTO_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"retina_{timestamp}.png"
    filepath = os.path.join(PHOTO_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"Photo saved: {filepath}")
    return filepath


def main():
    # ── Mode selection ──
    print("\n╔══════════════════════════════════════╗")
    print("║   Retina Flash Photography           ║")
    print("╠══════════════════════════════════════╣")
    print("║  1) Manual  – press SPACE to capture ║")
    print("║  2) Auto    – capture when pupil is  ║")
    print("║              centred (Orlosky det.)   ║")
    print("╚══════════════════════════════════════╝")
    choice = input("Select mode [1/2] (default 1): ").strip()
    auto_mode = (choice == "2")
    mode_label = "AUTO (pupil-centred)" if auto_mode else "MANUAL (SPACE)"
    print(f"\n→ Mode: {mode_label}")
    if auto_mode:
        print(f"  Centre tolerance : {CENTRE_TOLERANCE_PX} px")
        print(f"  Cooldown         : {AUTO_COOLDOWN_SECS} s")

    setup_gpio()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Minimise internal buffer so captured frame is current
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        cleanup_gpio()
        return

    print("╔══════════════════════════════════════╗")
    print("║   Retina Flash Photography           ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  LED pin : GPIO {LED_PIN} (BCM)             ║")
    print(f"║  Photos  : ~/Desktop/retina_photos/  ║")
    print(f"║  Mode    : {mode_label:<27s}║")
    print("║  SPACE   : Flash + Capture            ║")
    print("║  Q       : Quit                       ║")
    print("╚══════════════════════════════════════╝")

    cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_WINDOW, CAMERA_WIDTH, CAMERA_HEIGHT)

    create_settings_trackbars()
    # Apply initial live settings
    apply_camera_settings(cap, read_trackbar_settings("Live"))

    last_time = time.time()
    photo_window_open = False
    photo_count = 0
    last_auto_capture = 0.0  # timestamp of last auto-capture
    cooldown_remaining = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # FPS calculation
            now = time.time()
            dt = now - last_time
            fps = 1.0 / dt if dt > 0 else 0
            last_time = now

            # Apply live camera settings every frame (follows trackbar changes)
            apply_camera_settings(cap, read_trackbar_settings("Live"))

            # ── Pupil detection (always run in auto mode, draws on frame) ──
            #pupil_info = None
            pupil_centred = False
            pupil_info = detect_pupil(frame)
            if pupil_info is not None:
                px, py, _ = pupil_info
                frame_cx, frame_cy = frame.shape[1] // 2, frame.shape[0] // 2
                dist = ((px - frame_cx) ** 2 + (py - frame_cy) ** 2) ** 0.5
                pupil_centred = dist <= CENTRE_TOLERANCE_PX

            # Cooldown tracking
            cooldown_remaining = max(0.0, AUTO_COOLDOWN_SECS - (now - last_auto_capture))

            # Overlay info on live feed (on a copy so raw frame stays clean)
            display = frame.copy()
            live_s = read_trackbar_settings("Live")
            cv2.putText(display, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Photos: {photo_count}  Gain:{live_s['gain']}  Exp:{live_s['exposure']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if auto_mode:
                # Show auto-mode status
                if cooldown_remaining > 0:
                    status = f"COOLDOWN {cooldown_remaining:.1f}s"
                    colour = (0, 165, 255)  # orange
                elif pupil_centred:
                    status = "PUPIL CENTRED - CAPTURING"
                    colour = (0, 255, 0)
                elif pupil_info is not None:
                    status = "PUPIL FOUND - NOT CENTRED"
                    colour = (0, 255, 255)  # yellow
                else:
                    status = "NO PUPIL DETECTED"
                    colour = (0, 0, 255)
                cv2.putText(display, status, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                # Draw tolerance circle
                h, w = display.shape[:2]
                cv2.circle(display, (w // 2, h // 2), CENTRE_TOLERANCE_PX, (255, 255, 0), 1)

            cv2.putText(display, "SPACE: Capture  Q: Quit",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Draw centre crosshair to help aiming
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2
            cross_size = 20
            cv2.line(display, (cx - cross_size, cy), (cx + cross_size, cy), (0, 0, 255), 1)
            cv2.line(display, (cx, cy - cross_size), (cx, cy + cross_size), (0, 0, 255), 1)

            cv2.imshow(LIVE_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF

            # ── Trigger capture (manual or auto) ──
            do_capture = False
            if key == ord(' '):
                do_capture = True
            elif auto_mode and pupil_centred and cooldown_remaining <= 0:
                do_capture = True

            if do_capture:
                print("Flash + capture...")
                photo = flash_and_capture(cap)
                if photo is not None:
                    save_photo(photo)
                    photo_count += 1
                    last_auto_capture = time.time()

                    # Show captured photo in its own window
                    if not photo_window_open:
                        cv2.namedWindow(PHOTO_WINDOW, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(PHOTO_WINDOW, CAMERA_WIDTH, CAMERA_HEIGHT)
                        photo_window_open = True
                    cv2.imshow(PHOTO_WINDOW, photo)

            elif key == ord('q'):
                print("Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cleanup_gpio()
        print(f"Done. {photo_count} photo(s) captured.")


if __name__ == "__main__":
    main()
