#!/usr/bin/env python3
"""
Retina Flash Photography - Raspberry Pi Zero (Manual Mode Only)

Manual Mode: Press SPACE to flash LED and capture.

The captured photo is displayed in a second window until the next capture or quit.
Photos are saved to ~/Desktop/retina_photos/ with timestamps.
Press 'q' to quit.
"""

import cv2
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
    print("\n╔══════════════════════════════════════╗")
    print("║   Retina Flash Photography           ║")
    print("╠══════════════════════════════════════╣")
    print("║  Manual Mode: Press SPACE to capture ║")
    print("╚══════════════════════════════════════╝")
    print("→ Mode: MANUAL (SPACE)")

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
    print("║  Mode    : MANUAL (SPACE)             ║")
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

            # Overlay info on live feed (on a copy so raw frame stays clean)
            display = frame.copy()
            live_s = read_trackbar_settings("Live")
            cv2.putText(display, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Photos: {photo_count}  Gain:{live_s['gain']}  Exp:{live_s['exposure']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

            # ── Trigger capture (manual only) ──
            if key == ord(' '):
                print("Flash + capture...")
                photo = flash_and_capture(cap)
                if photo is not None:
                    save_photo(photo)
                    photo_count += 1

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