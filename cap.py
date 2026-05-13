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

# ──────────────────────────


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

    # Rotate anticlockwise
    img = img.rotate(90, expand=True)

    return img


def capture_image(picam2):

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


def streaming_mode(picam2):
    """Live video feed. Hold SPACE to turn LED on, r to lock/unlock.
    Press ENTER or e to capture a still. Press s to exit."""

    if not CV2_AVAILABLE:
        print("cv2 not available - cannot show live feed.")
        return

    print("\nStreaming mode ON")
    print("SPACE=LED | r=lock | ENTER/e=capture | s=exit\n")

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
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_W, DISPLAY_H)))
    # Wait long enough for the window to fully render
    cv2.waitKey(500)

    last_space_time = 0.0
    led_on = False
    lock_on = False  # Space+Shift toggle

    try:

        while True:

            array = picam2.capture_array()
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_W, DISPLAY_H)))

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
                apply_camera_settings(picam2, FLASH_GAIN)

            # Track spacebar hold via auto-repeat
            elif key == 32:
                last_space_time = now
                space_held = True

            # r toggles LED lock
            elif key == ord('r'):
                lock_on = not lock_on

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
    print("s     = streaming mode (SPACE=LED, r=lock, e/ENTER=capture)")
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