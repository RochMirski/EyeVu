#!/usr/bin/env python3
"""
Retina Flash Photography - Raspberry Pi Zero (Manual Mode Only, No OpenCV, No GUI)

Manual Mode: Press Enter to flash LED and capture.

Photos are saved to ~/Desktop/retina_photos/ with timestamps.
Run 'rpicam-vid' in another terminal for live video feed.
Press 'q' + Enter to quit.
"""

from picamera2 import Picamera2
import time
import os
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
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FLASH_GAIN = 0.5
LIVE_GAIN = 1.0
# ────────────────────────────────────────────────


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


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   Retina Flash Photography           ║")
    print("╠══════════════════════════════════════╣")
    print("║  Manual Mode: Press Enter to capture ║")
    print("║  Run 'rpicam-vid' for live feed       ║")
    print("╚══════════════════════════════════════╝")
    print("→ Mode: MANUAL (Enter)")

    setup_gpio()

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)})
    picam2.configure(config)
    picam2.set_controls({"AnalogueGain": LIVE_GAIN})
    picam2.start()

    photo_count = 0

    try:
        print("Ready. Press Enter to capture, or 'q' + Enter to quit.")
        while True:
            cmd = input().strip()
            if cmd == 'q':
                break
            elif cmd == '':
                print("Flash + capture...")
                # Set flash gain
                picam2.set_controls({"AnalogueGain": FLASH_GAIN})
                time.sleep(0.02)

                if GPIO_AVAILABLE:
                    GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(FLASH_PRE_DELAY)

                array = picam2.capture_array()

                if GPIO_AVAILABLE:
                    GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(FLASH_POST_DELAY)

                # Restore live gain
                picam2.set_controls({"AnalogueGain": LIVE_GAIN})

                # Save
                from PIL import Image
                img = Image.fromarray(array)
                os.makedirs(PHOTO_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"retina_{timestamp}.png"
                filepath = os.path.join(PHOTO_DIR, filename)
                img.save(filepath)
                print(f"Photo saved: {filepath}")
                os.system(f"xdg-open '{filepath}'")
                photo_count += 1

    except KeyboardInterrupt:
        print("Quitting...")

    finally:
        picam2.stop()
        cleanup_gpio()
        print(f"Done. {photo_count} photo(s) captured.")


if __name__ == "__main__":
    main()