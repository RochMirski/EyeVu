#!/usr/bin/env python3
"""
Retina Flash Photography - Raspberry Pi Zero (Manual Mode Only, No OpenCV)

Manual Mode: Press SPACE to flash LED and capture.

The captured photo is displayed below the live feed.
Photos are saved to ~/Desktop/retina_photos/ with timestamps.
Press 'q' to quit.
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
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
LIVE_GAIN = 1.0
FLASH_GAIN = 0.5
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


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Retina Flash Photography - Manual Mode")

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)})
        self.picam2.configure(config)
        self.picam2.start()

        self.live_label = tk.Label(root, text="Live Feed")
        self.live_label.pack()
        self.label = tk.Label(root)
        self.label.pack()

        self.photo_label = tk.Label(root, text="Captured Photo")
        self.photo_label.pack()
        self.photo_display = tk.Label(root)
        self.photo_display.pack()

        self.photo_count = 0
        self.last_time = time.time()

        setup_gpio()
        self.update_image()

        self.root.bind('<space>', self.capture)
        self.root.bind('<Key-q>', self.quit_app)

    def update_image(self):
        array = self.picam2.capture_array()
        img = Image.fromarray(array)

        # Overlay info
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        now = time.time()
        dt = now - self.last_time
        fps = 1.0 / dt if dt > 0 else 0
        self.last_time = now

        draw.text((10, 10), f"FPS: {int(fps)}", fill=(0, 255, 0), font=font)
        draw.text((10, 30), f"Photos: {self.photo_count}", fill=(0, 255, 0), font=font)
        draw.text((10, img.height - 20), "SPACE: Capture  Q: Quit", fill=(200, 200, 200), font=font)

        # Crosshair
        cx, cy = img.width // 2, img.height // 2
        draw.line((cx - 20, cy, cx + 20, cy), fill=(0, 0, 255), width=1)
        draw.line((cx, cy - 20, cx, cy + 20), fill=(0, 0, 255), width=1)

        self.imgtk = ImageTk.PhotoImage(img)
        self.label.config(image=self.imgtk)

        self.root.after(30, self.update_image)  # Update ~30fps

    def capture(self, event):
        print("Flash + capture...")
        # Set flash gain
        self.picam2.set_controls({"AnalogueGain": FLASH_GAIN})
        time.sleep(0.02)

        if GPIO_AVAILABLE:
            GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(FLASH_PRE_DELAY)

        array = self.picam2.capture_array()

        if GPIO_AVAILABLE:
            GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(FLASH_POST_DELAY)

        # Restore live gain
        self.picam2.set_controls({"AnalogueGain": LIVE_GAIN})

        # Save
        img = Image.fromarray(array)
        os.makedirs(PHOTO_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"retina_{timestamp}.png"
        filepath = os.path.join(PHOTO_DIR, filename)
        img.save(filepath)
        print(f"Photo saved: {filepath}")

        self.photo_count += 1

        # Display captured photo
        self.photo_imgtk = ImageTk.PhotoImage(img)
        self.photo_display.config(image=self.photo_imgtk)

    def quit_app(self, event):
        self.picam2.stop()
        cleanup_gpio()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    print(f"Done. {app.photo_count} photo(s) captured.")