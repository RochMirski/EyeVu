#!/usr/bin/env python3
"""
Offline pupil-detection test harness
────────────────────────────────────

Re-runs the Orlosky pupil detector from cap.py on the ambient/flash capture
pairs that the Raspberry Pi pushes into the Transfers/ folder, and writes an
annotated flash image into each capture folder for visual inspection.

This lets the detector be tuned on the dev machine against real captures,
without a Pi or a camera attached.

LAYOUT EXPECTED IN Transfers/
─────────────────────────────
Transfers/
    capture_YYYYMMDD_HHMMSS/
        ambient.jpg     raw ambient image  — detector input
        flash.jpg       raw flash image    — overlay is drawn onto this
        meta.json       capture metadata   — live_rotation, params (optional)
    capture_.../
        ...

Each run writes  annotated.jpg  next to the inputs in every capture folder.

USAGE
─────
    python test_pupil_detection.py                  # all capture folders
    python test_pupil_detection.py capture_test     # a single folder (name)
    python test_pupil_detection.py path/to/folder   # a single folder (path)

The detector itself lives in cap.py and is reused unchanged via
cap.detect_and_annotate(), so this harness and the Pi run identical detection.
Edit cap.SWIRSKI_DEBUG / cap.detect_pupil parameters in cap.py to tune.
"""

import os
import sys
import json

import numpy as np
from PIL import Image

# cap.py guards its picamera2 / RPi.GPIO imports, so it imports cleanly here
# (cv2, PIL and numpy are all that the detector needs).
import cap


HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFERS_DIR = os.path.join(HERE, "Transfers")


def load_capture(folder):
    """Load (ambient_array, flash_array, meta) from a capture folder.

    Images are returned as RGB uint8 arrays — the same format picam2's
    capture_array() produces, which is what cap.detect_and_annotate() expects.
    Returns None if either image is missing.
    """
    ambient_path = os.path.join(folder, "ambient.jpg")
    flash_path = os.path.join(folder, "flash.jpg")

    if not os.path.isfile(ambient_path) or not os.path.isfile(flash_path):
        print(f"  SKIP — missing ambient.jpg or flash.jpg in {folder}")
        return None

    ambient = np.array(Image.open(ambient_path).convert("RGB"))
    flash = np.array(Image.open(flash_path).convert("RGB"))

    meta = {}
    meta_path = os.path.join(folder, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  WARN — could not read meta.json: {e}")

    return ambient, flash, meta


def run_one(folder):
    """Run detection on a single capture folder and write annotated.jpg.

    Returns the number of overlays drawn (candidate count in debug mode, 0 or 1
    in normal mode), or -1 if the folder could not be processed.
    """
    name = os.path.basename(os.path.normpath(folder))
    loaded = load_capture(folder)
    if loaded is None:
        return -1
    ambient, flash, meta = loaded

    # Rotate exactly as the Pi did at capture time.  detect_and_annotate() and
    # process_image() read cap.LIVE_ROTATION as a module global, so set it here.
    if "live_rotation" in meta:
        cap.LIVE_ROTATION = int(meta["live_rotation"])

    img, overlays = cap.detect_and_annotate(ambient, flash)

    out_path = os.path.join(folder, "annotated.jpg")
    img.save(out_path)

    if overlays:
        status = f"{len(overlays)} overlay(s)"
    else:
        status = "no pupil detected"
    print(f"  {name}: {status}  ->  {os.path.relpath(out_path, HERE)}")
    return len(overlays)


def find_capture_folders():
    """Return sorted paths of all Transfers/capture_* folders."""
    if not os.path.isdir(TRANSFERS_DIR):
        return []
    folders = []
    for entry in sorted(os.listdir(TRANSFERS_DIR)):
        path = os.path.join(TRANSFERS_DIR, entry)
        if os.path.isdir(path) and entry.startswith("capture_"):
            folders.append(path)
    return folders


def main():
    if not cap.CV2_AVAILABLE:
        print("ERROR: cv2 is not available — install opencv-python.")
        sys.exit(1)

    mode = "debug (all candidates)" if cap.SWIRSKI_DEBUG else "normal (best pupil)"
    print(f"Pupil-detection test harness - detector mode: {mode}\n")

    # Optional CLI arg: a single capture folder, by name or by path.
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        folder = arg if os.path.isdir(arg) else os.path.join(TRANSFERS_DIR, arg)
        if not os.path.isdir(folder):
            print(f"ERROR: capture folder not found: {arg}")
            sys.exit(1)
        folders = [folder]
    else:
        folders = find_capture_folders()

    if not folders:
        print(f"No capture folders found in {TRANSFERS_DIR}")
        print("Capture some images on the Pi first, or drop an ambient.jpg +")
        print("flash.jpg pair into Transfers/capture_test/ to test the harness.")
        return

    print(f"Processing {len(folders)} capture folder(s):")
    processed = 0
    for folder in folders:
        if run_one(folder) >= 0:
            processed += 1

    print(f"\nDone - {processed}/{len(folders)} folder(s) processed.")
    print("Open each annotated.jpg to inspect the detected pupil overlay.")


if __name__ == "__main__":
    main()
