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

# Per-stage debug output of the detection pipeline.  Toggle here (or via the
# DEBUG_MODE env var):
#   "off"      no stage output, just annotated.jpg
#   "stages"   one stage_N_<name>.jpg per pipeline stage in each capture folder
#   "montage"  a single labelled grid, debug_montage.jpg, per capture folder
DEBUG_MODE = os.environ.get("DEBUG_MODE", "montage")


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

    # Optional flash+ambient combined frame (new 3-frame capture).  When present
    # it is the input the iris-band concentric detector will use; for now we load
    # it so a channel/gradient diagnostic can be written for it.
    both = None
    both_path = os.path.join(folder, "both.jpg")
    if os.path.isfile(both_path):
        both = np.array(Image.open(both_path).convert("RGB"))

    meta = {}
    meta_path = os.path.join(folder, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  WARN — could not read meta.json: {e}")

    return ambient, flash, meta, both


def _label_bar(img, text):
    """Return `img` (a BGR uint8 array) with a black caption bar + `text` on top."""
    h, w = img.shape[:2]
    bar = np.zeros((22, w, 3), dtype=np.uint8)
    cap.cv2.putText(bar, text, (4, 16), cap.cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cap.cv2.LINE_AA)
    return np.vstack([bar, img])


def save_both_diagnostic(folder, both_rgb, live_rotation):
    """Write a channel/gradient diagnostic for the flash+ambient combined frame.

    On `both.jpg` the pupil retroreflects bright (best in RED) while the iris /
    limbus is ambient-lit structure (best in GREEN), and the two boundaries differ
    by gradient sign.  This panel (Red | Green | Sobel-magnitude) is what the
    iris-band detector will be designed against — written only when a combined
    frame exists, so it is dormant until the new capture mode produces data.
    """
    cv2 = cap.cv2
    bgr = cv2.cvtColor(both_rgb, cv2.COLOR_RGB2BGR)
    rot = cap._CV2_ROTATIONS[live_rotation]
    if rot is not None:
        bgr = cv2.rotate(bgr, rot)
    b, g, r = cv2.split(bgr)
    gb = cv2.GaussianBlur(g, (7, 7), 0)
    mag = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=5),
                        cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=5))
    mag = np.uint8(255 * mag / (mag.max() + 1e-6))

    def lab(im, t):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        return im

    row = np.hstack([lab(r, "Red (bright pupil)"),
                     lab(g, "Green (iris/limbus)"),
                     lab(mag, "grad mag")])
    cv2.imwrite(os.path.join(folder, "both_diag.jpg"), row)


def save_debug_stages(folder, stages, mode):
    """Write the collected per-stage debug images for one capture folder.

    `stages` is the list cap.detect_pupil() filled — (name, BGR_image) tuples.
    `mode` is "stages" (one stage_N_<name>.jpg each) or "montage" (one grid).
    """
    if not stages or mode not in ("stages", "montage"):
        return
    cv2 = cap.cv2

    if mode == "stages":
        for i, (name, img) in enumerate(stages, 1):
            path = os.path.join(folder, f"stage_{i}_{name}.jpg")
            cv2.imwrite(path, img)
        return

    # montage — scale every stage to a common width, caption, stack in a grid
    cell_w = 360
    cells = []
    for name, img in stages:
        h, w = img.shape[:2]
        scaled = cv2.resize(img, (cell_w, max(1, int(h * cell_w / w))))
        cells.append(_label_bar(scaled, name))
    row_h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, row_h - c.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for c in cells]
    cols = 3
    rows = []
    for r in range(0, len(cells), cols):
        row = cells[r:r + cols]
        while len(row) < cols:                       # pad short final row
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    montage = np.vstack(rows)
    cv2.imwrite(os.path.join(folder, "debug_montage.jpg"), montage)


def run_one(folder):
    """Run detection on a single capture folder and write annotated.jpg.

    Returns the number of overlays drawn (candidate count in debug mode, 0 or 1
    in normal mode), or -1 if the folder could not be processed.
    """
    name = os.path.basename(os.path.normpath(folder))
    loaded = load_capture(folder)
    if loaded is None:
        return -1
    ambient, flash, meta, both = loaded

    # Rotate exactly as the Pi did at capture time.  detect_and_annotate() and
    # process_image() read cap.LIVE_ROTATION as a module global, so set it here.
    if "live_rotation" in meta:
        cap.LIVE_ROTATION = int(meta["live_rotation"])

    # Combined flash+ambient frame: write its channel/gradient diagnostic (the
    # input the iris-band detector will be built against).
    if both is not None:
        save_both_diagnostic(folder, both, int(meta.get("live_rotation", 1)))

    # Arm the per-stage debug sink so detect_pupil() records intermediates.
    if DEBUG_MODE in ("stages", "montage"):
        cap.PUPIL_DEBUG_STAGES = []

    img, overlays = cap.detect_and_annotate(ambient, flash)

    save_debug_stages(folder, cap.PUPIL_DEBUG_STAGES or [], DEBUG_MODE)
    cap.PUPIL_DEBUG_STAGES = None

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
