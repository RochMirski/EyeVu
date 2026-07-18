#!/usr/bin/env python3
"""
EyeVu capture receiver
──────────────────────

Tiny stdlib HTTP server that accepts capture uploads from the Pi (cap.py's
transfer_capture()) and writes them into Transfers/ on this machine.

Run while you take captures on the Pi:

    python receiver.py

Listens on 0.0.0.0:8000.  Ctrl-C to stop.

PROTOCOL
────────
POST /upload/<folder>/<filename>
    body: raw file bytes (Content-Length must be set)

<folder>   = capture_YYYYMMDD_HHMMSS   -> written under Transfers/<folder>/
           | calibration              -> written under calibration/; an uploaded
                                          led_cover_calib.jpg triggers building the
                                          cover mask HERE (cap.build_cover_mask)
<filename> = ambient.jpg | flash.jpg | both.jpg | meta.json | led_cover_calib.jpg

Anything else returns 400.  No auth — only listen on a trusted local network
(e.g. this machine's Windows mobile hotspot, where the Pi is the only client).
"""

import csv
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "0.0.0.0"
PORT = 8000

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFERS_DIR = os.path.join(HERE, "Transfers")
# The Pi uploads its LED-cover calibration under the special `calibration`
# folder; route it to calibration/ (next to cap.py, where cap.load_cover_mask
# reads it) instead of Transfers/, so one Pi calibration covers this machine too.
CALIBRATION_FOLDER = "calibration"
CALIBRATION_DIR = os.path.join(HERE, "calibration")
CALIB_IMAGE_NAME = "led_cover_calib.jpg"   # the raw calibration frame the Pi ships


# ── Alignment guidance on received captures ───────────────────────────────
# When a capture folder has its full triple, run the coarse-locate cascade
# (ridge + red-eye + RITnet — RITnet runs fine here on the PC) and tell the
# operator which way to move the device.  One tracker, updated in arrival order,
# gives relative phrasing across the session ("keep going" / "almost there").
GUIDANCE_TARGET_MODE = "cover_top_mid" # drive the pupil to the LED-cover inner edge
_GUIDANCE = None
_GUIDANCE_PRIOR = None
_PROCESSED = set()

# Completed captures are sorted into these subfolders of Transfers/ by which
# detector(s) found the pupil (see cap.classify_detection).
CATEGORY_DIRS = ("no_pupil", "ridge_only", "ritnet_only", "both")

# Per-capture detection log appended to Transfers/index.csv (raw centres +
# confidences from both detectors, the category, and the guidance instruction).
_INDEX_LOCK = threading.Lock()
_INDEX_FIELDS = [
    "folder", "category",
    "ridge_x", "ridge_y", "ridge_conf", "ridge_source",
    "ritnet_x", "ritnet_y", "ritnet_conf",
    "agree_dist", "chosen_source", "chosen_x", "chosen_y",
    "off_px", "state", "instruction", "target_x", "target_y",
    "in_valid", "reached_valid", "endpoint", "live_rotation",
]


def _append_index_row(row):
    """Append one detection record to Transfers/index.csv (thread-safe)."""
    path = os.path.join(TRANSFERS_DIR, "index.csv")
    with _INDEX_LOCK:
        try:
            os.makedirs(TRANSFERS_DIR, exist_ok=True)
            new = not os.path.isfile(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError as e:
            print(f"  index.csv write error: {e}")


def _maybe_run_guidance(folder, dest_dir):
    """Run guidance + categorise/sort once a folder has ambient + flash + meta."""
    if folder in _PROCESSED:
        return
    need = ("ambient.jpg", "flash.jpg", "meta.json")
    if not all(os.path.isfile(os.path.join(dest_dir, n)) for n in need):
        return
    _PROCESSED.add(folder)
    try:
        category = _run_guidance(folder, dest_dir)
        if category:
            _sort_capture(folder, dest_dir, category)
    except Exception as e:                       # noqa: BLE001 — never crash receiver
        print(f"  guidance error: {e}")


def _run_guidance(folder, dest_dir):
    """Detect on the received triple; print/draw guidance; return the category.

    Runs the ridge cascade and RITnet separately (cap.detect_both) and sorts the
    capture by which found the pupil (cap.classify_detection); the chosen result
    also drives the alignment guidance.  Returns the category string.
    """
    global _GUIDANCE, _GUIDANCE_PRIOR
    import json
    import numpy as np
    from PIL import Image
    import cap
    import guidance
    cv2 = cap.cv2
    cap.RITNET_ALWAYS = True            # PC: always run RITnet so sorting has both detectors
    if _GUIDANCE is None:
        _GUIDANCE = guidance.GuidanceTracker()

    meta = {}
    try:
        with open(os.path.join(dest_dir, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        pass
    if "live_rotation" in meta:
        cap.LIVE_ROTATION = int(meta["live_rotation"])

    amb = np.array(Image.open(os.path.join(dest_dir, "ambient.jpg")).convert("RGB"))
    fla = np.array(Image.open(os.path.join(dest_dir, "flash.jpg")).convert("RGB"))
    rot = cap._CV2_ROTATIONS[cap.LIVE_ROTATION]
    amb_bgr = cv2.cvtColor(amb, cv2.COLOR_RGB2BGR)
    fla_bgr = cv2.cvtColor(fla, cv2.COLOR_RGB2BGR)
    if rot is not None:
        amb_bgr = cv2.rotate(amb_bgr, rot)
        fla_bgr = cv2.rotate(fla_bgr, rot)

    cover = cap.load_cover_mask(amb_bgr.shape[:2])

    # Run the two detectors separately and categorise this capture.
    ridge, ritnet = cap.detect_both(amb_bgr, fla_bgr, cover, _GUIDANCE_PRIOR,
                                    allow_ml=True)
    category, chosen = cap.classify_detection(ridge, ritnet, amb_bgr.shape)
    center = chosen[0] if chosen else None
    conf = chosen[2] if chosen else 0.0
    source = chosen[3] if chosen else ""

    target = guidance.target_point(fla_bgr.shape, cover, GUIDANCE_TARGET_MODE)
    g = _GUIDANCE.update(center, fla_bgr.shape, target, conf, source)
    _GUIDANCE_PRIOR = center if center is not None else _GUIDANCE_PRIOR

    rs = f"{ridge[2]:.2f}" if ridge else "-"
    ns = f"{ritnet[2]:.2f}" if ritnet else "-"
    verify = ("  >>> ENDPOINT: aligned or blocked (was at cover edge, now lost)"
              if g.endpoint else "  [in valid region]" if g.in_valid_region
              else "  [reached valid region earlier]" if g.reached_valid else "")
    print(f"  >>> GUIDANCE [{folder}]: {g.instruction}"
          + (f"  (off={g.distance:.0f}px {source} conf={conf:.2f})" if center else "")
          + verify)
    print(f"  >>> CATEGORY [{folder}]: {category}  (ridge conf={rs}, ritnet conf={ns})")
    vis = guidance.annotate(cv2.convertScaleAbs(fla_bgr, alpha=3.0), g, target, center)
    cv2.imwrite(os.path.join(dest_dir, "guidance.jpg"), vis)

    # Log the raw detection numbers (both detectors) to Transfers/index.csv.
    def _xy(d):                                  # (x, y) or ("", "")
        return (round(d[0][0], 1), round(d[0][1], 1)) if d else ("", "")
    agree = (round(float(np.hypot(ridge[0][0] - ritnet[0][0],
                                  ridge[0][1] - ritnet[0][1])), 1)
             if ridge and ritnet else "")
    rx, ry = _xy(ridge)
    nx, ny = _xy(ritnet)
    _append_index_row({
        "folder": folder, "category": category,
        "ridge_x": rx, "ridge_y": ry,
        "ridge_conf": round(ridge[2], 3) if ridge else "",
        "ridge_source": ridge[3] if ridge else "",
        "ritnet_x": nx, "ritnet_y": ny,
        "ritnet_conf": round(ritnet[2], 3) if ritnet else "",
        "agree_dist": agree, "chosen_source": source,
        "chosen_x": round(center[0], 1) if center else "",
        "chosen_y": round(center[1], 1) if center else "",
        "off_px": round(g.distance, 1) if center else "",
        "state": g.state, "instruction": g.instruction,
        "target_x": target[0], "target_y": target[1],
        "in_valid": int(g.in_valid_region), "reached_valid": int(g.reached_valid),
        "endpoint": int(g.endpoint),
        "live_rotation": cap.LIVE_ROTATION,
    })
    return category


def _sort_capture(folder, dest_dir, category):
    """Move a completed capture folder into Transfers/<category>/<folder>."""
    import shutil
    target_parent = os.path.join(TRANSFERS_DIR, category)
    target = os.path.join(target_parent, folder)
    if os.path.abspath(dest_dir) == os.path.abspath(target):
        return
    os.makedirs(target_parent, exist_ok=True)
    try:
        if os.path.isdir(target):                # merge into an existing target
            for fn in os.listdir(dest_dir):
                shutil.move(os.path.join(dest_dir, fn), os.path.join(target, fn))
            os.rmdir(dest_dir)
        else:
            shutil.move(dest_dir, target)
        print(f"  sorted {folder} -> {category}/")
    except OSError as e:
        print(f"  sort error ({folder} -> {category}): {e}")


def _build_cover_mask_from(image_path):
    """Build + save the LED-cover mask from an uploaded calibration image.

    Reuses cap.build_cover_mask (near-black + edge-touching + smoothing/fill) and
    save_cover_calibration, so the mask lands at cap.COVER_CALIB_MASK_PATH where
    cap.load_cover_mask reads it.  Fully guarded — the receiver must stay up even
    if cv2/cap are unavailable or the build fails.
    """
    try:
        import cap
        bgr = cap.cv2.imread(image_path)
        mask = cap.build_cover_mask(bgr)
        if mask is not None and cap.save_cover_calibration(mask):
            frac = 100.0 * float((mask > 0).sum()) / mask.size
            print(f"  built cover mask ({frac:.1f}% of frame) -> "
                  f"{cap.COVER_CALIB_MASK_PATH}")
        else:
            print("  cover-mask build failed (no dark edge-touching region?)")
    except Exception as e:                       # noqa: BLE001 — never crash receiver
        print(f"  cover-mask build error: {e}")

# Only safe path segments — no slashes, no "..", no exotic characters.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_BYTES = 50 * 1024 * 1024     # 50 MB cap per upload — sanity guard


def _safe(name):
    return bool(_SAFE_NAME.match(name)) and ".." not in name


class UploadHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Expect path  /upload/<folder>/<filename>
        parts = self.path.lstrip("/").split("/")
        if len(parts) != 3 or parts[0] != "upload":
            self._reply(400, "bad path")
            return

        folder, filename = parts[1], parts[2]
        if not _safe(folder) or not _safe(filename):
            self._reply(400, "unsafe path component")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply(400, "bad Content-Length")
            return
        if length <= 0 or length > MAX_BYTES:
            self._reply(400, f"bad length {length}")
            return

        try:
            data = self.rfile.read(length)
        except OSError as e:
            self._reply(500, f"read failed: {e}")
            return

        # `calibration` is special — it lands in calibration/ (where cap.py loads
        # the cover mask), not under Transfers/.
        if folder == CALIBRATION_FOLDER:
            dest_dir = CALIBRATION_DIR
        else:
            dest_dir = os.path.join(TRANSFERS_DIR, folder)
        dest = os.path.join(dest_dir, filename)
        try:
            os.makedirs(dest_dir, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
        except OSError as e:
            self._reply(500, f"write failed: {e}")
            return

        kb = len(data) / 1024.0
        size = f"{kb:.1f} KB" if kb < 1024 else f"{kb / 1024:.2f} MB"
        print(f"received {folder}/{filename} ({size})")

        # Build the LED-cover mask HERE (on this machine) from an uploaded
        # calibration image — the Pi only ships the image.  Guarded so a missing
        # cv2 / build failure never takes the receiver down.
        if folder == CALIBRATION_FOLDER and filename == CALIB_IMAGE_NAME:
            _build_cover_mask_from(dest)
        self._reply(200, "ok")        # ack the Pi before any heavy detection

        # Once the capture triple is complete, run alignment guidance (off the
        # response path so RITnet inference never blocks the upload).
        if folder != CALIBRATION_FOLDER:
            _maybe_run_guidance(folder, dest_dir)

    def _reply(self, status, msg):
        body = (msg + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    # Quiet the default per-request access log — we already log uploads above.
    def log_message(self, fmt, *args):
        return


def main():
    os.makedirs(TRANSFERS_DIR, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), UploadHandler)
    print(f"Listening on {HOST}:{PORT}, writing to {TRANSFERS_DIR}")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
