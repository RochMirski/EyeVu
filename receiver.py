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

import os
import re
import sys
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
        self._reply(200, "ok")

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
