#!/usr/bin/env python3
"""
Build the LED-cover calibration mask from a calibration image.

Capture the LED cover with NO eye in place — ideally the cover against a bright
white background, so the cover is the only dark thing in frame.  Point this
script at that image and it writes calibration/led_cover_mask.png, which
cap.load_cover_mask() then applies during detection (ray edges that fall inside
the cover are dropped, so the occluder can't drag the pupil fit).

    python build_cover_calibration.py path/to/cover_on_white.jpg [--rot N]

--rot N is the LIVE_ROTATION index the device captures at (default cap.LIVE_ROTATION);
the mask is stored in that display orientation, the one detection runs in.

(On the Pi you can instead just run the `calibrate` command in cap.py, which
captures the frame and builds the mask in one step.)
"""

import os
import sys
import argparse

import cv2
import numpy as np
from PIL import Image

import cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="calibration image (LED cover on white, no eye)")
    ap.add_argument("--rot", type=int, default=cap.LIVE_ROTATION,
                    help="LIVE_ROTATION index used at capture time")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"Not found: {args.image}")
        sys.exit(1)

    bgr = cv2.cvtColor(np.array(Image.open(args.image).convert("RGB")),
                       cv2.COLOR_RGB2BGR)
    rot = cap._CV2_ROTATIONS[args.rot]
    if rot is not None:
        bgr = cv2.rotate(bgr, rot)

    mask = cap.build_cover_mask(bgr)
    if mask is None:
        print("No cover-sized dark region found — is the background bright enough? "
              "(tune cap._COVER_CALIB_DARK)")
        sys.exit(1)

    cap.save_cover_calibration(mask, bgr)
    frac = 100.0 * float((mask > 0).sum()) / mask.size
    print(f"Cover mask: {frac:.1f}% of frame -> {cap.COVER_CALIB_MASK_PATH}")

    preview = bgr.copy()
    preview[mask > 0, 2] = 255
    pv = os.path.join(cap.COVER_CALIB_DIR, "_calib_preview.jpg")
    cv2.imwrite(pv, preview)
    print(f"Preview -> {pv}")


if __name__ == "__main__":
    main()
