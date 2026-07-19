#!/usr/bin/env python3
"""Unit tests for cap.redeye_extract (no hardware).

Run: python test_redeye.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

import cap


def _synthetic():
    """flash (red fundus disc + white specular), near-zero dark, ambient dark-disc."""
    h, w = 240, 320
    cx, cy, r = 160, 120, 30
    flash = np.zeros((h, w, 3), np.uint8)
    cv2.circle(flash, (cx, cy), r, (0, 0, 200), -1)        # red disc (BGR)
    cv2.circle(flash, (170, 110), 6, (255, 255, 255), -1)  # white specular on it
    dark = np.zeros((h, w, 3), np.uint8)                   # near-black
    ambient = np.full((h, w, 3), 110, np.uint8)            # lit eye
    cv2.circle(ambient, (cx, cy), r, (20, 20, 20), -1)     # dark pupil disc
    return flash, dark, ambient, (cx, cy), r


def test_selects_red_excludes_specular():
    flash, dark, ambient, center, r = _synthetic()
    res = cap.redeye_extract(flash, dark, ambient, center, r)
    assert res.valid, res.notes
    assert res.coverage > 0.0
    assert res.mask[120, 160] == 255, "red disc centre should be selected"
    assert res.mask[110, 170] == 0, "white specular must be excluded"
    # extract keeps the red pixel, blacks the specular
    assert tuple(int(v) for v in res.extract[120, 160]) == (0, 0, 200)
    assert tuple(int(v) for v in res.extract[110, 170]) == (0, 0, 0)


def test_gate_on_cover():
    flash, dark, ambient, center, r = _synthetic()
    cover = np.zeros((240, 320), np.uint8)
    cv2.circle(cover, center, 40, 255, -1)                 # pupil sits on the cover
    res = cap.redeye_extract(flash, dark, ambient, center, r, cover_mask=cover)
    assert not res.valid and "cover" in res.notes, res.notes
    assert res.overlay is not None                          # banner still rendered


def test_gate_no_pupil():
    flash, dark, ambient, _, _ = _synthetic()
    res = cap.redeye_extract(flash, dark, ambient, None, None)
    assert not res.valid and "no pupil" in res.notes, res.notes


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
