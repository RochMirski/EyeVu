#!/usr/bin/env python3
"""Unit tests for guidance.py (no hardware, no torch needed).

Run: python test_guidance.py
"""

import numpy as np

import guidance


def test_target_centre():
    assert guidance.target_point((1000, 600)) == (300, 500)


def test_target_cover_top_mid():
    # Cover = a block across the bottom third (bottom half) -> pupil sits ABOVE it,
    # so the target is the cover's TOP edge-middle.
    mask = np.zeros((1000, 600), np.uint8)
    mask[700:, 100:500] = 255
    tx, ty = guidance.target_point((1000, 600), mask, mode="cover_top_mid")
    assert ty == 700, ty
    assert 290 <= tx <= 310, tx          # middle of the 100..500 block


def test_target_cover_top_mid_top_cover():
    # Cover = a block across the top third (top half) -> pupil sits BELOW it, so the
    # target is the cover's BOTTOM edge-middle (auto-detected side).
    mask = np.zeros((1000, 600), np.uint8)
    mask[:300, 100:500] = 255
    tx, ty = guidance.target_point((1000, 600), mask, mode="cover_top_mid")
    assert ty == 299, ty                 # bottom edge of the 0..299 cover rows
    assert 290 <= tx <= 310, tx


def test_centred():
    t = guidance.GuidanceTracker()
    g = t.update((300, 500), (1000, 600), (300, 500))
    assert g.state == "centred", g.state


def test_direction_up_left():
    t = guidance.GuidanceTracker()
    # pupil above-and-left of target -> "up" and "left"
    g = t.update((200, 300), (1000, 600), (300, 500))
    assert g.state == "move"
    assert "up" in g.instruction and "left" in g.instruction, g.instruction


def test_direction_down_right():
    t = guidance.GuidanceTracker()
    g = t.update((450, 700), (1000, 600), (300, 500))
    assert "down" in g.instruction and "right" in g.instruction, g.instruction


def test_relative_keep_going():
    t = guidance.GuidanceTracker()
    t.update((150, 500), (1000, 600), (300, 500))      # far left
    g = t.update((230, 500), (1000, 600), (300, 500))  # closer -> improved
    assert "keep going" in g.instruction or "almost" in g.instruction, g.instruction


def test_relative_wrong_way():
    t = guidance.GuidanceTracker()
    t.update((250, 500), (1000, 600), (300, 500))      # close
    g = t.update((100, 500), (1000, 600), (300, 500))  # farther -> worse
    assert "wrong way" in g.instruction, g.instruction


def test_searching_when_no_pupil():
    t = guidance.GuidanceTracker()
    g = t.update(None, (1000, 600), (300, 500))
    assert g.state == "searching", g.state


def test_annotate_runs():
    frame = np.zeros((640, 480, 3), np.uint8)
    t = guidance.GuidanceTracker()
    g = t.update((100, 100), (480, 640), (240, 320))
    out = guidance.annotate(frame, g, (240, 320), (100, 100))
    assert out.shape == frame.shape


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
