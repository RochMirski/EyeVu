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


def test_in_valid_region():
    # frame 1000x600 -> scale 600, dead=36px, valid=72px.  Pupil 50px from target:
    t = guidance.GuidanceTracker()
    g = t.update((350, 500), (1000, 600), (300, 500))
    assert g.state == "move", g.state              # 50 > dead(36)
    assert g.in_valid_region, "50px < valid(72) should be in the valid region"


def test_endpoint_after_valid():
    # pupil reaches the valid region, then is lost -> aligned/blocked endpoint.
    t = guidance.GuidanceTracker()
    t.update((350, 500), (1000, 600), (300, 500))  # in valid region
    g = t.update(None, (1000, 600), (300, 500))    # then lost
    assert g.state == "reached", g.state
    assert g.endpoint and g.reached_valid, (g.endpoint, g.reached_valid)


def test_no_endpoint_without_valid():
    # pupil never entered the valid region -> losing it is just "searching".
    t = guidance.GuidanceTracker()
    t.update((150, 500), (1000, 600), (300, 500))  # 150px > valid(72): not valid
    g = t.update(None, (1000, 600), (300, 500))
    assert g.state == "searching" and not g.endpoint, (g.state, g.endpoint)


# ── approach-stage camera direction ──
# The LED cover is bolted to the scope, so it never moves in the frame; closing
# the cover/pupil gap means moving the PUPIL toward that edge, and image content
# travels OPPOSITE to camera motion.  Getting this backwards drives the cover away
# from the pupil, so it is pinned down here in both words and arrow.

def test_close_word_top_cover_moves_camera_down():
    assert guidance._close_word("top") == "down"


def test_close_word_bottom_cover_moves_camera_up():
    assert guidance._close_word("bottom") == "up"


def test_camera_hint_top_cover_says_down():
    h = guidance.camera_hint((300, 500), (300, 500), (1000, 600), close_to="top")
    assert "down" in h and "up" not in h, h


def test_camera_hint_bottom_cover_says_up():
    h = guidance.camera_hint((300, 500), (300, 500), (1000, 600), close_to="bottom")
    assert "up" in h and "down" not in h, h


def test_camera_arrow_agrees_with_words():
    # Screen y grows downward, so "down" must be a POSITIVE dy.
    for side, want_down in (("top", True), ("bottom", False)):
        words = guidance.camera_hint((300, 500), (300, 500), (1000, 600),
                                     close_to=side)
        vec = guidance.camera_arrow((300, 500), (300, 500), (1000, 600),
                                    close_to=side)
        assert vec is not None, side
        assert (vec[1] > 0) == want_down, (side, words, vec)
        assert ("down" in words) == (vec[1] > 0), (side, words, vec)


def test_camera_arrow_keeps_vertical_dominant():
    # A modest sideways drift must not swamp the approach's vertical push: the
    # two components come from different worlds (pixels vs a unit "keep going").
    vec = guidance.camera_arrow((330, 500), (300, 500), (1000, 600), close_to="top")
    assert abs(vec[1]) > abs(vec[0]), vec


def test_camera_hint_hold_steady_when_on_anchor():
    h = guidance.camera_hint((300, 500), (300, 500), (1000, 600))
    assert "hold steady" in h, h


# ── overshoot recovery ──
# Driving the pupil PAST the cover leaves the approach with no reflex to find, so
# the instruction has to reverse.  It must be the exact opposite of the approach
# move, or the operator is sent further into the failure they are already in.

def test_back_off_is_the_reverse_of_the_approach():
    for side in ("top", "bottom"):
        assert guidance._back_off_word(side) != guidance._close_word(side), side


def test_back_off_hint_top_cover_says_up():
    # Top cover: the approach moves the camera DOWN, so backing off is UP.
    h = guidance.back_off_hint("top")
    assert "up" in h and "down" not in h, h


def test_back_off_hint_bottom_cover_says_down():
    h = guidance.back_off_hint("bottom")
    assert "down" in h and "up" not in h, h


def test_back_off_arrow_agrees_with_words():
    for side in ("top", "bottom"):
        words = guidance.back_off_hint(side)
        vec = guidance.back_off_vector(side, (1000, 600))
        assert ("down" in words) == (vec[1] > 0), (side, words, vec)


def test_back_off_arrow_opposes_the_approach_arrow():
    for side in ("top", "bottom"):
        fwd = guidance.camera_arrow((300, 500), (300, 500), (1000, 600),
                                    close_to=side)
        back = guidance.back_off_vector(side, (1000, 600))
        assert fwd[1] * back[1] < 0, (side, fwd, back)


# ── reversing the approach ──
# The reflex has a sweet spot in the cover/pupil gap and the pupil looks fine on
# both sides of it, so "keep closing" can be an instruction to walk away from the
# only place the reflex exists.  The vertical half has to be reversible.

def test_vertical_word_reverses():
    for side in ("top", "bottom"):
        assert (guidance._vertical_word(side, reverse=True)
                != guidance._vertical_word(side, reverse=False))
        assert guidance._vertical_word(side, False) == guidance._close_word(side)


def test_camera_hint_reverse_flips_the_vertical():
    fwd = guidance.camera_hint((300, 500), (300, 500), (1000, 600), close_to="top")
    rev = guidance.camera_hint((300, 500), (300, 500), (1000, 600), close_to="top",
                               reverse=True)
    assert "down" in fwd and "up" in rev, (fwd, rev)


def test_camera_arrow_reverse_flips_the_vertical():
    for side in ("top", "bottom"):
        fwd = guidance.camera_arrow((300, 500), (300, 500), (1000, 600),
                                    close_to=side)
        rev = guidance.camera_arrow((300, 500), (300, 500), (1000, 600),
                                    close_to=side, reverse=True)
        assert fwd[1] * rev[1] < 0, (side, fwd, rev)


def test_reversed_words_and_arrow_still_agree():
    for side in ("top", "bottom"):
        words = guidance.camera_hint((300, 500), (300, 500), (1000, 600),
                                     close_to=side, reverse=True)
        vec = guidance.camera_arrow((300, 500), (300, 500), (1000, 600),
                                    close_to=side, reverse=True)
        assert ("down" in words) == (vec[1] > 0), (side, words, vec)


def test_reverse_keeps_the_lateral_correction():
    # Reversing is a VERTICAL decision; sideways drift is still corrected.
    h = guidance.camera_hint((400, 500), (300, 500), (1000, 600), close_to="top",
                             reverse=True)
    assert "up" in h and "right" in h, h


# ── the readout drawn beside the fixation target ──

def test_annotate_draws_the_readout():
    frame = np.zeros((640, 480, 3), np.uint8)
    g = guidance.Guidance(state="approach", instruction=guidance.INSTRUCTION_LOOK,
                          hint="Camera: move up", readout="red 300 / 650 px")
    plain = guidance.annotate(frame, guidance.Guidance(
        state="approach", instruction=guidance.INSTRUCTION_LOOK,
        hint="Camera: move up"), (240, 320), (100, 100), fixation=True)
    with_read = guidance.annotate(frame, g, (240, 320), (100, 100), fixation=True)
    assert not np.array_equal(plain, with_read), "readout was not drawn"


def test_annotate_readout_is_green_when_met():
    frame = np.zeros((640, 480, 3), np.uint8)
    args = dict(state="approach", instruction=guidance.INSTRUCTION_LOOK,
                readout="red 900 / 650 px")
    unmet = guidance.annotate(frame, guidance.Guidance(**args, readout_ok=False),
                              (240, 320), None, fixation=True)
    met = guidance.annotate(frame, guidance.Guidance(**args, readout_ok=True),
                            (240, 320), None, fixation=True)
    assert not np.array_equal(unmet, met), "met/unmet readouts look identical"


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
