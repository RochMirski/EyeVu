#!/usr/bin/env python3
"""Unit tests for the live alignment stages in cap.py (no hardware needed).

Both stages are driven entirely by wall-clock deadlines, so the clock is
simulated: the tests advance it themselves and step the state machines by hand,
which makes a ~15s gaze scan run in milliseconds and keeps the timing
assertions exact rather than flaky.

Run: python test_stages.py
"""

import math
import time

# Simulated clock, installed BEFORE cap is imported so the module-level `time`
# it captures is this one.  Everything in cap/guidance reads time.time().
_NOW = [1000.0]
_real_time = time.time
time.time = lambda: _NOW[0]

import cap                                                     # noqa: E402
import guidance                                                # noqa: E402

SHAPE = (640, 480, 3)          # display frame: h, w, c
FRAME_DT = 1.0 / 15.0          # live feed frame interval


def _advance(dt):
    _NOW[0] += dt


def _meas(px, valid=True, source="redetect"):
    """One measure_red_fraction result, shaped as the stages expect."""
    return {"red_px": px, "valid": valid, "center": (240.0, 320.0), "radius": 12.0,
            "fraction": 0.4, "source": source, "notes": "", "mask": None,
            "extract": object()}


PX_SCALE = 4.0                 # the scan measures on frames 2x larger each way


def _run_scan(peak_offset=-40.0, limit_s=240.0, valid=True):
    """Drive a whole GazeScan the way the streaming loop does.

    Returns a record of what happened: stops reached, captures fired, frames the
    eye spent lit, and the dark time preceding each capture.
    """
    scan = cap.GazeScan((240, 320), (240.0, 320.0), 12.0, SHAPE,
                        cover_side="bottom", px_scale=PX_SCALE)
    r = {"stops": [], "captures": [], "saves": 0, "lit": 0, "frames": 0,
         "dark_before": []}
    prev_mode = scan.mode
    start = _NOW[0]
    while not scan.done and not scan.abort and _NOW[0] - start < limit_s:
        r["frames"] += 1
        _advance(FRAME_DT)
        scan.tick()
        if scan.lighting() == "flash":
            r["lit"] += 1
        if scan.mode != prev_mode:
            if scan.mode == scan.MODE_HOLD:
                r["stops"].append((round(scan.offset[scan.axis], 1),
                                   _NOW[0] - start))
            prev_mode = scan.mode
        # ── the streaming loop's per-stop capture driver ──
        if scan.ready_to_capture():
            r["dark_before"].append(_NOW[0] - scan.dark_since)
            scan.to_dark_ref()
        elif scan.phase == scan.PHASE_DARK:
            scan.to_flash()                        # dark reference taken
        elif scan.phase == scan.PHASE_FLASH and scan.ready():
            off = scan.offset[scan.axis]
            px = int(1500 * max(0.0, 1.0 - abs(off - peak_offset) / 90.0)) + 600
            m = _meas(px, valid=valid)
            r["captures"].append((round(off, 1), px, scan.mode))
            m["red_px"] = int(px * PX_SCALE)        # the loop measures full-res...
            if scan.worth_saving(m):
                r["saves"] += 1
            m["red_px"] = px                        # ...then normalises for submit
            scan.submit(m)
            scan.to_show()
        elif scan.phase == scan.PHASE_SHOW and scan.ready():
            scan.capture_done()
    return scan, r


# ── gaze scan: one capture per gaze direction, dark in between ──
# A saccade costs ~200-300ms of latency plus flight, so a point in constant motion
# is measured at a gaze angle that is always behind it.  And a continuously lit eye
# is a constricted eye.  So the point stops, the eye sits dark while the gaze
# arrives and the pupil re-opens, and exactly one flash frame is taken.

def test_scan_completes():
    scan, _ = _run_scan()
    assert scan.done and not scan.abort, (scan.done, scan.abort)


def test_scan_stops_repeatedly():
    _, r = _run_scan()
    assert len(r["stops"]) >= 6, len(r["stops"])


def test_scan_takes_one_capture_per_stop():
    _, r = _run_scan()
    hold_caps = [c for c in r["captures"] if c[2] == "hold"]
    assert len(hold_caps) == len(r["stops"]), (len(hold_caps), len(r["stops"]))


def test_scan_keeps_the_eye_dark_between_captures():
    """The flash is on only for the measured frame, not held across the scan."""
    _, r = _run_scan()
    assert r["lit"] > 0, "never flashed at all"
    assert r["lit"] < r["frames"] * 0.25, (r["lit"], r["frames"])


def test_scan_lets_the_pupil_dilate_before_every_capture():
    _, r = _run_scan()
    assert r["dark_before"], "no captures fired"
    assert min(r["dark_before"]) >= cap.SCAN_DILATE_S, min(r["dark_before"])


def test_scan_waits_for_the_gaze_before_capturing():
    scan = cap.GazeScan((240, 320), (240.0, 320.0), 12.0, SHAPE,
                        cover_side="bottom", px_scale=PX_SCALE)
    scan.mode = scan.MODE_HOLD
    scan.hold_start = _NOW[0]
    scan.dark_since = _NOW[0] - 10.0               # long since dilated
    assert not scan.ready_to_capture(), "captured before the gaze arrived"
    _advance(cap.SCAN_GAZE_LATENCY_S + 0.01)
    assert scan.ready_to_capture()


def test_scan_waits_for_dilation_even_once_the_gaze_has_arrived():
    scan = cap.GazeScan((240, 320), (240.0, 320.0), 12.0, SHAPE,
                        cover_side="bottom", px_scale=PX_SCALE)
    scan.mode = scan.MODE_HOLD
    scan.hold_start = _NOW[0] - 10.0               # gaze long since settled
    scan.dark_since = _NOW[0]
    assert not scan.ready_to_capture(), "flashed a pupil that had not re-opened"
    _advance(cap.SCAN_DILATE_S + 0.01)
    assert scan.ready_to_capture()


def test_scan_banks_captures():
    _, r = _run_scan()
    assert r["saves"] >= 4, r["saves"]


def test_scan_save_bar_is_scaled_for_full_resolution_frames():
    """A reflex that would pass at display scale must not pass on a 4x frame."""
    scan = cap.GazeScan((240, 320), (240.0, 320.0), 12.0, SHAPE,
                        cover_side="bottom", px_scale=PX_SCALE)
    scan.mode = scan.MODE_HOLD
    m = _meas(cap.SESSION_SAVE_MIN_PX + 10)        # fine at display scale
    assert not scan.worth_saving(m), "unscaled bar let a small reflex through"
    m["red_px"] = int((cap.SESSION_SAVE_MIN_PX + 10) * PX_SCALE)
    assert scan.worth_saving(m)


def test_scan_finds_the_peak():
    scan, _ = _run_scan(peak_offset=-40.0)
    assert abs(scan.best_offset[1] - (-40.0)) < 40.0, scan.best_offset


def test_scan_leaves_a_stop_whose_captures_keep_failing():
    """Invalid captures must not park the scan on one point for ever."""
    scan, r = _run_scan(valid=False, limit_s=60.0)
    assert len(r["stops"]) >= 2, "stuck on the first stop"


# ── approach: overshoot / lost eye ──
# The operator pushes the scope one way on the strength of a red count that may
# never arrive.  Sail past the eye and there is nothing left to measure, so the
# stage has to say so instead of repeating "keep closing" until it times out.

def _run_approach(pupil_seq, side="bottom"):
    """Feed one ambient pupil result per lighting cycle.  Returns (state, warn, abort)."""
    cap._PUPIL_SIZE.reset()             # the red-eye bar is module state
    st = cap.ApproachState(side, (240.0, 320.0), 12.0)
    warned = aborted = None
    for i, p in enumerate(pupil_seq):
        _advance(1.1)                                  # one lighting cycle
        st.check_ambient_pupil(p, 0.0 if p is None else 0.6, SHAPE,
                               spoiled=(p is None))
        if st.recover and warned is None:
            warned = i + 1
        if st.abort and aborted is None:
            aborted = i + 1
            break
        st.submit(_meas(0, valid=False))
    return st, warned, aborted


def test_approach_warns_then_aborts_when_the_pupil_is_lost():
    st, warned, aborted = _run_approach([None] * 12)
    assert warned == cap.APPROACH_WARN_TRIES, warned
    assert aborted == cap.APPROACH_ABORT_TRIES, aborted
    assert st.abort, st.abort


def test_approach_detects_being_driven_past_the_cover_edge():
    # Bottom cover: a pupil pinned against the bottom of the frame has gone past
    # the zone where a reflex can be found, even though it is still detected.
    y = SHAPE[0] - cap.APPROACH_EDGE_FRAC * SHAPE[0] + 1
    st, warned, aborted = _run_approach([(240.0, y)] * 12, side="bottom")
    assert warned == cap.APPROACH_WARN_TRIES, warned
    assert aborted == cap.APPROACH_ABORT_TRIES, aborted


def test_approach_recovery_reverses_the_instruction():
    st, _, _ = _run_approach([None] * cap.APPROACH_WARN_TRIES)
    hint = st.camera_hint(SHAPE, (240, 320))
    vec = st.camera_vector(SHAPE, (240, 320))
    assert hint == guidance.back_off_hint("bottom"), hint
    assert ("down" in hint) == (vec[1] > 0), (hint, vec)


def test_approach_recovery_outranks_hold_steady():
    st, _, _ = _run_approach([None] * cap.APPROACH_WARN_TRIES)
    st.held = True                       # a reflex was found earlier
    assert "LOST" in st.camera_hint(SHAPE, (240, 320))


def test_approach_survives_ordinary_blinks():
    seq = [(240.0, 320.0), None, (240.0, 322.0), (240.0, 318.0), None,
           (240.0, 321.0), (240.0, 319.0), None, (240.0, 320.0)] * 3
    st, warned, aborted = _run_approach(seq)
    assert warned is None and aborted is None, (warned, aborted)


def test_approach_survives_consecutive_blinks():
    st, _, aborted = _run_approach([(240.0, 320.0), None, None, (240.0, 320.0)] * 4)
    assert aborted is None, "two blinks in a row are not an overshoot"


# ── approach: the camera instruction is an AMBIENT-only measurement ──
# Under flash-only light the eye is near-black and the re-detected pupil is the
# least trustworthy thing on the frame, so steering sideways by it chases
# detection noise.  The instruction is measured on the ambient frames and held
# through the dark and flash phases in between.

ANCHOR = (240, 320)


def _approach_at(x, y=320.0):
    """An approach whose last AMBIENT reading put the pupil at (x, y)."""
    cap._PUPIL_SIZE.reset()             # the red-eye bar is module state
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((float(x), float(y)), 0.6, SHAPE)
    return st


def test_approach_lateral_follows_the_ambient_pupil():
    assert "right" in _approach_at(380).camera_hint(SHAPE, ANCHOR)
    assert "left" in _approach_at(100).camera_hint(SHAPE, ANCHOR)


def test_approach_ignores_the_flash_lit_pupil_for_steering():
    st = _approach_at(380)                       # ambient says: move right
    before = st.camera_hint(SHAPE, ANCHOR)
    m = _meas(200)
    m["center"] = (100.0, 320.0)                 # flash re-detection says the other way
    st.submit(m)
    assert st.camera_hint(SHAPE, ANCHOR) == before, st.camera_hint(SHAPE, ANCHOR)
    assert "right" in before, before


def test_approach_still_moves_the_roi_with_the_flash_measurement():
    """Steering ignores it, but the red-eye ROI must still follow the eye."""
    st = _approach_at(380)
    m = _meas(200)
    m["center"] = (100.0, 320.0)
    st.submit(m)
    assert st.center == (100.0, 320.0), st.center
    assert st.amb_center == (380.0, 320.0), st.amb_center


def test_approach_instruction_holds_between_ambient_frames():
    st = _approach_at(380)
    first = st.camera_hint(SHAPE, ANCHOR)
    for _ in range(5):                           # dark + flash phases, no ambient
        _advance(0.2)
        assert st.camera_hint(SHAPE, ANCHOR) == first


def test_approach_instruction_updates_on_the_next_ambient_frame():
    st = _approach_at(380)
    assert "right" in st.camera_hint(SHAPE, ANCHOR)
    st.check_ambient_pupil((100.0, 320.0), 0.6, SHAPE)
    assert "left" in st.camera_hint(SHAPE, ANCHOR)


def test_approach_ignores_a_blink_spoiled_ambient_frame():
    st = _approach_at(380)
    st.check_ambient_pupil((100.0, 320.0), 0.6, SHAPE, spoiled=True)
    assert "right" in st.camera_hint(SHAPE, ANCHOR)


def test_approach_words_and_arrow_share_one_reading():
    st = _approach_at(380)
    hint = st.camera_hint(SHAPE, ANCHOR)
    vec = st.camera_vector(SHAPE, ANCHOR)
    assert ("right" in hint) == (vec[0] > 0), (hint, vec)
    st.check_ambient_pupil((100.0, 320.0), 0.6, SHAPE)
    hint, vec = st.camera_hint(SHAPE, ANCHOR), st.camera_vector(SHAPE, ANCHOR)
    assert ("left" in hint) == (vec[0] < 0), (hint, vec)


def test_approach_vertical_push_is_unaffected():
    """The vertical half never depended on a measurement: keep closing on the cover."""
    for side, word in (("bottom", "up"), ("top", "down")):
        st = cap.ApproachState(side, (240.0, 320.0), 12.0)
        st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
        assert word in st.camera_hint(SHAPE, ANCHOR), (side, st.camera_hint(SHAPE, ANCHOR))


# ── the red-eye bar tracks the pupil's size ──
# A fixed pixel count is unreachable on a dilated pupil and trivial on a small
# one, so the requirement is a third of the pupil's area.  Which pupil is the
# problem: the live fit usually reads too large (corneal reflections drag it off),
# so only the rare confident fits are believed, and the most RECENT of those wins.

def _sized(radius, conf):
    cap._PUPIL_SIZE.reset()
    cap.note_pupil_size(radius, conf)
    return cap.redeye_min_px()


def test_bar_starts_at_the_default_before_any_confident_fit():
    cap._PUPIL_SIZE.reset()
    assert cap.redeye_min_px() == cap.APPROACH_GOOD_PX == 650


def test_bar_is_a_third_of_the_pupil_area():
    r = 30.0
    assert _sized(r, 0.9) == int(math.pi * r * r / 3.0)


def test_bar_ignores_unconfident_fits():
    cap._PUPIL_SIZE.reset()
    for conf in (0.0, 0.25, 0.45, cap.REDEYE_SIZE_CONF - 0.01):
        cap.note_pupil_size(30.0, conf)
    assert cap.redeye_min_px() == cap.APPROACH_GOOD_PX


def test_bar_survives_poor_fits_after_a_confident_one():
    """The whole point: later low-confidence fits must not move the bar."""
    good = _sized(30.0, 0.9)
    for conf in (0.1, 0.3, 0.44, 0.0):
        cap.note_pupil_size(80.0, conf)          # wildly oversized, unconfident
    assert cap.redeye_min_px() == good


def test_bar_follows_the_most_recent_confident_fit():
    """Not the best-ever: the pupil dilates, so recency wins among confident fits."""
    _sized(40.0, 0.95)                            # very confident, big pupil
    cap.note_pupil_size(20.0, cap.REDEYE_SIZE_CONF)   # later, smaller, still confident
    assert cap.redeye_min_px() == int(math.pi * 20.0 ** 2 / 3.0)


def test_bar_is_clamped_to_a_reachable_range():
    assert _sized(200.0, 0.9) == cap.REDEYE_MIN_PX_CEIL
    assert _sized(2.0, 0.9) == cap.REDEYE_MIN_PX_FLOOR


def test_bar_resets_between_runs():
    _sized(40.0, 0.9)
    cap._PUPIL_SIZE.reset()
    assert cap.redeye_min_px() == cap.APPROACH_GOOD_PX


def test_approach_hands_over_on_the_dynamic_bar():
    """A small confident pupil lowers the bar, and the stage advances on it."""
    cap._PUPIL_SIZE.reset()
    cap.note_pupil_size(20.0, 0.9)               # bar = pi*400/3 = 418px
    need = cap.redeye_min_px()
    assert need < cap.APPROACH_GOOD_PX, need
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    for _ in range(cap.APPROACH_HOLD):
        st.submit(_meas(need + 1))
    assert st.done, st.status()


def test_approach_holds_out_for_a_big_pupil():
    """A wide confident pupil raises the bar, and a reflex that used to pass fails."""
    cap._PUPIL_SIZE.reset()
    cap.note_pupil_size(38.0, 0.9)               # bar = pi*1444/3 = 1512px
    assert cap.redeye_min_px() > cap.APPROACH_GOOD_PX
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    for _ in range(4):
        st.submit(_meas(cap.APPROACH_GOOD_PX + 10))
    assert not st.done, st.status()
    cap._PUPIL_SIZE.reset()


def test_approach_picks_up_a_confident_fit_mid_run():
    """The bar is re-read per measurement, not frozen when the stage started."""
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.submit(_meas(500))                        # below the 650 default
    assert not st.done and st.need_px == 650, st.need_px
    cap.note_pupil_size(20.0, 0.9)               # confident fit lands: bar drops
    for _ in range(cap.APPROACH_HOLD):
        st.submit(_meas(500))
    assert st.done, st.status()
    cap._PUPIL_SIZE.reset()


# ── the lighting cycle around the measured flash frame ──
# The measurement is only as good as the frame it runs on, and the frame is only
# trustworthy if the LED state it claims was actually the LED state at exposure.

def test_flash_stays_lit_for_the_whole_measurement():
    """The lighting cannot change out from under a red-eye pass in progress.

    `lighting()` is what drives the LEDs, and it reports "flash" for as long as
    the phase is FLASH — which only ends when the stage is handed a measurement.
    However long the CV takes, the ambient cannot come back mid-pass.
    """
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.to_flash()
    assert st.lighting() == "flash"
    _advance(5.0)                                # a very slow CV pass
    assert st.lighting() == "flash", "flash dropped while measuring"
    assert st.phase == st.PHASE_FLASH
    st.submit(_meas(300))                        # the pass finally returns
    assert st.phase == st.PHASE_SHOW


def test_measured_frame_is_held_up_before_the_next_cycle():
    """The operator gets a look at what was selected, not just the count."""
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.to_flash()
    st.submit(_meas(300))
    assert st.phase == st.PHASE_SHOW
    assert st.lighting() == "off", "the eye is still being flashed during the hold"
    assert not st.ready(), "hold ended immediately"
    _advance(cap.REDEYE_SHOW_S / 2.0)
    assert not st.ready(), "hold ended early"
    _advance(cap.REDEYE_SHOW_S)
    assert st.ready(), "hold never ended"
    st.to_ambient()                              # the loop starts the next cycle
    assert st.phase == st.PHASE_AMBIENT


def test_finishing_does_not_wait_on_the_hold():
    """The hold paces the cycle; handing over to the gaze scan must not wait on it."""
    cap._PUPIL_SIZE.reset()
    cap.note_pupil_size(20.0, 0.9)               # a low bar, cleared immediately
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.to_flash()
    need = cap.redeye_min_px()
    for _ in range(cap.APPROACH_HOLD - 1):
        assert not st.submit(_meas(need + 50))
    assert not st.done
    assert st.submit(_meas(need + 50)), "final measurement did not signal the loop"
    assert st.done
    cap._PUPIL_SIZE.reset()


def test_flash_is_measured_promptly_not_after_a_long_dwell():
    """The first flash frame carries the biggest reflex, so the wait is short."""
    assert cap.SWEEP_FLASH_S <= 0.1, cap.SWEEP_FLASH_S
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.to_flash()
    assert not st.ready()
    _advance(cap.SWEEP_FLASH_S + 0.01)
    assert st.ready(), "flash frame not available promptly after the LED came on"


def test_cycle_visits_every_phase_in_order():
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    seen = []
    st.to_ambient()
    for step in (st.to_dark, st.to_flash):
        seen.append((st.phase, st.lighting()))
        _advance(1.0)
        step()
    seen.append((st.phase, st.lighting()))
    assert seen == [("ambient", "ambient"), ("dark", "off"), ("flash", "flash")], seen


def test_cv_duration_is_recorded():
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.note_cv(0.21)
    st.note_cv(0.09)
    assert st.cv_s == 0.09 and st.cv_max == 0.21
    assert "cv=90ms" in st.status(), st.status()


# ── reversing when the REFLEX disappears (the pupil is still fine) ──
# Distinct from the lost-pupil overshoot: here the eye is tracked perfectly and it
# is the red reflex that has been driven past.

def _collapse(st, n=None):
    """Feed enough empty readings to trip the sustained-collapse rule."""
    for _ in range(n or cap.APPROACH_RELEASE_TRIES):
        st.submit(_meas(0))


def test_no_reversal_before_any_reflex_has_appeared():
    """At the start there is no evidence either way — keep closing."""
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    _collapse(st, cap.APPROACH_RELEASE_TRIES * 3)
    assert not st.reverse and st.flips == 0, (st.reverse, st.flips)
    assert "up" in st.camera_hint(SHAPE, ANCHOR)      # bottom cover closes upward


def test_reverses_once_a_seen_reflex_collapses():
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    st.submit(_meas(400))                             # a respectable reflex appears
    assert st.seen_reflex and not st.reverse
    _collapse(st)                                     # ...and then dies
    assert st.reverse and st.flips == 1, (st.reverse, st.flips)
    hint = st.camera_hint(SHAPE, ANCHOR)
    assert "down" in hint and "reflex lost" in hint, hint


def test_reversal_flips_the_arrow_too():
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    fwd = st.camera_vector(SHAPE, ANCHOR)
    st.submit(_meas(400))
    _collapse(st)
    rev = st.camera_vector(SHAPE, ANCHOR)
    assert fwd[1] * rev[1] < 0, (fwd, rev)


def test_reversal_takes_effect_without_waiting_for_an_ambient_frame():
    """A direction change is a decision, not a measurement — it must not be latched."""
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    st.camera_hint(SHAPE, ANCHOR)                     # latch the forward instruction
    st.submit(_meas(400))
    _collapse(st)                                     # no ambient frame in between
    assert "down" in st.camera_hint(SHAPE, ANCHOR)


def test_reversal_clears_when_the_reflex_comes_back():
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    st.submit(_meas(400))
    _collapse(st)
    assert st.reverse
    st.submit(_meas(400))                             # backing off found it again
    assert not st.reverse
    assert "up" in st.camera_hint(SHAPE, ANCHOR)      # resume closing


def test_reversal_flips_again_if_backing_off_does_not_help():
    """Walked back and forth across the peak, not sent one way for ever."""
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    st.submit(_meas(400))
    _collapse(st)
    assert st.reverse and st.flips == 1
    _collapse(st)
    assert not st.reverse and st.flips == 2, (st.reverse, st.flips)


def test_a_brief_dip_does_not_reverse():
    """One or two weak readings are normal flicker, not an overshoot."""
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    st.submit(_meas(400))
    _collapse(st, cap.APPROACH_RELEASE_TRIES - 1)
    assert not st.reverse, "reversed on a brief dip"


def test_lost_pupil_outranks_a_reflex_reversal():
    cap._PUPIL_SIZE.reset()
    st = cap.ApproachState("bottom", (240.0, 320.0), 12.0)
    st.check_ambient_pupil((240.0, 320.0), 0.6, SHAPE)
    st.submit(_meas(400))
    _collapse(st)
    assert st.reverse
    for _ in range(cap.APPROACH_WARN_TRIES):          # then the eye goes entirely
        st.check_ambient_pupil(None, 0.0, SHAPE, spoiled=True)
    assert "LOST THE PUPIL" in st.camera_hint(SHAPE, ANCHOR)


def test_approach_recovers_silently_when_the_pupil_returns():
    st, warned, aborted = _run_approach(
        [None] * cap.APPROACH_WARN_TRIES + [(240.0, 320.0)] * 4)
    assert warned == cap.APPROACH_WARN_TRIES and aborted is None
    assert not st.recover, "the warning must clear once the eye is back"


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
