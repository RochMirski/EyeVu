"""Operator alignment guidance — shared by the Pi (cap.py), receiver.py and pupillab.

Given the pupil's coarse centre in a frame, compute a vector from a target point
(camera centre for now; top-middle of the LED cover later) to the pupil and turn
it into a plain-language instruction ("Move up and left a little").  A stateful
``GuidanceTracker`` remembers the previous capture so it can add relative phrasing
("keep going" / "almost there" / "wrong way") — valid because the gaze is fixed
between captures and only the device moves.

numpy + cv2 only (no cap.py / pupillab imports), so it is safe to import anywhere,
including next to cap.py on the Pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Defaults (fractions of the smaller frame dimension unless noted).
DEADZONE_FRAC = 0.06     # within this of the target => "centred"
SMALL_FRAC = 0.10        # |offset| below this is "a little"
LARGE_FRAC = 0.28        # |offset| above this is "a lot"
IMPROVE_FRAC = 0.02      # min change in distance to call it better/worse
VALID_FRAC = 0.12        # within this of the target (cover edge) => "valid region":
                         # the pupil is at the alignment point.  Once it has been
                         # seen here and is then lost, the eye is either aligned
                         # (fundus filling the pupil) or the cover has moved over it.

# ── Session stages (cap.py's live staged guidance) ──
# They live here (not in cap.py) so the names and the instruction wording stay with
# the rest of the guidance vocabulary; the LED/camera driving stays on the Pi side.
#
#   find_cover — ambient on: locate the dark LED-cover outline live (no calibration).
#                Skipped unless cap.USE_LIVE_COVER is on.
#   centre     — ambient on: bring the pupil into the deadzone box on the target
#   approach   — the OPERATOR moves the scope vertically, closing the gap between the
#                LED cover and the pupil, until a decent chunk of red reflex appears
#   target     — the FIXATION TARGET is walked around the screen (hill-climb, both
#                axes); the patient keeps looking at it, so their gaze follows and the
#                red reflex is maximised without anyone estimating a direction
#   capture    — dark dilation, then the dark+flash pair
#
# Throughout, the patient is told exactly one thing — INSTRUCTION_LOOK — and the eye
# is steered by MOVING THE TARGET, never by asking for "up a bit" in words.
STAGE_FIND_COVER = "find_cover"
STAGE_CENTRE = "centre"
STAGE_APPROACH = "approach"
STAGE_TARGET = "target"
STAGE_CAPTURE = "capture"
STAGES = (STAGE_FIND_COVER, STAGE_CENTRE, STAGE_APPROACH, STAGE_TARGET,
          STAGE_CAPTURE)

# The only thing the patient is ever asked to do.
INSTRUCTION_LOOK = "Look at the specified point."

# During the staged session the instruction is rendered small and just off the
# middle of the frame, rather than large in the top-left corner: whoever is
# looking at the centre of the screen can then read it without looking away.
SIDE_FONT_SCALE = 0.42
SIDE_GAP_FRAC = 0.09     # gap from the frame centre, as a fraction of min(h, w).
                         # Wide enough to clear the fixation rings and the
                         # "pupil here" marker that sit near the middle.
SIDE_LINE_PX = 16        # line spacing inside the side block

# Pupil within this of where it should be (fraction of min(h, w)) => hold steady.
CAMERA_DEAD_FRAC = 0.035
# ...but once it HAS settled, this much larger drift is needed before nagging the
# operator again.  Without the hysteresis the hint flickers between "hold steady"
# and a direction on every small tremor, which reads as noise and invites the
# operator to chase it — the one thing that must not happen while the gaze is
# being walked around.
CAMERA_RELEASE_FRAC = 0.10


def _close_word(cover_side):
    """Camera direction that closes the gap between the LED cover and the pupil.

    THE COVER IS BOLTED TO THE SCOPE.  It does not move in the frame — it sits at
    the same edge in every shot, which is exactly why a single stored calibration
    mask stays valid.  So closing the gap means walking the PUPIL toward that
    edge, and image content travels OPPOSITE to camera motion:

        cover at the TOP    -> pupil must move UP the frame   -> camera moves DOWN
        cover at the BOTTOM -> pupil must move DOWN the frame -> camera moves UP

    Getting this backwards drives the cover away from the pupil, which is the one
    thing the approach stage exists to prevent.  See test_guidance.py.
    """
    return "down" if str(cover_side).lower().startswith("t") else "up"


def _back_off_word(cover_side):
    """Camera direction that UNDOES the approach — the exact reverse of closing."""
    return "up" if _close_word(cover_side) == "down" else "down"


def _vertical_word(cover_side, reverse=False):
    """The approach's vertical direction: closing on the cover, or backing off it.

    The approach is not a one-way push.  The red reflex has a sweet spot in the
    cover/pupil gap — too far away and it has not appeared, too far past and it is
    gone again — and the pupil stays perfectly visible on both sides of it, so
    "keep closing" can easily be an instruction to walk further away from the only
    place the reflex exists.  Reversing has to be sayable.

    Both the words and the arrow come through here, so they cannot disagree.
    """
    return _back_off_word(cover_side) if reverse else _close_word(cover_side)


def camera_hint(pupil_center, anchor, frame_shape, close_to=None, settled=False,
                reverse=False):
    """What the OPERATOR should do with the scope, in one short line.

    Same sign convention as the movement instructions above: the word names the
    direction the DEVICE moves to bring the pupil back to `anchor`.

    `close_to` ("top"/"bottom") switches the vertical half into approach mode —
    keep closing on that edge rather than holding a position — while the
    horizontal half still corrects sideways drift.  `reverse` flips that vertical
    half to back OFF the cover instead, for when the reflex has been overshot; the
    lateral correction is unaffected either way.

    `settled` applies the hysteresis: once the operator has been told to hold
    steady, only a LARGE drift earns another instruction.
    """
    if pupil_center is None:
        return "Camera: searching for the pupil"
    h, w = frame_shape[:2]
    dead = (CAMERA_RELEASE_FRAC if settled else CAMERA_DEAD_FRAC) * float(min(h, w))
    dx = float(pupil_center[0] - anchor[0])
    dy = float(pupil_center[1] - anchor[1])
    parts = []
    if close_to:
        parts.append(_vertical_word(close_to, reverse))
    else:
        v = _axis_word(dy, "down", "up", dead)
        if v:
            parts.append(v)
    hz = _axis_word(dx, "right", "left", dead)
    if hz:
        parts.append(hz)
    if not parts:
        return "Camera: hold steady"
    return "Camera: move " + " and ".join(parts)


# How far the camera-move arrow reaches, as a fraction of min(h, w).  It shows a
# DIRECTION and a destination for the camera centre, not a calibrated distance —
# the mapping from scope travel to image travel is not known here.
CAMERA_ARROW_FRAC = 0.16
# Lateral drift of this much (fraction of min(h, w)) is treated as "as urgent as"
# the approach's vertical push, so the two combine into a sensible diagonal.
CAMERA_ARROW_REF_FRAC = 0.15


def camera_arrow(pupil_center, anchor, frame_shape, close_to=None, settled=False,
                 reverse=False):
    """(dx, dy) px the camera centre should move — the arrow drawn from centre.

    Matches `camera_hint` exactly, so the words and the arrow can never disagree:
    the lateral component corrects drift off `anchor`, and `close_to` pins the
    vertical to keep closing on that frame edge.  Returns None when there is
    nothing to do (hold steady).

    In approach mode the two components come from different worlds — the vertical
    is an open-ended "keep going", the horizontal a measured pixel error — so the
    horizontal is first normalised against CAMERA_ARROW_REF_FRAC.  Without that
    the pixel-valued term swamps the unit-valued one and the arrow points almost
    due sideways however much the pupil still has to travel vertically.
    """
    if pupil_center is None:
        return None
    h, w = frame_shape[:2]
    scale = float(min(h, w))
    dead = (CAMERA_RELEASE_FRAC if settled else CAMERA_DEAD_FRAC) * scale
    dx = float(pupil_center[0] - anchor[0])
    dy = float(pupil_center[1] - anchor[1])
    if close_to:
        # Screen y grows downward, so "down" is +1.  Derived from the same
        # _vertical_word as the text, so arrow and words cannot drift apart —
        # including when the approach reverses to back off an overshot reflex.
        vy = 1.0 if _vertical_word(close_to, reverse) == "down" else -1.0
        ref = CAMERA_ARROW_REF_FRAC * scale
        vx = 0.0 if abs(dx) <= dead else max(-1.0, min(1.0, dx / ref))
    else:
        vx = dx if abs(dx) > dead else 0.0
        vy = dy if abs(dy) > dead else 0.0
    if vx == 0.0 and vy == 0.0:
        return None
    n = float(np.hypot(vx, vy))
    reach = CAMERA_ARROW_FRAC * scale
    return (vx / n * reach, vy / n * reach)


def back_off_hint(cover_side):
    """Operator instruction for an overshot approach.

    Says what went wrong as well as what to do: the operator has been pushing in
    one direction on the strength of a red count that never arrived, so an
    unexplained reversal reads as the guidance dithering.
    """
    return f"Camera: LOST THE PUPIL - move {_back_off_word(cover_side)}, gone too far"


def back_off_vector(cover_side, frame_shape):
    """The back-off instruction as an arrow, matching `back_off_hint` exactly."""
    h, w = frame_shape[:2]
    reach = CAMERA_ARROW_FRAC * float(min(h, w))
    return (0.0, reach if _back_off_word(cover_side) == "down" else -reach)


def cover_edge_word(cover_side):
    """Which frame edge the LED cover intrudes from: "top" or "bottom"."""
    return "bottom" if str(cover_side).lower().startswith("b") else "top"


# How far from the cover's frame edge the pupil is aimed during the approach, as a
# fraction of the frame height.  Only used when there is no cover mask to measure
# against — with one, the mask's own inner edge is the destination.
APPROACH_DEST_FRAC = 0.20


def approach_destination(frame_shape, cover_side, x, cover_mask=None):
    """Where the PUPIL should end up during the approach — near the cover edge.

    This is NOT the fixation target: the patient keeps looking at the same point
    while the operator walks the pupil toward the cover, so the two markers pull
    apart during this stage and have to be shown separately.
    """
    h, w = frame_shape[:2]
    if cover_mask is not None and cv2.countNonZero(cover_mask):
        return target_point(frame_shape, cover_mask, mode="cover_top_mid")
    dy = int(APPROACH_DEST_FRAC * h)
    y = dy if cover_edge_word(cover_side) == "top" else h - dy
    return (int(x), int(y))


def approach_hint(cover_side, red_px, need_px):
    """Operator sub-line for the approach stage (scope moved by hand).

    Phrased against what is visible in the feed — the pupil's position relative to
    the cover edge — rather than a scope direction, because which way to push the
    scope depends on the rig's handedness but the picture never lies.
    """
    return (f"scope: pupil -> {cover_edge_word(cover_side)} edge | "
            f"red {int(red_px)}/{int(need_px)}px")


def target_point(frame_shape, cover_mask=None, mode="centre"):
    """Return the target (tx, ty) the pupil should be driven to.

    mode="centre"        -> image centre (current default).
    mode="cover_top_mid" -> middle of the LED cover's INNER edge (the edge facing
                            the pupil): the bottom-middle of a cover in the top
                            half, the top-middle of a cover in the bottom half, so
                            the pupil is driven just past it.  The side is detected
                            from the mask.  Falls back to the image centre if no
                            cover mask is supplied.

    `tx` is the cover's horizontal centre of mass and `ty` is its inner edge
    measured AT that column (median over a narrow window, so one ragged column
    cannot set it).  Taking instead the mask's single deepest row — which is what
    this did originally — only coincides with "the middle" for a compact blob: on
    the full-width band that the live, calibration-free detector produces, the
    deepest point is a side lobe where the brow shadow dips, and the target landed
    out at the frame edge, below the pupil instead of above it.
    """
    h, w = frame_shape[:2]
    if mode == "cover_top_mid" and cover_mask is not None and cv2.countNonZero(cover_mask):
        m = cover_mask > 0
        ys, xs = np.where(m)
        top = int(ys.min()); bot = int(ys.max())
        top_cover = (top + bot) * 0.5 < h * 0.5   # cover in the TOP half of the frame
        tx = int(round(float(xs.mean())))
        tx = max(0, min(w - 1, tx))

        half = max(1, int(round(0.05 * w)))
        x0, x1 = max(0, tx - half), min(w, tx + half + 1)
        sub = m[:, x0:x1]
        has = sub.any(axis=0)
        if not has.any():                         # window missed the cover entirely
            return (tx, bot if top_cover else top)
        sub = sub[:, has]
        if top_cover:                             # inner edge = the LOWEST cover row
            edge_per_col = sub.shape[0] - 1 - np.argmax(sub[::-1, :], axis=0)
        else:                                     # inner edge = the HIGHEST cover row
            edge_per_col = np.argmax(sub, axis=0)
        return (tx, int(round(float(np.median(edge_per_col)))))
    return (w // 2, h // 2)


@dataclass
class Guidance:
    state: str = "searching"          # "searching" | "move" | "centred" | "reached"
                                      # | "sweep" (set by the live gaze sweep, which
                                      #   builds its own Guidance for display)
    instruction: str = "Searching for pupil..."
    vector: Optional[tuple] = None    # (dx, dy) = pupil - target, px
    distance: float = 0.0             # |vector|, px
    confidence: float = 0.0
    source: str = ""                  # which detector found the pupil
    in_valid_region: bool = False     # pupil currently detected near the cover edge
    reached_valid: bool = False       # pupil has EVER reached the valid region this session
    endpoint: bool = False            # was in the valid region, now lost -> aligned OR
                                      # blocked over (check for fundus / capture)
    hint: str = ""                    # operator camera movement, shown under the
                                      # patient instruction in the side block
    hint_vector: Optional[tuple] = None  # (dx, dy) px the CAMERA CENTRE should move;
                                      # drawn as an arrow from the frame centre
    readout: str = ""                 # the measurement being worked to (red pixels
                                      # against the requirement), shown beside the
                                      # fixation target with the instructions —
                                      # the operator is looking THERE, not at the
                                      # status line in the corner
    readout_ok: bool = False          # requirement met -> draw the readout green


def _axis_word(value, positive, negative, dead):
    if value < -dead:
        return negative
    if value > dead:
        return positive
    return None


def _magnitude_word(dist, scale):
    if dist >= LARGE_FRAC * scale:
        return "a lot"
    if dist <= SMALL_FRAC * scale:
        return "a little"
    return ""


class GuidanceTracker:
    """Stateful guidance across the captures of one alignment session.

    `dead_x_frac` / `dead_y_frac` optionally replace the isotropic radial deadzone
    with a per-axis box, for stages that need one axis far tighter than the other
    (the live staged session wants near-perfect alignment on the axis the gaze will
    later sweep along).  Both must be given for the box to apply; left as None the
    behaviour is exactly the radial `deadzone_frac` test, so the receiver and
    pupillab replays are unchanged.
    """

    def __init__(self, deadzone_frac=DEADZONE_FRAC, valid_frac=VALID_FRAC,
                 dead_x_frac=None, dead_y_frac=None):
        self.deadzone_frac = deadzone_frac
        self.valid_frac = valid_frac
        self.dead_x_frac = dead_x_frac
        self.dead_y_frac = dead_y_frac
        self._last_dist = None
        self._last_vec = None
        self._was_in_valid = False    # pupil has reached the valid region this session

    def reset(self):
        self._last_dist = None
        self._last_vec = None
        self._was_in_valid = False

    def deadzone_box(self, frame_shape):
        """(half_w, half_h) of the per-axis deadzone in px, or None if radial."""
        if self.dead_x_frac is None or self.dead_y_frac is None:
            return None
        scale = float(min(frame_shape[0], frame_shape[1]))
        return (self.dead_x_frac * scale, self.dead_y_frac * scale)

    def _centred(self, dx, dy, dead, scale):
        """Is the pupil within the deadzone — per-axis box if set, else radial."""
        if self.dead_x_frac is not None and self.dead_y_frac is not None:
            return (abs(dx) <= self.dead_x_frac * scale
                    and abs(dy) <= self.dead_y_frac * scale)
        return float(np.hypot(dx, dy)) <= dead

    def update(self, pupil_center, frame_shape, target, confidence=1.0,
               source="") -> Guidance:
        """Compute guidance for this capture and fold in the previous one.

        `pupil_center` is (x, y) or None.  `target` is (tx, ty).  Returns a
        ``Guidance``; also stores state for the next call's relative phrasing and
        tracks whether the pupil has reached the valid region (near the cover
        edge) so a subsequent loss can be reported as aligned/blocked.
        """
        h, w = frame_shape[:2]
        scale = float(min(h, w))
        dead = self.deadzone_frac * scale
        valid = self.valid_frac * scale

        if pupil_center is None:
            # Pupil lost.  If it had reached the valid region, losing it means the
            # eye is either aligned (fundus filling the pupil) or the cover has
            # moved over it -> endpoint reached.
            if self._was_in_valid:
                return Guidance(
                    state="reached",
                    instruction="Pupil reached the cover edge then lost - aligned or "
                                "covered. Check for fundus / capture.",
                    confidence=confidence, source=source,
                    reached_valid=True, endpoint=True)
            return Guidance(state="searching",
                            instruction="Searching for pupil - reposition the device.",
                            confidence=confidence, source=source)

        dx = float(pupil_center[0] - target[0])
        dy = float(pupil_center[1] - target[1])
        dist = float(np.hypot(dx, dy))
        in_valid = dist <= valid
        if in_valid:
            self._was_in_valid = True

        if self._centred(dx, dy, dead, scale):
            self._last_dist, self._last_vec = dist, (dx, dy)
            return Guidance(state="centred", instruction="Centred - hold steady.",
                            vector=(dx, dy), distance=dist, confidence=confidence,
                            source=source, in_valid_region=in_valid,
                            reached_valid=self._was_in_valid)

        # Name a direction on each axis well before the centred deadzone, so an
        # off-centre pupil always gets an actionable "up/down/left/right".  With a
        # per-axis box the two axes get their own thresholds, so the tight axis
        # keeps naming a direction long after the loose one has gone quiet.
        box = self.deadzone_box(frame_shape)
        if box is None:
            dead_x = dead_y = max(8.0, 0.35 * dead)
        else:
            dead_x = max(4.0, 0.35 * box[0])
            dead_y = max(4.0, 0.35 * box[1])
        vert = _axis_word(dy, "down", "up", dead_y)
        horiz = _axis_word(dx, "right", "left", dead_x)
        parts = [p for p in (vert, horiz) if p]
        direction = " and ".join(parts) if parts else "to centre"
        mag = _magnitude_word(dist, scale)
        instruction = f"Move {direction}" + (f" {mag}" if mag else "")

        # Relative phrasing vs the previous capture (fixed gaze, device moved).
        if self._last_dist is not None:
            improved = self._last_dist - dist
            thresh = IMPROVE_FRAC * scale
            if improved > thresh:
                instruction += " - keep going" if dist > 2 * dead else " - almost there"
            elif improved < -thresh:
                instruction += " - wrong way, reverse"

        self._last_dist, self._last_vec = dist, (dx, dy)
        return Guidance(state="move", instruction=instruction, vector=(dx, dy),
                        distance=dist, confidence=confidence, source=source,
                        in_valid_region=in_valid, reached_valid=self._was_in_valid)


def annotate(frame_bgr, guidance: Guidance, target, pupil_center=None,
             deadzone_box=None, extra="", cover_mask=None, fixation=False,
             hold_point=None):
    """Draw the target, pupil, guidance arrow and instruction text (in place-ish).

    `deadzone_box` = (half_w, half_h) px draws the per-axis centred box instead of
    leaving the tolerance implicit.  `extra` is appended to the status sub-line
    (the live session puts the stage and red fraction there).  `cover_mask`, when
    given, is outlined so the operator can see what was detected as the LED cover.

    Returns the annotated BGR image (a copy)."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    tx, ty = int(target[0]), int(target[1])
    if cover_mask is not None and cover_mask.shape[:2] == (h, w):
        cnts, _ = cv2.findContours(cover_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 0, 255), 2, cv2.LINE_AA)
    # Valid region (the alignment zone around the cover-edge target).
    valid_r = int(VALID_FRAC * min(h, w))
    cv2.circle(img, (tx, ty), valid_r, (0, 200, 200), 1, cv2.LINE_AA)
    if deadzone_box is not None:
        bw, bh = int(round(deadzone_box[0])), int(round(deadzone_box[1]))
        cv2.rectangle(img, (tx - bw, ty - bh), (tx + bw, ty + bh),
                      (0, 255, 255), 1, cv2.LINE_AA)
    # Target crosshair (cyan).
    cv2.drawMarker(img, (tx, ty), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(img, (tx, ty), 6, (255, 255, 0), 1)
    if fixation:
        # The patient is looking AT this point and it moves during the target
        # search, so make it unmistakable: concentric rings and a solid centre.
        # NOTE this is a GAZE target, not a destination for the pupil — the two
        # mean opposite things, hence the label.
        cv2.circle(img, (tx, ty), 14, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 20, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 3, (255, 255, 255), -1, cv2.LINE_AA)
        for col, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, "LOOK", (tx - 16, ty - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, thick, cv2.LINE_AA)

    # Where the PUPIL should end up — a different thing entirely from the gaze
    # target above, and in the approach stage a different PLACE too, so it gets
    # its own marker, colour and label.  Magenta = pupil, cyan/white = gaze.
    if hold_point is not None:
        hx, hy = int(hold_point[0]), int(hold_point[1])
        # When the two separate, join them with a faint line so it is obvious
        # they are two distinct instructions rather than one confused one.
        if abs(hx - tx) > 6 or abs(hy - ty) > 6:
            cv2.line(img, (tx, ty), (hx, hy), (120, 60, 120), 1, cv2.LINE_AA)
        cv2.circle(img, (hx, hy), 11, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (hx, hy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
        for col, thick in (((0, 0, 0), 3), ((255, 0, 255), 1)):
            cv2.putText(img, "pupil here", (hx - 28, hy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, thick, cv2.LINE_AA)

    # Camera-move arrow: from the current camera centre to where it should end up.
    if guidance.hint_vector is not None:
        ccx, ccy = w // 2, h // 2
        ex = int(round(ccx + guidance.hint_vector[0]))
        ey = int(round(ccy + guidance.hint_vector[1]))
        cv2.arrowedLine(img, (ccx, ccy), (ex, ey), (0, 0, 0), 4, cv2.LINE_AA,
                        tipLength=0.28)
        cv2.arrowedLine(img, (ccx, ccy), (ex, ey), (0, 255, 255), 2, cv2.LINE_AA,
                        tipLength=0.28)
        cv2.circle(img, (ex, ey), 5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (ccx, ccy), (0, 255, 255), cv2.MARKER_SQUARE, 7, 1)

    colour = {"centred": (0, 255, 0), "move": (0, 165, 255),
              "searching": (0, 0, 255), "reached": (0, 255, 0),
              "approach": (255, 200, 0), "target": (255, 200, 0)}.get(
                  guidance.state, (0, 165, 255))

    if pupil_center is not None:
        px, py = int(pupil_center[0]), int(pupil_center[1])
        # Fill the pupil dot green when it's inside the valid region.
        dot = (0, 255, 0) if guidance.in_valid_region else colour
        cv2.circle(img, (px, py), 5, dot, -1)
        if guidance.state == "move":
            cv2.arrowedLine(img, (tx, ty), (px, py), colour, 2, tipLength=0.2)

    if fixation:
        # Whoever is looking at the middle of the screen gets both lines small and
        # just off-centre, where they sit in peripheral vision — the patient's
        # instruction and, under it, the camera move the operator needs to make.
        # Placed to the right unless that would overflow, then flipped to the left.
        # Patient instruction, then the operator's camera move, then the number
        # they are working to.  All three sit by the fixation point rather than in
        # the corner: that is where both pairs of eyes already are.
        lines = [(guidance.instruction, (255, 255, 255))]
        if guidance.hint:
            lines.append((guidance.hint, (0, 255, 255)))          # amber
        if guidance.readout:
            lines.append((guidance.readout,
                          (0, 255, 0) if guidance.readout_ok else (0, 255, 255)))
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                                  SIDE_FONT_SCALE, 1)[0][0] for t, _ in lines]
        block_w = max(widths)
        gap = int(SIDE_GAP_FRAC * min(h, w))
        ix = w // 2 + gap
        if ix + block_w > w - 6:
            ix = w // 2 - gap - block_w
        ix = max(6, min(ix, w - block_w - 6))
        iy = h // 2 - (len(lines) - 1) * SIDE_LINE_PX // 2 + 5
        for n, (text, tone) in enumerate(lines):
            y = iy + n * SIDE_LINE_PX
            for col, thick in (((0, 0, 0), 3), (tone, 1)):
                cv2.putText(img, text, (ix, y), cv2.FONT_HERSHEY_SIMPLEX,
                            SIDE_FONT_SCALE, col, thick, cv2.LINE_AA)
    else:
        cv2.putText(img, guidance.instruction, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, guidance.instruction, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 1, cv2.LINE_AA)
    sub = f"[{guidance.state}]"
    if guidance.source:
        sub += f" {guidance.source}"
    if guidance.distance:
        sub += f"  off={guidance.distance:.0f}px"
    if guidance.in_valid_region:
        sub += "  IN-VALID"
    elif guidance.reached_valid:
        sub += "  was-valid"
    if guidance.endpoint:
        sub += "  ENDPOINT: aligned/blocked"
    if extra:
        sub += f"  {extra}"
    cv2.putText(img, sub, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, sub, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return img
