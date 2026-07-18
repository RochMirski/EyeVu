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


def target_point(frame_shape, cover_mask=None, mode="centre"):
    """Return the target (tx, ty) the pupil should be driven to.

    mode="centre"        -> image centre (current default).
    mode="cover_top_mid" -> middle of the LED cover's INNER edge (the edge facing
                            the pupil): the bottom-middle of a cover in the top
                            half, the top-middle of a cover in the bottom half, so
                            the pupil is driven just past it.  The side is detected
                            from the mask.  Falls back to the image centre if no
                            cover mask is supplied.
    """
    h, w = frame_shape[:2]
    if mode == "cover_top_mid" and cover_mask is not None and cv2.countNonZero(cover_mask):
        ys, xs = np.where(cover_mask > 0)
        top = int(ys.min()); bot = int(ys.max())
        if (top + bot) * 0.5 < h * 0.5:        # cover in TOP half -> inner edge is its bottom
            edge = bot
            band = ys >= bot - 0.10 * (bot - top + 1)
        else:                                  # cover in BOTTOM half -> inner edge is its top
            edge = top
            band = ys <= top + 0.10 * (bot - top + 1)
        tx = int(round(xs[band].mean())) if np.any(band) else int(round(xs.mean()))
        return (tx, edge)
    return (w // 2, h // 2)


@dataclass
class Guidance:
    state: str = "searching"          # "searching" | "move" | "centred" | "reached"
    instruction: str = "Searching for pupil..."
    vector: Optional[tuple] = None    # (dx, dy) = pupil - target, px
    distance: float = 0.0             # |vector|, px
    confidence: float = 0.0
    source: str = ""                  # which detector found the pupil
    in_valid_region: bool = False     # pupil currently detected near the cover edge
    reached_valid: bool = False       # pupil has EVER reached the valid region this session
    endpoint: bool = False            # was in the valid region, now lost -> aligned OR
                                      # blocked over (check for fundus / capture)


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
    """Stateful guidance across the captures of one alignment session."""

    def __init__(self, deadzone_frac=DEADZONE_FRAC, valid_frac=VALID_FRAC):
        self.deadzone_frac = deadzone_frac
        self.valid_frac = valid_frac
        self._last_dist = None
        self._last_vec = None
        self._was_in_valid = False    # pupil has reached the valid region this session

    def reset(self):
        self._last_dist = None
        self._last_vec = None
        self._was_in_valid = False

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

        if dist <= dead:
            self._last_dist, self._last_vec = dist, (dx, dy)
            return Guidance(state="centred", instruction="Centred - hold steady.",
                            vector=(dx, dy), distance=dist, confidence=confidence,
                            source=source, in_valid_region=in_valid,
                            reached_valid=self._was_in_valid)

        # Name a direction on each axis well before the centred deadzone, so an
        # off-centre pupil always gets an actionable "up/down/left/right".
        axis_dead = max(8.0, 0.35 * dead)
        vert = _axis_word(dy, "down", "up", axis_dead)
        horiz = _axis_word(dx, "right", "left", axis_dead)
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


def annotate(frame_bgr, guidance: Guidance, target, pupil_center=None):
    """Draw the target, pupil, guidance arrow and instruction text (in place-ish).

    Returns the annotated BGR image (a copy)."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    tx, ty = int(target[0]), int(target[1])
    # Valid region (the alignment zone around the cover-edge target).
    valid_r = int(VALID_FRAC * min(h, w))
    cv2.circle(img, (tx, ty), valid_r, (0, 200, 200), 1, cv2.LINE_AA)
    # Target crosshair (cyan).
    cv2.drawMarker(img, (tx, ty), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(img, (tx, ty), 6, (255, 255, 0), 1)

    colour = {"centred": (0, 255, 0), "move": (0, 165, 255),
              "searching": (0, 0, 255), "reached": (0, 255, 0)}.get(
                  guidance.state, (0, 165, 255))

    if pupil_center is not None:
        px, py = int(pupil_center[0]), int(pupil_center[1])
        # Fill the pupil dot green when it's inside the valid region.
        dot = (0, 255, 0) if guidance.in_valid_region else colour
        cv2.circle(img, (px, py), 5, dot, -1)
        if guidance.state == "move":
            cv2.arrowedLine(img, (tx, ty), (px, py), colour, 2, tipLength=0.2)

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
    cv2.putText(img, sub, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return img
