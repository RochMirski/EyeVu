"""Core data structures and the detector interface for pupillab.

Three things live here:

* ``DetectionContext`` — the shared, preprocessed view of one capture (built once
  by ``context.build_context`` and handed to every detector).
* ``DetectionResult`` — what a detector returns: the fitted pupil (and optional
  iris), a confidence, drawing info, and the list of labelled debug ``stages`` it
  produced.  The ``stages`` list is the montage-extension point — whatever a
  module appends shows up in the montage and the dashboard with no other changes.
* ``PupilDetector`` — the ABC every module subclasses.  ``params_spec`` declares
  tunable knobs so the dashboard can auto-generate sliders.

Ellipses everywhere use OpenCV's convention: ``((cx, cy), (major_axis,
minor_axis), angle_deg)`` — the same tuple ``cv2.ellipse`` / ``cap._draw_overlays``
expect, so a circle of radius ``r`` is ``((cx, cy), (2*r, 2*r), 0.0)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# OpenCV-style ellipse: ((cx, cy), (major, minor), angle_deg)
Ellipse = tuple


@dataclass
class ParamSpec:
    """One tunable parameter, used to auto-build a dashboard slider.

    ``kind`` is "float" or "int"; "int" sliders/values are rounded.  ``help`` is
    shown as slider help text.
    """

    name: str
    default: float
    min: float
    max: float
    step: float = 1.0
    kind: str = "float"
    help: str = ""

    def coerce(self, value: float) -> float:
        v = max(self.min, min(self.max, value))
        return int(round(v)) if self.kind == "int" else float(v)


@dataclass
class DetectionContext:
    """Shared, preprocessed view of a single capture (built once per capture).

    All images are BGR uint8 in the **display (rotated) orientation**, matching
    what ``cap.detect_and_annotate`` works in, so overlays line up with the
    rotated flash photo.  Detectors should treat this as read-only.
    """

    ambient: np.ndarray            # BGR ambient frame (detector input)
    flash: np.ndarray              # BGR flash frame (overlays drawn here)
    both: Optional[np.ndarray]     # BGR flash+ambient combined frame, or None
    gray: np.ndarray               # ambient grayscale
    green: np.ndarray              # ambient green channel (cleanest under violet)
    cover_mask: Optional[np.ndarray]   # calibrated LED-cover mask, or None
    anchor: Optional[tuple]        # corneal-reflex anchor (cx, cy, r), or None
    reflex_mask: Optional[np.ndarray]  # reflex blob mask, or None
    rmin: float                    # min plausible pupil radius (px)
    rmax: float                    # max plausible pupil radius (px)
    meta: dict = field(default_factory=dict)
    # Search ROI: pupil is assumed to lie in a box just above the LED cover.
    # `roi` is (x0, y0, x1, y1) in display-orientation pixels, or None (no
    # restriction).  `search_mask` is a uint8 mask (255 = searchable) = the box
    # with the known cover region subtracted; detectors restrict to it.
    roi: "Optional[tuple]" = None
    search_mask: "Optional[np.ndarray]" = None
    # `box_mask` is the ROI box alone (255 inside), *without* the cover removed.
    # `ambient_filled` / `green_filled` are the images with the LED cover region
    # inpainted + smoothed, so detectors get a clean fill instead of a black hole.
    box_mask: "Optional[np.ndarray]" = None
    ambient_filled: "Optional[np.ndarray]" = None
    green_filled: "Optional[np.ndarray]" = None

    @property
    def shape(self) -> tuple:
        return self.gray.shape[:2]

    def in_roi(self, x, y) -> bool:
        """True if (x, y) is inside the search mask (or if no ROI is set)."""
        if self.search_mask is None:
            return True
        h, w = self.search_mask.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        return bool(self.search_mask[yi, xi])

    def restrict(self, img):
        """Return a copy of `img` with everything outside the search mask zeroed.

        No-op (returns `img` unchanged) when no ROI is set."""
        if self.search_mask is None:
            return img
        out = img.copy()
        out[self.search_mask == 0] = 0
        return out

    def restrict_box(self, img):
        """Zero everything outside the ROI box (but keep the filled cover inside).

        Unlike `restrict`, this does NOT subtract the cover, so a cover that was
        inpainted/smoothed stays intact.  No-op when no ROI is set."""
        if self.box_mask is None:
            return img
        out = img.copy()
        out[self.box_mask == 0] = 0
        return out


@dataclass
class DetectionResult:
    """What a detector returns for one capture.

    ``pupil`` / ``iris`` are OpenCV ellipses or None.  ``stages`` is a list of
    ``(label, bgr_image)`` debug images — append freely; they flow straight into
    the montage and dashboard.  ``ok`` is False for a graceful non-result (e.g. a
    missing optional dependency) and carries an explanation in ``notes``.
    """

    detector: str
    pupil: Optional[Ellipse] = None
    iris: Optional[Ellipse] = None
    confidence: float = 0.0
    label: str = ""
    color: tuple = (0, 255, 0)          # BGR overlay colour
    stages: list = field(default_factory=list)   # list[(name, bgr_image)]
    ok: bool = True
    notes: str = ""
    elapsed_ms: float = 0.0

    def add_stage(self, name: str, img: np.ndarray) -> None:
        """Record a debug image (grayscale promoted to BGR for uniform montaging)."""
        import cv2
        if img is None:
            return
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        self.stages.append((name, out))

    def overlay(self) -> Optional[dict]:
        """The pupil as a ``cap._draw_overlays`` dict, or None if no pupil."""
        if self.pupil is None:
            return None
        return {"ellipse": self.pupil, "label": self.label, "color": self.color}


class PupilDetector(ABC):
    """Base class for a pluggable pupil detector.

    Subclass, set ``name`` and (optionally) ``params``, implement ``detect``, and
    decorate the class with ``@registry.register`` — it then appears in the
    dashboard and the montage automatically.
    """

    name: str = "unnamed"
    description: str = ""
    params: list = []                  # list[ParamSpec]

    def default_params(self) -> dict:
        return {p.name: p.default for p in self.params}

    def coerce_params(self, values: dict) -> dict:
        """Clamp/round a params dict against this detector's spec, filling gaps."""
        out = self.default_params()
        by_name = {p.name: p for p in self.params}
        for k, v in (values or {}).items():
            if k in by_name:
                out[k] = by_name[k].coerce(v)
        return out

    @abstractmethod
    def detect(self, ctx: DetectionContext, params: dict) -> DetectionResult:
        """Run detection on ``ctx`` and return a ``DetectionResult``."""
        raise NotImplementedError
