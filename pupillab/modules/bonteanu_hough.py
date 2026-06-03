"""Bonteanu circular-Hough pupil detector (literature module).

After G. Bonteanu et al., *A New Pupil Detection Algorithm Based on Circular
Hough Transform Approaches* (ISSCS 2019).  The pupil is found as the strongest
circular edge in the eye image via the Circular Hough Transform.  Here it is
adapted to EyeVu's frames: the green channel (cleanest dark-pupil contrast under
violet light) is reflex-inpainted and blurred, optionally binarised, and
``cv2.HoughCircles`` (gradient accumulator) proposes circles; the candidate is
chosen by dark-interior support and proximity to the corneal-reflex anchor and
constrained to the plausible pupil-radius band.

Expected to be brittle on heavily occluded / partial-arc frames (Hough wants a
near-complete circle) — included as an honest literature baseline to compare the
targeted ridge fitter against.
"""

from __future__ import annotations

import time

import numpy as np

from .. import context as _ctx
from ..base import DetectionContext, DetectionResult, ParamSpec, PupilDetector
from ..registry import register

cap = _ctx.cap
cv2 = cap.cv2


@register
class BonteanuHough(PupilDetector):
    name = "bonteanu_hough"
    description = "Circular Hough Transform pupil detector (Bonteanu et al., 2019)."
    params = [
        ParamSpec("blur", 7, 1, 21, 2, "int", "Gaussian blur kernel (odd)."),
        ParamSpec("dp", 1.2, 1.0, 3.0, 0.1, "float", "Hough accumulator inverse ratio."),
        ParamSpec("param1", 80, 30, 200, 5, "int", "Canny upper threshold (edges)."),
        ParamSpec("param2", 18, 8, 100, 1, "int", "Accumulator threshold (lower = more circles)."),
        ParamSpec("minRadius", 20, 4, 200, 1, "int", "Min circle radius (px)."),
        ParamSpec("maxRadius", 130, 8, 300, 1, "int", "Max circle radius (px)."),
        ParamSpec("binarize", 0, 0, 1, 1, "int", "Show/feed Bonteanu adaptive binarisation (0/1)."),
    ]

    def detect(self, ctx: DetectionContext, params: dict) -> DetectionResult:
        p = self.coerce_params(params)
        res = DetectionResult(detector=self.name, color=(255, 200, 0))
        if not cap.CV2_AVAILABLE:
            res.ok = False
            res.notes = "cv2 unavailable"
            return res
        t0 = time.perf_counter()

        # 1. Prep: reflex-inpaint the green channel (so the bright spot can't seed
        #    a spurious edge) then blur.  Reuses cap's inpainting helper.
        green = ctx.green_filled            # cover-filled (smooth, no hard edge)
        prepped = cap._inpaint_reflections(green, ctx.reflex_mask)
        k = int(p["blur"]) | 1
        prepped = cv2.GaussianBlur(prepped, (k, k), 0)
        # Confine the Hough search to the ROI box (cover already filled).
        prepped = ctx.restrict_box(prepped)
        res.add_stage("1_green_prepped", prepped)

        # 2. Bonteanu adaptive binarisation (feedback-thresholded): Otsu gives the
        #    global pupil/background split here.  Shown for inspection; optionally
        #    fed to Hough instead of the grayscale.
        thr_val, binar = cv2.threshold(prepped, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res.add_stage(f"2_binarised otsu={thr_val:.0f}", binar)

        hough_src = binar if p["binarize"] else prepped

        # 3. Edge view (what the gradient accumulator builds on).
        edges = cv2.Canny(hough_src, p["param1"] // 2, p["param1"])
        res.add_stage("3_edges", edges)

        # 4. Circular Hough Transform.
        circles = cv2.HoughCircles(
            hough_src, cv2.HOUGH_GRADIENT, dp=p["dp"],
            minDist=max(8, int(ctx.rmin)),
            param1=float(p["param1"]), param2=float(p["param2"]),
            minRadius=int(p["minRadius"]), maxRadius=int(p["maxRadius"]))

        cand_vis = cv2.cvtColor(prepped, cv2.COLOR_GRAY2BGR)
        chosen = None
        if circles is not None:
            circles = np.around(circles)[0]
            best_score = -1e9
            diag = float(np.hypot(*ctx.shape))
            for (x, y, r) in circles:
                x, y, r = int(x), int(y), int(r)   # avoid uint16 wrap in scoring
                cv2.circle(cand_vis, (x, y), r, (0, 140, 255), 1)
                if not ctx.in_roi(x, y):           # reject centres outside the box
                    continue
                interior = self._interior_mean(green, x, y, r)
                dark = 255.0 - interior                       # darker pupil = better
                if ctx.anchor is not None:
                    ax, ay, _ = ctx.anchor
                    anchor_pen = 120.0 * (np.hypot(x - ax, y - ay) / diag)
                else:
                    anchor_pen = 0.0
                score = dark - anchor_pen
                if score > best_score:
                    best_score = score
                    chosen = (int(x), int(y), int(r), interior)
        res.add_stage(f"4_candidates n={0 if circles is None else len(circles)}",
                      cand_vis)

        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0

        final_vis = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
        if chosen is not None:
            x, y, r, interior = chosen
            res.pupil = ((float(x), float(y)), (2.0 * r, 2.0 * r), 0.0)
            # Confidence: dark interior contrast vs the surrounding annulus.
            ring = cap._annulus_mean(green, x, y, r)
            res.confidence = float(np.clip((ring - interior) / 80.0, 0.0, 1.0))
            res.label = "hough"
            cv2.circle(final_vis, (x, y), r, (255, 200, 0), 2)
            cv2.circle(final_vis, (x, y), 3, (255, 200, 0), -1)
            res.notes = f"r={r} interior={interior:.0f}"
        else:
            res.notes = "no circle found"
        _ctx.draw_roi(final_vis, ctx)
        res.add_stage("5_final", final_vis)
        return res

    @staticmethod
    def _interior_mean(gray, x, y, r) -> float:
        """Mean intensity inside the disc (x, y, r), clipped to the frame."""
        h, w = gray.shape[:2]
        ri = max(1, int(r * 0.7))
        x0, x1 = max(0, x - ri), min(w, x + ri + 1)
        y0, y1 = max(0, y - ri), min(h, y + ri + 1)
        win = gray[y0:y1, x0:x1]
        if win.size == 0:
            return 255.0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= ri * ri
        return float(win[disc].mean()) if np.any(disc) else 255.0
