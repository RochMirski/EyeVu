"""Baseline module — wraps the production detector in cap.py unchanged.

This is the proven radial gradient-ridge + robust circle fit + gated concentric
iris detector, fused with the flash red-eye cue, exactly as it runs on the Pi and
in ``cap.detect_and_annotate``.  We reuse it verbatim (no logic duplicated) and
simply harvest the per-stage debug images it already produces via
``cap.PUPIL_DEBUG_STAGES``.  It is registered first, so it is the default source
of ``annotated.jpg`` and the reference every other module is compared against.
"""

from __future__ import annotations

import time

from .. import context as _ctx
from ..base import DetectionContext, DetectionResult, PupilDetector
from ..registry import register

cap = _ctx.cap


@register
class RidgeBaseline(PupilDetector):
    name = "ridge_baseline"
    description = ("Existing cap.py detector: reflex anchor -> green-channel radial "
                   "gradient ridges -> robust circle fit -> gated concentric iris, "
                   "fused with flash red-eye.")
    params = []   # tuned via cap.py module globals; exposed read-only here

    def detect(self, ctx: DetectionContext, params: dict) -> DetectionResult:
        res = DetectionResult(detector=self.name)
        if not cap.CV2_AVAILABLE:
            res.ok = False
            res.notes = "cv2 unavailable"
            return res

        cv2 = cap.cv2
        t0 = time.perf_counter()

        # Arm cap's debug sink and label mode so the wrapped detector records the
        # same stages it does in the Pi harness; restore afterwards.
        prev_sink = cap.PUPIL_DEBUG_STAGES
        prev_debug = cap.SWIRSKI_DEBUG
        cap.PUPIL_DEBUG_STAGES = []
        cap.SWIRSKI_DEBUG = True
        try:
            # Ambient dark-disc fit (mirrors detect_and_annotate's inner block,
            # but on the already-rotated ctx images so no re-rotation happens).
            # Run on the cover-filled image (smooth fill, no hard cover edge to
            # drag the fit), confined to the ROI box.
            amb_in = ctx.restrict_box(ctx.ambient_filled)
            cap.detect_pupil(amb_in)
            amb_pupil = cap._LAST_PUPIL
            amb_conf = cap._LAST_CONF
            anchor = cap._LAST_ANCHOR

            redeye = None
            if anchor is not None:
                redeye = cap.detect_redeye(ctx.flash, anchor[0], anchor[1],
                                           ctx.rmin, ctx.rmax)
                rv = cv2.convertScaleAbs(ctx.flash, alpha=3.0)
                cv2.circle(rv, (anchor[0], anchor[1]), 5, (255, 0, 255), -1)
                if redeye is not None:
                    cv2.circle(rv, (int(redeye[0]), int(redeye[1])),
                               max(6, int(redeye[2])), (0, 255, 255), 2)
                cap._dbg("7_flash_redeye " +
                         (f"peak={redeye[3]:.0f}" if redeye else "none"), rv)

            fused = cap._fuse_pupil(amb_pupil, amb_conf, redeye)
            stages = cap.PUPIL_DEBUG_STAGES or []
        finally:
            cap.PUPIL_DEBUG_STAGES = prev_sink
            cap.SWIRSKI_DEBUG = prev_debug

        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Lead with the actual (cover-filled, box-restricted) detector input.
        roi_vis = cv2.cvtColor(ctx.restrict_box(ctx.green_filled), cv2.COLOR_GRAY2BGR)
        _ctx.draw_roi(roi_vis, ctx)
        res.add_stage("0_search_roi", roi_vis)
        for name, img in stages:
            res.stages.append((name, img))

        if fused is not None:
            cx, cy, r, source, confident = fused
            res.pupil = ((float(cx), float(cy)), (2.0 * r, 2.0 * r), 0.0)
            res.confidence = float(amb_conf if source == "ambient" else
                                   min(1.0, (redeye[3] / 255.0) if redeye else amb_conf))
            res.color = (0, 255, 0) if confident else (0, 165, 255)
            res.label = source + ("" if confident else " low-conf")
            res.notes = f"source={source} confident={confident}"
        else:
            res.notes = "no pupil"
        return res
