"""RITnet pre-trained segmentation module (ML) — thin wrapper.

The actual inference lives in the shared, Pi-friendly ``ritnet_infer`` module (so
the cap.py guidance cascade and this detector run identical code).  This wrapper
just builds the crop centre from the detection context, calls
``ritnet_infer.locate``, and adapts the result into a ``DetectionResult`` with
debug stages.

Self-guarding: if torch or the weights are missing, ``detect`` returns a graceful
``ok=False`` result (see ``models/ritnet/README.md``).  Near-IR -> violet domain
gap: treat as a comparison baseline, not ground truth.
"""

from __future__ import annotations

import time

from .. import context as _ctx
from ..base import DetectionContext, DetectionResult, ParamSpec, PupilDetector
from ..registry import register

cap = _ctx.cap
cv2 = cap.cv2

import ritnet_infer  # repo root is on sys.path via pupillab.context  # noqa: E402


@register
class RITnetSeg(PupilDetector):
    name = "ritnet"
    description = ("Pre-trained RITnet eye-part segmentation (OpenEDS, near-IR) -> "
                   "pupil-class ellipse fit. Optional; needs torch + weights.")
    params = [
        ParamSpec("crop_dx", 0, -600, 600, 10, "int",
                  "Shift the 640x400 crop window right (+) / left (-), px."),
        ParamSpec("crop_dy", 0, -600, 600, 10, "int",
                  "Shift the 640x400 crop window down (+) / up (-), px."),
        ParamSpec("min_area", 60, 5, 5000, 5, "int",
                  "Min pupil-mask area (px, in full-res) to accept."),
    ]

    def detect(self, ctx: DetectionContext, params: dict) -> DetectionResult:
        p = self.coerce_params(params)
        res = DetectionResult(detector=self.name, color=(0, 0, 255))

        if not ritnet_infer.available():
            res.ok = False
            res.notes = ritnet_infer._load_error or "RITnet unavailable"
            return res

        h, w = ctx.shape
        # Initial crop centre: tangent above the LED cover (bottom edge at the
        # cover top, centred on its top-middle) — that's where the pupil is
        # expected.  Falls back to reflex anchor -> ROI-box centre -> image centre.
        if ctx.cover_mask is not None and cv2.countNonZero(ctx.cover_mask):
            ccx, ccy = ritnet_infer.tangent_crop_center(ctx.cover_mask, (h, w))
        elif ctx.anchor is not None:
            ccx, ccy = ctx.anchor[0], ctx.anchor[1]
        elif ctx.roi is not None:
            bx0, by0, bx1, by1 = ctx.roi
            ccx, ccy = (bx0 + bx1) // 2, (by0 + by1) // 2
        else:
            ccx, ccy = w // 2, h // 2
        ccx += p["crop_dx"]
        ccy += p["crop_dy"]

        t0 = time.perf_counter()
        r = ritnet_infer.locate(
            ctx.green_filled, reflex_mask=ctx.reflex_mask, anchor=ctx.anchor,
            crop_center=(ccx, ccy), min_area=p["min_area"],
            search_mask=ctx.search_mask, want_stages=True)
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not r.ok:
            res.ok = False
            res.notes = r.error
            return res

        res.stages = list(r.stages)
        # Add the ROI box to the final stage for context.
        if res.stages:
            _ctx.draw_roi(res.stages[-1][1], ctx)
        res.notes = r.notes
        if r.pupil is not None:
            res.pupil = r.pupil
            res.confidence = r.confidence
            res.label = "ritnet"
        return res
