# pupillab — extensible pupil-detection lab

A small framework for trying, comparing, and tuning pupil detectors on EyeVu
captures. Several detectors run side-by-side on the same capture; each contributes
its own labelled debug stages, so the montage and the Streamlit dashboard extend
**automatically** when you add a module. The production detector in `cap.py` is
reused unchanged (the `ridge_baseline` module just wraps it), so nothing here
touches the Pi path.

## Setup

```powershell
# from the repo root (…/Software/EyeVu)
./setup_dev.ps1            # creates/updates conda vision_env, installs deps
#   ./setup_dev.ps1 -SkipML    # skip torch (no RITnet)
#   ./setup_dev.ps1 -Launch    # set up then open the dashboard
```

If PowerShell blocks the script ("running scripts is disabled on this system"),
either run it once without changing settings:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_dev.ps1
```

or allow your own local scripts for your user (persists):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # then: ./setup_dev.ps1
```

Or manually, in your `vision_env`:

```powershell
pip install -r requirements.txt        # core
pip install -r requirements-ml.txt     # optional: torch for the RITnet module
```

For RITnet, also drop `best_model.pkl` into `models/ritnet/` — see
[models/ritnet/README.md](../models/ritnet/README.md). Without it the RITnet
module just reports "unavailable"; everything else still runs.

## Run

**Interactive dashboard** (primary surface):

```powershell
streamlit run pupillab/app.py
```

Pick a capture, tick detectors on/off, drag the per-detector sliders, and watch
the overlays and debug stages re-run live. Buttons in the sidebar export
`debug_montage.jpg` / `annotated.jpg` into the capture folder.

**Headless batch** (no browser; good for sweeping all captures):

```powershell
python pupillab/run_batch.py                          # all Transfers/capture_*
python pupillab/run_batch.py capture_20260530_145755  # one folder
python pupillab/run_batch.py --only ridge_baseline,ritnet
```

The original `test_pupil_detection.py` still works as before and is untouched.

## Detectors included

| name             | what it is                                                              |
|------------------|-------------------------------------------------------------------------|
| `ridge_baseline` | the existing `cap.py` detector (reflex anchor → radial gradient ridges → robust circle fit → gated concentric iris, fused with flash red-eye). Drives `annotated.jpg`. |
| `bonteanu_hough` | Circular Hough Transform pupil detector (Bonteanu et al., 2019). Literature baseline; weak on heavily-occluded arcs by design. |
| `ritnet`         | pre-trained RITnet segmentation (OpenEDS near-IR) → pupil-class ellipse. Optional ML; needs torch + weights. |

## Search ROI (restrict where the pupil is looked for)

The pupil is assumed to lie in a box just above the LED cover. From the calibrated
cover mask, `build_context` derives a square ROI whose side spans the image's
**smaller dimension**, centred horizontally on the cover and slightly **above its
top edge**, with the known cover region subtracted. The corneal-reflex anchor and
every detector are confined to it (`ctx.search_mask` / `ctx.roi`), which removes
the surrounding eyelid/glow/socket clutter that previously dragged fits off-pupil.

- Tune it live in the dashboard sidebar (**Search ROI**): toggle on/off, box side
  (× smaller image dim), and how far above the cover top to centre it (× side).
- Batch: `--no-roi`, `--roi-side <frac>`, `--roi-offset <frac>` on `run_batch.py`.
- Defaults live in `pupillab/context.py` (`ROI_ENABLE`, `ROI_SIDE_FRAC`,
  `ROI_OFFSET_FRAC`). With no cover calibration the ROI is skipped (full-frame).

A new module gets this for free: call `ctx.restrict(img)` to zero everything
outside the box, `ctx.in_roi(x, y)` to reject out-of-box candidates, and
`context.draw_roi(vis, ctx)` to draw the box on a debug image.

## Add a new detector (3 steps)

1. Create `pupillab/modules/my_detector.py`:

   ```python
   from .. import context as _ctx
   from ..base import DetectionContext, DetectionResult, ParamSpec, PupilDetector
   from ..registry import register

   cap = _ctx.cap            # reuse cap.py helpers / cv2 if useful

   @register
   class MyDetector(PupilDetector):
       name = "my_detector"
       description = "one-line summary"
       params = [ParamSpec("thresh", 40, 0, 255, 1, "int", "binarisation level")]

       def detect(self, ctx: DetectionContext, params: dict) -> DetectionResult:
           p = self.coerce_params(params)
           res = DetectionResult(detector=self.name, color=(0, 255, 255))
           # ... compute on ctx.green / ctx.gray / ctx.ambient (BGR, rotated) ...
           res.add_stage("1_something", some_image)     # appears in montage + GUI
           res.pupil = ((cx, cy), (2*r, 2*r), 0.0)       # OpenCV ellipse tuple
           res.confidence = 0.8
           return res
   ```

2. Add one import line to `pupillab/modules/__init__.py`:
   `from . import my_detector  # noqa: F401`

3. Done — the dashboard shows a checkbox + sliders for it, the montage gains a
   block for it, and `run_batch.py` includes it. No other file changes.

### What you get from the shared context

`DetectionContext` (built once per capture, in display/rotated orientation):
`ambient`/`flash`/`both` (BGR), `gray`, `green`, `cover_mask` (calibrated LED
cover, reuse it as a known-occluded mask), `anchor` (corneal reflex `(cx,cy,r)`),
`reflex_mask`, `rmin`/`rmax` (plausible pupil-radius band), and `meta`.

Useful `cap.py` helpers to reuse: `cap._inpaint_reflections`,
`cap._find_corneal_reflex`, `cap._annulus_mean`, `cap._draw_overlays`,
`cap.load_cover_mask`.
