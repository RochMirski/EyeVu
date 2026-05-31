At the next attempt at improving the algorithm, add the ability for the capture of a calibration image which will allow for the detection of the placement of the led cover, without the eye in place.

# Three-frame capture + arc / eye-structure plan (2026-05-31)

**Capture now grabs three frames** in one short LED sequence (eye barely moves):
flash-only → **flash+ambient (both)** → ambient-only (`capture_image`). All three are
transferred (`ambient.jpg`, `flash.jpg`, `both.jpg`, `meta.json`; `meta.has_both`). The
combined `both.jpg` is the key new asset: a **bright retroreflecting pupil** *plus*
ambient-lit eyelid/iris structure in one frame.

**Arc-detection feasibility (investigated on existing ambient frames):** naive
Canny + HoughCircles is **unreliable** here — the pupil/iris boundary is a weak partial
arc, eyelid edges are fragmented, and Hough returns spurious large circles (only the
clearest frame landed on the pupil). Conclusion: an off-the-shelf arc detector won't carry
an eye-structure model on ambient-alone data. The current detector already *is* a targeted
pupil-arc fitter (radial gradient ridges → circle on the visible arc + gated concentric
iris), which is more robust than Hough here.

**Iris-band concentric detector (approved 2026-05-31, gated on `both.jpg` data).**
On a heavily-obstructed frame the radial edge cluster brackets the iris band (pupil edge
inside, limbus outside); fit the two as **concentric** circles so the larger iris arc pins
the centre/pupil when the pupil itself is occluded — **output both pupil and iris circles**.
On `both.jpg` the pupil retroreflects bright (best in RED) and the limbus is ambient-lit
(best in GREEN), and the two boundaries differ by gradient **sign** (pupil bright→dark,
limbus dark→bright) — that separates the rings cleanly, unlike ambient-alone. Build by
generalising `_ray_ridges` (signed, two ridges/ray) + reusing `_fit_concentric`; gate on
annulus consistency with fallback to today's pupil-only fit. Full plan:
`~/.claude/plans/wondrous-sparking-trinket.md`. **Harness is ready**: `test_pupil_detection.py`
loads `both.jpg` when present and writes `both_diag.jpg` (Red | Green | grad-mag) to design
against; the detector itself waits on real combined captures.

**Plan for heavy obstruction (needs the new `both.jpg` data to develop/validate):**
model the eye as **concentric arcs at three scales** — eyelid (large-radius arcs, only
top/bottom visible), iris/limbus (medium), pupil (small) — sharing the iris/pupil centre.
Fit each from whatever arc is visible and use the larger, better-supported arcs (eyelid
aperture, iris) to constrain the pupil centre when the pupil itself is mostly occluded. The
combined frame should give cleaner edges (bright pupil anchor + lit structure) than the
ambient-only frames that defeated Hough above. Defer building until `both.jpg` captures
exist; extend the radial-ridge machinery (multi-ridge per ray → group into arcs) rather
than Hough.

# Pupil-detection checkpoint — 2026-05-30

Snapshot of the pupil-detection rework in [cap.py](cap.py) (`detect_pupil`) and the
offline harness [test_pupil_detection.py](test_pupil_detection.py). Detection runs on the
**ambient** image (rotated to display orientation) and the result is drawn on the flash
image. Tuned against today's captures in `Transfers/capture_20260530_*`.

Interpreter with the deps: `C:/Users/zmirs/anaconda3/envs/vision_env/python.exe`.

## What the images look like

Macro shots of one eye under **violet** illumination. Every frame contains:

- **Pupil** — a dark disc; the most reliably dark thing in the frame.
- **Corneal reflex** — a bright near-white spot on/near the pupil. Sits at the pupil
  *edge* (often the top), and sometimes on the iris just outside the pupil — so it marks
  *where the eye is*, not the pupil centre.
- **Dark occluder patch** — a large near-black region (socket / brow / LED cover); the
  pupil's upper edge merges into it, so that part of the boundary has no intensity edge.
- **Bright glow / eyelid crease** — strong bright regions below/around the eye.

Facts that drive the design:
- **Green channel** has the cleanest eye structure. Under violet light, skin reflects
  red+blue and is bright in R and B but **dark in green**, so green suppresses the skin
  glow *regardless of skin or eye colour* — a robustness property tied to the illuminant,
  not the subject. (Blue saturates everywhere; luma includes the bright glow.)
- The **pupil/iris boundary is a clean gradient ridge** (visible as a partial circle in a
  Sobel magnitude map) even though the raw intensity step is small — so detect the gradient
  ridge, not an intensity threshold.
- **Pupil and iris are concentric** — exploited as a constraint (see below).
- The pupil's upper edge is usually missing (merges with the dark socket), so only a
  partial boundary **arc** is visible → fit a **circle** (3 DOF), never an ellipse (5 DOF).

## Current pipeline (`detect_pupil`, takes the BGR ambient image)

1. **Reflex anchor** — `_find_corneal_reflex` scores bright blobs by compactness × dark
   surround. Lands on the pupil in 8/8 today-captures.
2. **Green channel** prep — inpaint the reflex (`_inpaint_reflections` with the reflex
   mask) so the pupil reads continuous, then Gaussian blur. No CLAHE, no LED white-out.
3. **Gradient-ridge radial edges** (`_ray_ridges` / `_ray_edges`) — cast 96 rays from the
   centre; each ray's edge is the **innermost** outward gradient ridge (local max of the
   dark→bright radial derivative) above a per-ray gate (`_RAD_GRAD_K` × mean |gradient|, so
   contrast/illumination-robust). *Innermost*, not global-max, so when both the pupil/iris
   and the stronger iris/sclera edges lie on a ray the radius stays on the pupil. Rays into
   the dark patch have no outward ridge → drop out (occlusion handled implicitly). Rays
   start past the reflex blob (`r_start`).
4. **Iterative robust circle fit** (`_fit_circle_robust`, Kasa + median-radius refit) —
   recentres onto the pupil over a few passes. Stable on a partial arc (an ellipse fit is
   not), so it converges instead of drifting into the glow.
5. **Concentric iris refinement** (`_fit_concentric`) — detect the iris ridge in the
   annulus beyond the pupil and jointly fit both rings about a **shared centre**; the larger
   iris arc pins the centre when the pupil arc is short or the reflex was on the iris.
   **Gated**: adopted only if the iris points truly lie on a circle (low RMS,
   `_IRIS_MAX_RMS_FRAC`) and the radius ratio is plausible — otherwise a safe no-op.
6. **Validation** — radius bounds + angular coverage (`_RAD_MIN_COVER`).

### Per-stage debugging
- `cap.PUPIL_DEBUG_STAGES = []` makes `detect_pupil` record each stage
  (`1_green`, `2_reflex`, `3_green_prepped`, `4_recentre_edges`, `5_iris_concentric`,
  `9_final`).
- Harness `DEBUG_MODE` (env var or constant): `"off"` | `"stages"` | `"montage"`.
  Run: `DEBUG_MODE=montage <python> test_pupil_detection.py [capture_folder]`.
- `Transfers/_today_finals.jpg` is an 8-up contact sheet (regenerated by the inline script
  used during tuning).

## Results (today's 8 captures)

After switching to **gradient-ridge edges**, the circle bounds the pupil in **all 8**,
including the previously-failing reflex-on-iris frames (145416, 145557, 145710). Reflex
anchor correct in 8/8. Big improvement over both the old Orlosky detector (landed on the
left dark patch) and the earlier intensity-threshold radial version (drifted into the glow).
Residual: centres sit slightly high (reflex is at the pupil top) and a couple of radii catch
a bit of iris.

## What was tried and learned about concentricity

- The concentric two-ring fit is implemented and sound, **but the iris/limbus is not a clean
  circle in these tight macro frames** — the annulus gradient picks up scattered eyelid /
  glow / noise (visible in the `5_iris_concentric` stage). Unguarded, this inflated the fit.
- So the refinement is **gated on the iris points actually lying on a circle**; on this data
  it correctly does nothing, and the gradient-only pupil fit stands. The machinery remains
  for setups where a real limbus is visible (wider field of view, brown iris on white
  sclera, etc.).

## Ambient + flash fusion (2026-05-30, implemented)

`detect_and_annotate` now fuses two complementary cues (both computed in the rotated
display orientation):

- **Ambient dark-disc** — `detect_pupil`, as above. Stashes its result + a confidence
  (`coverage × inlier-fraction`) and the reflex anchor in module globals `_LAST_*`.
- **Flash red-eye** — `detect_redeye(flash_bgr, ax, ay, rmin, rmax)`. Searches a window
  around the ambient anchor for the bright warm retroreflection on a red-weighted
  (`R − 0.4·B`), blurred glow map; returns `(cx, cy, r, peak)` only if the glow is bright
  (`peak ≥ _REDEYE_MIN_PEAK`), pupil-scale, and genuinely warm (`R > B`). Constraining to
  the anchor neighbourhood is essential — unconstrained it grabs stray warm skin reflections.
- **`_fuse_pupil`** — a strongly-trusted red-eye (`peak ≥ _REDEYE_TRUST`) that *disagrees*
  with the ambient fit (or when there is no ambient fit) **wins** (the occluded case);
  otherwise the ambient circle stands and an agreeing red-eye corroborates it. Low-confidence
  results are drawn **amber** and labelled `low-conf` so the operator can re-capture.

Debug stages `7_flash_redeye` and `8_fused` are added to the montage (drawn on the
brightened flash), so the whole fused pipeline is inspectable.

**Result:** `145755` (the occluded frame whose ambient fit was wrong) now correctly uses the
red-eye (`peak=155`) and lands on the pupil retroreflection; `145630` likewise. The clear
frames continue to use the ambient fit.

**Caveats:**
- Ambient-sourced results are drawn on the flash, which is captured slightly later — visible
  **inter-capture eye motion** offsets some ambient circles on the flash image. The red-eye
  path is immune (it is measured in the flash itself). If this matters, consider detecting
  primarily in the flash, or capturing closer together.
- The red-eye radius comes from the glow blob, which may not equal the true pupil aperture;
  it localises the pupil well but the radius is approximate.

## Key finding — ambient vs flash are complementary (2026-05-30)

`capture_20260530_145755` exposes a hard limit of ambient-only detection: the pupil is
**mostly hidden behind the dark occluder patch**, so the visible boundary is one-sided and
the circle drifts onto the occluder/glow boundary (which is itself circle-like). Tested
confidence metrics — fit RMS, interior-vs-ring darkness contrast, and angular spread of the
edge points — and **none cleanly separates this bad fit from the good frames**, because the
wrong circle is geometrically plausible. So a simple quality gate can't rescue it.

Inspecting the **flash** images:
- `145755` (ambient occluded) → **strong red-eye retroreflection**: a bright orange pupil
  with visible retinal vessels. Light returns through the pupil → a strong *positive* pupil
  cue, independent of the ambient dark-disc contrast/occlusion.
- `145250` (ambient clear) → flash red-eye is **weak/partial** (a faint crescent).

The two signals are **complementary** (red-eye is strong only when illumination/pupil/camera
are well aligned, which varies per frame). A robust detector likely needs to **fuse** them:
the ambient dark-disc when the pupil is clear, the flash bright-pupil when the ambient is
occluded — cross-validating location and rejecting when neither agrees. This revisits the
original "red eye pupil detection" direction (see `blob_detection.py`).

**Decision (resolved):** move to ambient+flash fusion — implemented above. Occluded frames
that neither cue resolves are flagged low-confidence for re-capture (operator can retake).

## LED-cover calibration (2026-05-31, implemented)

A one-off **calibration capture of the LED cover with no eye in place** (ideally the cover
against a white background) now locates the cover as a fixed mask, so detection treats that
region as *known-occluded* instead of guessing per-frame. This replaces reliance on the
fragile dynamic `_find_led_cover_mask` (which bails when the cover fuses with the socket).

- **Build / store / load** (`cap.py`): `build_cover_mask(frame)` infers the cover as the
  **largest near-black (`_COVER_CALIB_DARK`), edge-touching** component — the cover blocks
  the LED so it reads ~0 and intrudes from a frame edge, so this needs no perfectly
  white/uniform background (an off-white or noisy backdrop is merely brighter; stray dark
  specks that don't touch an edge are ignored). → `save_cover_calibration()` →
  `calibration/led_cover_mask.png`, stored in the display (rotated) orientation.
  `load_cover_mask((h,w))` is cached and resizes to the frame. Toggle `USE_COVER_CALIB`.
- **Capture**: easiest is **live mode** — press **`c`**, which grabs a no-eye frame, shows
  the inferred cover tinted over the image, and waits (**y/ENTER = save, any key = cancel**).
  The `calibrate` command does the same non-interactively. On the dev side,
  `build_cover_calibration.py <image> [--rot N]` converts a calibration image to the mask.
- **Reaches this machine automatically**: after a Pi calibration saves, `upload_calibration()`
  POSTs the mask to the receiver under the special `calibration` folder; `receiver.py` routes
  that to its own `calibration/` dir (== `cap.COVER_CALIB_DIR`), so the dev-side harness loads
  the same mask with no manual copy. Verified end-to-end (POST → receiver → `load_cover_mask`).

## Pi-side processing OFF by default (2026-05-31)

`PI_DETECT = False`: `capture_image` now just captures + transfers the raw pair (no on-Pi
detection); detection runs on the dev machine via `test_pupil_detection.py` on the received
captures. Re-enable on the Pi at runtime with `pi_detect 1`. (Live-mode `p` overlay is
unchanged and still available for a quick visual check.)
- **Use in detection**: `_ray_ridges` takes the cover mask and **drops any boundary ridge
  whose point falls inside the cover** — those are the cover/eye edge, not the pupil, so they
  can no longer drag the circle fit onto the occluder. A `0_cover_calib` debug stage shows
  the mask. With **no calibration present, everything is a clean no-op** (graceful default).

Note: the throwaway "derive the cover from the median of existing captures" idea was dropped
— the underexposed ambient frames are dark almost everywhere, so it couldn't isolate the
cover from the general gloom (it masked ~37–60% of the frame). A real white-background
calibration shot (user-supplied) isolates the cover cleanly; that's the supported path.

## Known issues / next steps

1. **Centre bias / partial arc.** The pupil's upper edge is missing, so the circle centre
   sits a little high. Options: weight the fit by arc span, or require the arc to straddle
   both vertical halves before trusting the centre.
2. **Colour robustness is untested on other subjects.** The green/violet rationale is sound
   but only one eye/skin tone is in hand — validate on a brown iris and a different skin tone
   when captures exist.
3. **Dead code to prune** once locked in: `_swirski_support`, `_swirski_fit_ellipse`,
   `_fit_ellipse_robust`, `_segment_pupil_at`, `_refine_center` are no longer on the active
   path.

## Key tunables (in cap.py)

`_REFLEX_*` (anchor); `_RAD_N_ANGLES`, `_RAD_SEARCH_MULT`, `_RAD_GRAD_K`,
`_RAD_RECENTRE_ITERS`, `_RAD_MIN_COVER`, `_RAD_INLIER_PX` (pupil ridge + circle fit);
`_IRIS_RP_MIN/MAX`, `_IRIS_SEARCH_MAX`, `_IRIS_MIN_COVER`, `_IRIS_MAX_RMS_FRAC` (concentric
refine); `_SW_MIN_R_FRAC`/`_SW_MAX_R_FRAC` (pupil radius bounds as a fraction of min(h, w));
`USE_COVER_CALIB`, `_COVER_CALIB_DARK`, `_COVER_CALIB_DILATE` (LED-cover calibration).
