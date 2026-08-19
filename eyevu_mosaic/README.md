# eyevu_mosaic

Post-session retinal mosaicing for EyeVu. Replaces the generic
feature + homography + RANSAC stitcher in [`mosaic.py`](../mosaic.py), which
failed whenever two patches shared only a handful of correspondences — which,
on this material, is almost always.

```bash
python -m eyevu_mosaic.run_session Sessions/session_20260815_223238
python -m eyevu_mosaic.run_session Sessions/sess_A Sessions/sess_B   # combine
python -m eyevu_mosaic.synth    /tmp/fake --n 20      # synthetic session
python -m eyevu_mosaic.validate --n 20 --seed 1       # score against truth
```

## Combining sittings

`run()` takes one session directory or several. Combining different sittings of
the **same eye** is one of the highest-value things you can do with this
pipeline, because there is no temporal assumption anywhere in it to violate:
every pair is treated independently, so extra captures are simply extra chances
to find an overlap. Measured on two real sessions:

| | placed alone | combined |
|---|---|---|
| `session_20260815_223238` (12) | 5, groups [5, 4, 2] | |
| `session_20260815_224337` (10) | 6, groups [6, 3] | |
| **both together (22)** | | **20, groups [20, 2]** |

17 of the 37 accepted pairs joined the two sittings — they genuinely bridge each
other's gaps rather than merely sitting side by side.

Every patch records which session it came from (`patches[].session` in
`bundle.json`, `sessions` in `meta.json`), and `raw/` is split per session
because capture filenames repeat across them. Sessions belonging to **different
patients must never be combined**; see the patient note below.

---

## What the input actually is

Every design decision below follows from measuring the 141 real patches in
[`Sessions/`](../Sessions):

| | min | **median** | max |
|---|---|---|---|
| content bounding box | 7×9 px | **45×32 px** | 125×101 px |
| mask area | 49 px | **943 px** | 7484 px |
| frame | 480×640 or 960×1280 | | |
| content as % of frame | | **0.28 %** | 2.4 % |

A patch is a ~45×32 px blob of retina adrift in a mostly black frame. Sessions
hold 1–40 patches and **mix both capture resolutions**. The illuminated region
is *not* a circle — it is a red-shift selection gated to the pupil, so it has a
ragged, per-capture boundary and there is no aperture radius to read off.
Specular highlights are already excluded upstream by `cap.redeye_extract`, which
thresholds and dilates them out of the mask before the extract is written.

Two consequences run through everything:

* **The patches are tiny.** Detection has to *upsample*, not downsample. And the
  memory architecture a 640 px aperture would need — tiled compositing over a
  memmapped canvas, an LRU cache of full-resolution frames — is machinery for a
  problem that does not exist here: a whole session's actual *content* is under
  2 MB.
* **A true overlap yields 3–5 inlier correspondences.** Measured over a real
  14-capture session, true pairs scored 3–5 inliers and false pairs 5–8. The
  distributions do not separate. Any acceptance rule keyed on inlier count is
  reading noise.

---

## The model ladder

The central decision: **do not fit a homography.** Under rotation of the eye,
the patch-to-patch mapping is the conjugate rotation `H = K R K⁻¹` — 3 DOF given
`K`, determined by **two** correspondences and overconstrained by three. That
converts "not enough matches" into "enough matches" without touching the
detector.

| level | model | DOF | min corr. | promote when |
|---|---|---|---|---|
| **L0** | `K R K⁻¹`, rotation only | 3 | **2** | default entry point |
| L1 | rotation + shared focal | 4 | 3 | ≥ 6 inliers |
| L2 | affine | 6 | 3 | ≥ 10 inliers **and** median residual −20 % |
| L3 | homography | 8 | 4 | ≥ 20 inliers **and** −20 % vs L2 |
| L4 | 12-param quadratic (Can et al.) | 12 | 6 | ≥ 35 inliers, wide baseline, clear gain |

Promotion **always requires both** the inlier gate and the residual gate, because
a higher-DOF model always fits better in-sample. The level reached is recorded
per pair.

The brief's original gates (8 / 12 / 20 / 35) were written for ~640 px apertures
and are unreachable here; they are rescaled above. `ladder.enabled_max_level`
defaults to **2**. In practice every real edge settles at **L0** — which is the
correct outcome, not a failure. L3/L4 are implemented and unit-tested, ready for
when capture moves to full sensor resolution.

### What L0 really is at this scale

With a ~45 px patch and `f ≈ 10³` px, the projective terms of `K R K⁻¹` go as
`(x/f)² ≈ 3×10⁻⁴` px. **L0 is numerically a Euclidean transform** — in-plane
rotation plus translation. Two things follow:

* The uncalibrated focal length barely matters. There are no calibrated
  intrinsics for this rig; `focal_ref_px` is a documented placeholder, flagged
  as such in `meta.json`, and L1 refines it from data when patches grow.
* The physical story (camera at the eye's nodal point) is weaker than it looks —
  the scope sits at a working distance, so the pupil also translates. **The DOF
  argument is what carries the design**, not the optics. `K R K⁻¹` is kept over a
  bare Euclidean fit because it composes correctly on SO(3), which is what makes
  cycle consistency and rotation averaging meaningful, and because it stays
  correct as patches grow.

### Rotation error is not one number

A rotation's three DOF move the image by wildly different amounts here:

| component | image effect | at 1° |
|---|---|---|
| pointing (rx, ry) | `f · angle`, f ≈ 3600 analysis px | **63 px** |
| roll (rz) | `r · angle`, r ≈ 70 px | **1.2 px** |

Roll is simultaneously the **worst-observed** parameter (fitted from the angular
arrangement of a handful of points over a short lever arm) and the **least
consequential**. Measured on edges independently confirmed correct: median total
error 2.12°, of which pointing was **0.030°** and roll 2.12°.

So this package reports and thresholds on **image displacement in pixels**, not
degrees, everywhere it matters — cycle consistency, per-edge residuals, global
accuracy. A 2° cycle gate, the natural choice, discarded 3 of 4 triangles in a
session where every one was good.

---

## Pipeline

1. **Preprocess.** Green channel; mask from the capture-time selection, eroded
   slightly; illumination flattened; CLAHE; speculars re-excluded.
2. **Quality gate.** Focus (variance of Laplacian in mask), saturated fraction,
   mean luminance, mask area — gated in *native* px against the measured
   distribution. Every patch is recorded with its metrics; a rejected patch
   always carries a reason. Nothing is silently dropped.
3. **Detect.** SIFT with RootSIFT normalisation, ANMS spatial spread, capped.
   Named-landmark injection is wired and tested but unpopulated — see below.
4. **Shortlist.** Below 25 patches, all pairs (300 pairs is seconds). Above,
   global-descriptor ranking + top-k + a random sample so unexpected overlaps
   stay findable. Skipped pairs are logged with the reason.
5. **Match.** Mutual NN + a deliberately *loose* ratio test, similarity-space
   Hough seeding, PROSAC, the ladder, then the photometric gate.
6. **Direct alignment fallback** for pairs the feature path cannot solve.
7. **Graph.** Edge weighting, triangle cycle consistency, components, maximum
   spanning tree, chained initialisation from the best-connected node.
8. **Global refinement.** Rotation averaging, then bundle adjustment over
   per-patch rotations on SO(3) (one patch fixed for gauge), Huber loss, scipy
   `least_squares` with an explicit `jac_sparsity`. Then global outlier
   rejection, then re-solve.
9. **Composite.** Azimuthal equidistant about the mean gaze direction, exposure
   normalised, per-pixel weighted-median blend, tiled, plus a coverage map.

### Three things that were measured, not chosen

**The ratio test is loose (0.98), not strict (0.7).** The ratio test exists to
make putative matches trustworthy enough for a high-DOF model under plain
RANSAC. Here the model needs two correspondences, PROSAC orders its draws by
descriptor distance anyway, and acceptance is photometric. Solvable pairs
(≥ 3 GT-consistent matches), measured:

| ratio | 0.75 | 0.85 | 0.92 | 0.98 |
|---|---|---|---|---|
| solvable | 8 % | 19 % | 33 % | **42 %** |

…with **precision staying at 1.00 throughout**, because the NCC gate absorbs the
extra outliers exactly as intended. This also makes `min_inlier_frac` meaningless
— 4 inliers of 100 putative is 0.04 — so it drops from 0.25 to 0.02.

**Illumination is removed by a low-order polynomial, not a Gaussian high-pass.**
A high-pass can only separate illumination from anatomy if they occupy different
frequency bands. At ~15 native px of patch radius they do not. Median NCC over
pairs that genuinely overlap:

| method | subtract σ=0.125R | σ=0.4R | σ=0.6R | *no flattening* | **poly, order 2** |
|---|---|---|---|---|---|
| NCC | 0.075 | 0.173 | 0.213 | 0.244 | **0.93** |

Every Gaussian setting is worse than doing nothing. A 6-coefficient surface
separates by *form* instead of frequency, and has nowhere to hide vessel
structure however small the patch is.

**Analysis scale depends on frame resolution, never on the patch's own bbox.**
Sizing each patch so its own bounding box hits a fixed size — the obvious thing,
and what the previous implementation did — gives a 35 px patch ×6.86 and a 45 px
patch ×5.33: ~29 % of relative scale *invented* between two views of the same
retina. A similarity model hides that by spending a DOF on it. A 3-DOF rotation
model has no such DOF and simply fails. Sessions genuinely mix 480×640 and
960×1280 captures, so this has to be right.

### The direct-alignment fallback

A true overlap yields a median of 2–3 *correct* descriptor matches. Below three,
nothing can be fitted — yet those same pairs agree at NCC 0.8+ once warped
together. They are not ambiguous, they are **undescribable**.

This is tractable **only because the model is 3-DOF**: the search is two
translations and a roll, solved by masked FFT normalised cross-correlation
(Padfield 2012) per roll hypothesis, screened at zero roll first. An 8-DOF
homography could not be searched this way — the reduced-DOF model pays off twice.

Direct edges are held to a *stricter* photometric threshold (0.66 vs 0.50),
because a direct search optimises the very quantity it is then judged on. They
contribute no correspondences, so they enter bundle adjustment as explicit
relative-rotation constraints instead.

Measured on a 20-patch synthetic session:

| | feature path only | + direct fallback |
|---|---|---|
| edge recall | 0.20 | **0.39** |
| components | [5, 5, 3] | **[18]** |

### Robustness: two filters, because one is not enough

**Cycle consistency** rejects the weakest edge of any triangle whose loop
closure exceeds 18 analysis px. It uses evidence from *outside* the pair, which
is the only thing that reliably separates true from false edges here.

But it only tests edges that lie in a triangle, and these graphs are barely
denser than trees — 32 edges over 30 patches gave just **9 triangles**. So a
**global outlier rejection** pass follows: solve, ask each edge how far it is
from the consensus, drop the worst, re-solve. Two false edges took a 30-patch
session from ~1 px to 11 px of median misalignment. An edge whose removal would
disconnect the component is kept regardless — losing a patch entirely is worse
than placing it imperfectly — and reported either way.

Order matters: rejection runs **after** the solve. Rotation averaging *spreads* a
bad edge's error across the component and hides it; bundle adjustment pulls onto
the good structure and leaves it standing out at 117 px against a median of 0.2.

---

## Measured results

### Against ground truth (synthetic, 20 patches, 5 seeds)

The synthetic generator samples patches from a retinal map under known rotations
using the exact inverse of the compositor, so a perfect pipeline recovers the
input rotations exactly.

| seed | precision | recall | components | largest | alignment (native px) | time |
|---|---|---|---|---|---|---|
| 1 | 0.97 | 0.39 | [18] | 18 | **1.37** | 29.1 s |
| 2 | 0.92 | 0.40 | [19] | 19 | 4.92 | 29.4 s |
| 3 | 0.80 | 0.40 | [17, 2] | 17 | 3.16 | 26.3 s |
| 4 | 1.00 | 0.33 | [19] | 19 | 2.40 | 15.0 s |
| 5 | 0.90 | 0.32 | [17] | 17 | 4.10 | 25.8 s |
| **median** | **0.92** | **0.39** | | **18 / 20** | **3.16** | **26 s** |

Per-edge accuracy is much better than the global figure: pointing error **0.03°
≈ 0.5 native px**, from feature and direct edges alike. Roll error is ~1.6°,
which is ~1.9 analysis px and therefore near-irrelevant.

The gap between 0.5 px per edge and ~3 px globally is accumulation along the
graph, and it is bounded by how sparse the graph is: recall 0.39 leaves little
redundancy, so a single weak edge is rarely out-voted. When a false edge
survives as a bridge — one that cannot be dropped without disconnecting the
component — it shows up in `bundle.json` as a per-edge residual of ~117 px
against a median of 0.2 px, and is reported rather than hidden. More true edges
are what would improve this, which is why the learned-detector interface below
matters more than any further threshold tuning.

### Against the old pipeline (9 real sessions, 140 patches)

| | patches in largest mosaic | share |
|---|---|---|
| old `mosaic.py` | 29 | 21 % |
| **`eyevu_mosaic`** | **69** | **49 %** |

Two sessions that the old pipeline could only place 3–4 patches from now connect
almost entirely (12/13 and 13/14).

### Timing and memory

Measured on the **dev laptop** (x86-64, OpenCV 4.10, 4 threads):

| session | patches | match | global opt | compose | total | peak RSS |
|---|---|---|---|---|---|---|
| real, 12 patches | 12 | 6.8 s | 6.2 s | 0.3 s | **13.9 s** | 136 MB |
| real, 40 patches | 40 | 26.3 s | 18.2 s | 1.5 s | **47.1 s** | 140 MB |
| synthetic, 20 patches | 20 | — | — | — | **26 s** (median of 5) | ~120 MB |

> **These are laptop figures. The pipeline has not been run on a Pi Zero 2 W.**
> I have no access to the device from here, so the 5-minute / 30-patch target is
> **unverified**. Extrapolating from the ~6–10× single-thread gap, 30 patches
> would land at roughly 4–8 minutes — at or somewhat over target. Run
> `python -m eyevu_mosaic.run_session <session>` on the device; every stage's
> wall clock and peak RSS is written to `bundle.json`, so the real numbers need
> no extra instrumentation. If it is too slow, in order of impact:
> `direct_align=False` (roughly halves the time, at a large cost in
> connectivity), then `ransac_max_iters`, then `bundle_adjust=False`.

Peak RSS is ~140 MB against 512 MB of RAM, dominated by the Python and OpenCV
runtime rather than by image data. Memory was never the binding constraint at
this patch size, which is why the crop-and-hold design below replaces the tiled
architecture the brief specified.

---

## Deliberate deviations from the brief

Each of these was a measured decision, not an oversight.

| brief | here | why |
|---|---|---|
| Reduce to 640 px on the aperture's long edge | **Upsample ×3** | The patches are 45×32 px. 640 px would be a 14× upsample; ×3 and ×5 recover equally many pairs and ×3 is cheaper. |
| Never hold all patches in memory; LRU cache full-res frames | **Crop-and-hold** | A session's actual content is under 2 MB. Compositing samples the native crop directly — one resampling instead of two, and no disk IO in the compositing pass. |
| Tiled compositing, memmapped canvas | **Tiled, kept** | Implemented as specified so peak tile memory stays flat, even though it does not bind today. |
| Ladder inlier gates 8 / 12 / 20 / 35 | **6 / 10 / 20 / 35** | Written for 640 px apertures; unreachable when true pairs yield 3–5 inliers. |
| Ratio test ~0.8 | **0.98** | Measured: recall 19 % → 42 %, precision unchanged at 1.00. |
| Cycle error threshold 2° | **18 analysis px** | Degrees are dominated by roll, the least consequential DOF. A 2° gate discarded 3 of 4 good triangles. |
| Erode ~5 % of aperture radius | **5 %, capped at 8 px** | Kept, but capped: ~77 % of keypoints lie within 12 analysis px of the rim, and hard erosion drops true-pair recovery from 6/6 to 3/6. |
| Median blending rejects corneal reflections | **Kept, different reason** | Speculars are already excluded upstream. Median blending still earns its place against exposure outliers and misregistration. |
| Inject optic disc as a landmark prior | **Interface only** | There is no optic disc detector in this repo (every "disc" in `cap.py` is the *pupil*), and at 45×32 px the disc spans a few pixels. The mechanism is live and tested; nothing populates it. |
| Add a gaze-direction metadata field | **Not added** | Per your instruction: uncalibrated, and unreliable when the eye drifts between the target moving and the flash firing. `meta.json` carries a `fixation_targets` slot so a future calibrated capture needs no format change. |

Because there is no gaze prior, **disconnected components are emitted as separate
mosaics and the failure is reported**. No relative placement is fabricated.

---

## Session bundle

The contract with the dashboard. Written to `<session>/bundle/`:

```
raw/                original patches, byte-for-byte unmodified
meta.json           capture metadata, intrinsics (flagged uncalibrated), config used
bundle.npz          keypoints, descriptors (float16), analysis images + masks,
                    native crops, per-pair matches + inlier masks, rotations, tracks
bundle.json         per-patch quality + accept/reject reason; per-pair n_putative,
                    n_inliers, level, model, residual, NCC, accepted/rejected + reason;
                    skipped pairs + reason; triangle closure errors; MST edges;
                    components; BA convergence; per-stage timings and peak RSS
mosaic.png          full-resolution output
mosaic_outlined.png the same, with each contributing patch's boundary drawn in
                    its own colour and labelled with its index
mosaic_preview.jpg  downscaled for fast dashboard loading
coverage.png        per-pixel contribution count
log.txt
```

Disconnected groups add `mosaic_component_N.png`, `coverage_component_N.png` and
`mosaic_component_N_outlined.png`.

The outline **geometry** also goes into `bundle.json` under
`compose.components[i].outlines` as `{patch_index: [[x, y], …]}` in output
pixels, so the dashboard can hit-test, recolour or animate them rather than
being handed a flattened picture. Verified against the coverage map: 97 % of
covered pixels fall inside their patch's polygon.

The bundle is **sufficient to re-run matching and global optimisation offline**,
without the Pi and without re-reading the raw images. That is why the npz carries
analysis images and masks, not just descriptors: acceptance here is *photometric*,
so descriptors alone could re-fit models but could not re-decide which pairs to
believe. `test_mosaic_pipeline.py` pins this down by re-running a pair decision
from a bundle alone.

```python
from eyevu_mosaic.core import bundle
b = bundle.read("Sessions/.../bundle")
b["cfg"], b["patches"], b["features"], b["records"], b["rotations"], b["tracks"]
```

**The run never crashes the device.** Any failure at any stage still writes a
bundle containing whatever was computed plus the traceback.

---

## Layout

```
eyevu_mosaic/
  core/                 pure functions — no camera, GPIO, network or hardcoded paths
    preprocess.py       masking, illumination flattening, CLAHE, quality metrics
    features.py         detection, RootSIFT, spatial spread, landmark hooks
    pairwise.py         shortlisting, matching, Hough seeding, PROSAC, ladder, direct align
    graph.py            edge acceptance, cycle consistency, MST, components
    globalopt.py        tracks, rotation averaging, bundle adjustment, outlier rejection
    compose.py          projection, tiled warping, median blending
    models.py           transform models and the DOF ladder
    bundle.py           session bundle read/write
  run_session.py        Pi entry point: orchestration, timing, error handling
  config.py             every threshold, with its measurement
  synth.py              synthetic session generator (ground truth)
  validate.py           score the pipeline against that ground truth
```

`core/` is import-clean for the laptop dashboard — it calls these exact
functions. Every tunable lives in `config.py` and the config used is serialised
into the bundle.

Tests (`pytest test_mosaic_*.py`, 79 tests, no hardware):

* `test_mosaic_models.py` — round-trip, composition, gauge invariance, each ladder rung
* `test_mosaic_graph.py` — cycle consistency, components, MST, rotation averaging, tracks
* `test_mosaic_pipeline.py` — preprocessing, projection, bundle round-trip, failure handling

---

## Not done

**Learned keypoint detectors.** GLAMpoints (ICCV 2019) and SuperRetina (ECCV
2022) genuinely outperform SIFT on retinal imagery and are the accuracy ceiling
here — the 0.39 edge recall is a SIFT limitation, not a pipeline one. They are
out of scope for this deployment (no PyTorch, no ONNX, no GPU, 512 MB). The
`features.Detector` interface exists precisely so one can be dropped in:
implement `detect_and_describe(image, mask) -> (keypoints, descriptors)` and pass
it to `features.detect`. Nothing else changes.

**`cap.py` runs this package** — `o` in the live viewer calls
`build_session_mosaic`, which now runs `eyevu_mosaic` and **falls back to the
old `mosaic.py`** if the package will not import. That fallback is not
defensive padding: the Pi installer's pip block for numpy/scipy/scikit-image is
commented out ([`install_pi_deps.sh:52`](../install_pi_deps.sh#L52)), so a device
may genuinely have neither. Accordingly:

* **scipy** is imported lazily and guarded. Without it the pipeline still
  registers, still reconciles the graph by rotation averaging (pure numpy), and
  still composites — it skips bundle adjustment and says so in the bundle.
* **scikit-image** is already lazy. Without it the direct-alignment fallback is
  unavailable, so there are fewer edges; the feature path is unaffected.

Verified by blocking both imports: a 12-capture session still produced mosaics,
with 8 accepted pairs instead of 11 and groups of [4, 3, 2] instead of [5, 4, 2].

`o` shows **one** window — the mosaic, with `b` toggling patch boundaries and
`n` cycling groups when the session did not fully connect. Not the keypoint
sheets or contact sheets; `x` and `v` still do those. The old module's session
browser and keypoint-sheet helpers are still used by `cap.py` and by
`try_mosaic.py`, so `mosaic.py` stays.

> **`o` is now slow enough to plan around.** It was ~1–2 s; measured on the dev
> laptop it is 8 s for 12 captures and 47 s for 40, and the Pi is several times
> slower again. That is the price of the direct-alignment fallback, which is
> what took placement from 21 % to 49 %. Use it between patients, not with one
> at the scope. `direct_align=False` restores roughly the old speed at roughly
> the old connectivity.

`try_mosaic.py` **does** run this package now — it is the interactive tuning
harness, and it defaults to `eyevu_mosaic` with `--old` kept so the previous
pipeline can still be run on the same session for comparison. Alongside the
captures, keypoints and mosaic it also shows the coverage map, the model level
and photometric score of every accepted pair, and a tally of why the rejected
ones were rejected. Press **`b`** in any window to toggle the per-patch boundary
overlay on every mosaic at once; both variants are cropped to the same box so
the picture does not shift as you flip between them. Its picker is a
**multi-select list** — ctrl/shift-click several sessions and they are mosaicked
together, with each session's patient shown alongside its capture count.

## Patients

One patient has many sessions, and sessions of the same eye combine well — which
is exactly why they must not be combined across patients. `cap.PATIENT_ID`
(default `"0"`, override with `EYEVU_PATIENT` or `cap.set_patient()`) is stamped
into each session folder as `session.json` when its first capture is saved.

**Sessions recorded before this field existed have no `session.json` and read as
patient `"0"`**, which is the correct answer for everything captured so far — so
nothing needs backfilling and no existing path changes.

* `cap.sessions_for_patient(pid)` — every session folder for one patient
* `cap.build_patient_mosaic(pid)` — mosaic all of them together, bound to
  **`O`** (shift-o) in the live viewer, next to `o` for the current session alone

Session folders keep their flat `session_<stamp>` layout; the patient is
metadata inside them rather than a directory level, so existing sessions,
transfers and tooling are unaffected.

---

## References

* Stewart, Tsai & Roysam, *The dual-bootstrap ICP algorithm*, IEEE TMI 2003 — model escalation
* Can, Stewart, Roysam & Tanenbaum, PAMI 2002 — the 12-parameter quadratic
* Cattin, Bay, Van Gool & Székely, MICCAI 2006 — local features in avascular regions
* Brown & Lowe, IJCV 2007 — match graph, Hough seeding, rotation-model BA
* Lowe, IJCV 2004 — similarity-space Hough voting
* Hartley, Trumpf, Dai & Li, *Rotation averaging*, IJCV 2013
* Chum & Matas, *Matching with PROSAC*, CVPR 2005
* Padfield, *Masked object registration in the Fourier domain*, IEEE TIP 2012
* Hernández-Matas et al., *REMPE*, IEEE JBHI 2020, and the FIRE benchmark
