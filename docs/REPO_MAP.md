# Repository map

Step 1 of the documentation set. Every claim below is from reading the code or
from a mechanical scan of it; where I inferred something, I say so.

**Method.** File sizes are `wc -l` equivalents. Live/dead status comes from an
AST scan (`ast.parse` on every file outside `EyeTracking/`, `models/`,
`Sessions/`, `Transfers/`, `__pycache__/`) that builds a name-resolved
"definition mentions name" graph and computes transitive reachability from
seeds: every module's top-level code, every `main`, every `test_*`, and every
decorated class. String constants that are valid identifiers count as
references, so dispatch-by-name is caught. Limits are in
[What this scan misses](#what-this-scan-misses).

---

## 1. What you actually run

Ordered by evidence of recent use. Timestamps are file mtimes in this working
copy; there is no `git` on PATH here, so I could not use commit history.

| # | Command | Where | Evidence it is live |
|---|---|---|---|
| 1 | `python cap.py` | Pi | `cap.py` mtime 19/08 20:46; the only module with GPIO + Picamera2 driving; memory notes the Pi is reachable over SSH and the live loop only runs there |
| 2 | `python receiver.py` | laptop | `Transfers/**/{dark,guidance,captures}.jpg` newest 15/08 22:45 — written by `receiver.py`'s upload handler during a live Pi session |
| 3 | `python try_mosaic.py` | laptop | `Sessions/*/bundle/*` newest 19/08 20:51; `keypoints.png` 19/08 19:35, which only the `--old` path writes |
| 4 | `python -m eyevu_mosaic.run_session <session>` | laptop or Pi | 7 `bundle.npz` under `Sessions/`; also invoked from `cap.build_session_mosaic` ([cap.py:4145](../cap.py#L4145)) |
| 5 | `pytest test_stages.py test_mosaic*.py test_guidance.py test_redeye.py` | laptop | `.pytest_cache/v/cache/nodeids` mtime 19/08 20:50 |

I ran the offline suite: **206 passed in 25.8 s**. The six entries in
`.pytest_cache/v/cache/lastfailed` are stale — none of them fail now.

### Entry points that exist but show no recent use

| Command | Status |
|---|---|
| `streamlit run pupillab/app.py` | Dormant. Its outputs (`debug_montage.jpg`, `annotated.jpg`, 33 each) all carry the 06/08 12:17 bulk-checkout timestamp, so none was produced by a run in this working copy. |
| `python pupillab/run_batch.py` | Dormant, same evidence. |
| `python pupillab/session.py` | Dormant. Writes `session_guidance.jpg`; **no such file exists** anywhere in the repo. |
| `python mosaic.py <session_dir>` | Works standalone ([mosaic.py:1119](../mosaic.py#L1119)) but superseded — see §3. |
| `python -m eyevu_mosaic.synth` / `.validate` | Test harnesses; no synthetic session directories present. |
| `python build_cover_calibration.py <img>` | One-off utility; `calibrate` inside `cap.py` does the same job on-device. |
| `python export_ritnet_onnx.py` | One-off; already run (`models/ritnet/` holds the exported files). |
| `python flash_photo*.py` (4 files) | Superseded by `cap.py` — see §3. |
| `python _pi.py`, `_pi_time.py`, `_pi_wait.py`, `pi_install_runner.py` | Ad-hoc SSH scratch tools. |
| `python "Optics Calc Tester.py"` | Unrelated to the vision pipeline (lens-geometry Tk GUI). |

---

## 2. Module map

Size is lines of code. "Live" means reachable from an entry point you run.

### 2.1 Live vision path

| Module | Lines | One-line purpose | Status |
|---|---|---|---|
| [`cap.py`](../cap.py) | 5 280 | The whole Pi application: GPIO/LED control, camera, live pupil detection, staged operator guidance, red-eye extraction, session storage, transfer. | **Live** — but see §4, it is 4 programs in one file |
| [`guidance.py`](../guidance.py) | 501 | Turns a pupil centre into an operator instruction ("move up and left"); owns the staged-session vocabulary. Imported by `cap.py`, `receiver.py`, `pupillab`. | Live |
| [`eyevu_mosaic/`](../eyevu_mosaic/) | 3 964 | Current post-session mosaicing package (13 files). | Live |
| [`mosaic.py`](../mosaic.py) | 993 | Previous stitcher. | **Superseded but still executing** — see §3.1 |
| [`ritnet_infer.py`](../ritnet_infer.py) | 347 | RITnet pupil segmentation via PyTorch. Preferred ML backend. | Live-conditional |
| [`ncnn_infer.py`](../ncnn_infer.py) | 233 | Same pipeline, ncnn executor, for boards with no torch build (Pi Zero W / ARMv6). | Live-conditional |
| [`receiver.py`](../receiver.py) | 401 | Laptop-side HTTP server receiving captures from the Pi; also re-runs detection, guidance and mosaicing as files land. | Live |
| [`try_mosaic.py`](../try_mosaic.py) | 281 | Interactive mosaic tuning harness; runs `eyevu_mosaic` by default, `--old` for `mosaic.py`. | Live |

`cap.py` picks its ML backend at import time by trying `ritnet_infer` then
`ncnn_infer` and calling `available()` on each ([cap.py:104-115](../cap.py#L104-L115));
whichever answers first wins, and `RITNET_AVAILABLE` is False if neither does.

### 2.2 `eyevu_mosaic/` in detail

| File | Lines | Purpose |
|---|---|---|
| [`config.py`](../eyevu_mosaic/config.py) | 334 | Every threshold in the package, with the measurement that chose it. |
| [`run_session.py`](../eyevu_mosaic/run_session.py) | 324 | The only IO module: orchestration, per-stage timing + peak RSS, fail-soft bundle write. |
| [`core/pairwise.py`](../eyevu_mosaic/core/pairwise.py) | 644 | Shortlisting, descriptor matching, Hough seeding, the model ladder, NCC gate, direct-alignment fallback. |
| [`core/globalopt.py`](../eyevu_mosaic/core/globalopt.py) | 512 | Feature tracks, rotation averaging, bundle adjustment on SO(3), global outlier rejection. |
| [`core/compose.py`](../eyevu_mosaic/core/compose.py) | 338 | Azimuthal-equidistant projection, tiled warping, exposure normalisation, weighted-median blend. |
| [`core/models.py`](../eyevu_mosaic/core/models.py) | 321 | Transform models `K R K⁻¹` → affine → homography → 12-param quadratic, and the DOF ladder. |
| [`core/preprocess.py`](../eyevu_mosaic/core/preprocess.py) | 293 | Masking, polynomial illumination flattening, CLAHE, quality gate. |
| [`core/bundle.py`](../eyevu_mosaic/core/bundle.py) | 284 | Session bundle read/write — the contract with the dashboard. |
| [`core/graph.py`](../eyevu_mosaic/core/graph.py) | 230 | Edge weighting, triangle cycle consistency, components, maximum spanning tree. |
| [`core/features.py`](../eyevu_mosaic/core/features.py) | 154 | SIFT + RootSIFT, spatial spread, pluggable `Detector`, landmark hooks. |
| [`synth.py`](../eyevu_mosaic/synth.py) | 330 | Synthetic session generator with known rotations (ground truth). |
| [`validate.py`](../eyevu_mosaic/validate.py) | 181 | Scores the pipeline against that ground truth. |
| `__init__.py` ×2 | 19 | Re-exports. |

### 2.3 Dormant — dev-machine experimentation surface

| Module | Lines | Purpose | Status |
|---|---|---|---|
| [`pupillab/`](../pupillab/) | 1 243 | Registry framework running several pupil detectors side-by-side on one capture, with a Streamlit dashboard and a headless batch runner. 14 files. | **Dormant.** Imports `cap`, `guidance`, `ritnet_infer`, `ncnn_infer`; nothing imports it. Its own `__init__` says "Nothing here is imported by the Pi code". |
| — `modules/ridge_baseline.py` | 79 | Wraps `cap.detect_pupil` + `cap.detect_redeye` unchanged as the reference detector. | Dormant |
| — `modules/bonteanu_hough.py` | 121 | Circular Hough pupil detector (Bonteanu et al., ISSCS 2019) as a literature baseline. | Dormant |
| — `modules/ritnet_seg.py` / `ncnn_seg.py` | 143 | Thin wrappers over the two RITnet backends. | Dormant |

`pupillab` is the **only live caller** of four `cap.py` functions
(`detect_and_annotate`, `detect_redeye`, `_fuse_pupil`, `coarse_locate`). If you
ever delete `pupillab`, those become dead too — see §5.2.

### 2.4 Superseded

| Module | Lines | Superseded by | Evidence |
|---|---|---|---|
| [`flash_photo.py`](../flash_photo.py) | 386 | `cap.py` | Its own `detect_pupil` ([flash_photo.py:156](../flash_photo.py#L156)) is a separate, older implementation; nothing imports the file |
| [`flash_photo_manual.py`](../flash_photo_manual.py) | 207 | `cap.py` | Manual-only subset of the above |
| [`flash_photo_manual_no_cv.py`](../flash_photo_manual_no_cv.py) | 122 | `cap.py` | Tk instead of OpenCV — a workaround for a Pi that no longer applies |
| [`flash_photo_manual_no_cv_no_gui.py`](../flash_photo_manual_no_cv_no_gui.py) | 94 | `cap.py` | Headless subset of the above |

These four are a clear descent chain — each strips something from its
predecessor. All predate `cap.py`'s streaming mode.

### 2.5 Utilities and scaffolding

| Module | Lines | Purpose |
|---|---|---|
| [`build_cover_calibration.py`](../build_cover_calibration.py) | 50 | Builds `calibration/led_cover_mask.png` from a calibration frame by calling `cap.build_cover_mask`. |
| [`export_ritnet_onnx.py`](../export_ritnet_onnx.py) | 64 | One-off RITnet → ONNX export at fixed 1×1×400×640, for `onnx2ncnn`. |
| [`_pi.py`](../_pi.py) / [`_pi_time.py`](../_pi_time.py) / [`_pi_wait.py`](../_pi_wait.py) | 127 | Ad-hoc paramiko SSH helpers for the Pi: run a command, time ncnn on-device, poll for completion. |
| [`pi_install_runner.py`](../pi_install_runner.py) | 52 | Streams a list of install commands to the Pi over SSH. |
| `install_pi_*.sh` (5 files) | — | Pi dependency and ncnn install/recovery scripts. |
| [`models/ritnet/`](../models/ritnet/) | — | Exported RITnet artefacts + a README pointing at the upstream weights. |

> `_pi.py`, `_pi_time.py`, `_pi_wait.py` and `pi_install_runner.py` carry the Pi's
> IP, username and password as plaintext module-level constants
> ([`_pi.py:10`](../_pi.py#L10), [`_pi_wait.py:3`](../_pi_wait.py#L3)). Flagging
> because it is a repository fact, not because you asked.

### 2.6 Tests — 2 443 lines

| File | Lines | Targets | In your recent runs? |
|---|---|---|---|
| [`test_stages.py`](../test_stages.py) | 481 | `cap` staged-session state machines + `guidance` | Yes (55 nodes) |
| [`test_mosaic.py`](../test_mosaic.py) | 365 | **old** `mosaic.py` | Yes (35 nodes) |
| [`test_mosaic_pipeline.py`](../test_mosaic_pipeline.py) | 358 | `eyevu_mosaic` end to end | Yes (32) |
| [`test_mosaic_graph.py`](../test_mosaic_graph.py) | 276 | `eyevu_mosaic.core.graph` / `globalopt` | Yes (24) |
| [`test_mosaic_models.py`](../test_mosaic_models.py) | 207 | `eyevu_mosaic.core.models` | Yes (31) |
| [`test_guidance.py`](../test_guidance.py) | 201 | `guidance` | Yes (32) |
| [`test_redeye.py`](../test_redeye.py) | 55 | `cap` red-eye extraction | Yes (3) |
| [`test_ncnn.py`](../test_ncnn.py) | 288 | `ncnn_infer` | **No** — needs the ncnn model files |
| [`test_pupil_detection.py`](../test_pupil_detection.py) | 212 | `cap.detect_and_annotate` over `Transfers/` | **No** — needs capture folders |

The package README claims "79 tests" for `test_mosaic_*.py`; the actual count is
**122** across the four files (35 + 32 + 24 + 31).

### 2.7 Vendored / unrelated / empty

| Path | Size | Note |
|---|---|---|
| [`EyeTracking/`](../EyeTracking/) | ~166 KB, 8 `.py` | Vendored upstream Orlosky eye tracker (jeoresearch). **Nothing in the repo imports it.** Reference material — `cap.py`'s detector is a reimplementation, not a call into this. |
| [`Optics Calc Tester.py`](../Optics%20Calc%20Tester.py) | 279 | Tk + matplotlib lens-geometry calculator. Unrelated to the vision pipeline. |
| [`blob_detection.py`](../blob_detection.py) | 225 | Standalone "volcano" (pit/rim ring) detector over R and B channels. **Nothing imports it**; it has no `__main__` guard either, so running it executes module-level code. See §3.2. |
| `Pupil Finder.py` | **0** | Empty file. |
| `err.txt`, `out.txt`, `piconnection.md` | **0** | Empty files. |

---

## 3. Abandoned earlier attempts

You said there would be some. There are five, plus a large commented-out block.

### 3.1 `mosaic.py` — superseded, but still running in one place

`eyevu_mosaic/README.md` states it replaces `mosaic.py`, and `cap.py`'s `o` key
now calls the new package. But **`receiver.py` still auto-mosaics with the old
one**: [receiver.py:153](../receiver.py#L153) does `import mosaic` and
`mosaic.build(...)` on every arriving `*_extract.png`, with
`SESSION_AUTO_MOSAIC = True` ([receiver.py:60](../receiver.py#L60)).

So on the laptop, a live session produces mosaics from the **old** pipeline,
while `try_mosaic.py` and `cap.py` produce them from the **new** one. Same
session, two different algorithms, two different `mosaic.png` files. This is the
single most likely thing to waste your time when tuning, because a `config.py`
change will not move the mosaic `receiver.py` writes.

`mosaic.py` is also still genuinely needed for three helpers `cap.py` calls —
`browse_session_shots` ([cap.py:4304](../cap.py#L4304)), `show_session_keypoints`
([cap.py:4370](../cap.py#L4370)), and the `_legacy_session_mosaic` import-failure
fallback ([cap.py:4260](../cap.py#L4260)) — and `try_mosaic.py` imports it at
module level ([try_mosaic.py:37](../try_mosaic.py#L37)), so it is not deletable
as it stands.

### 3.2 `blob_detection.py` — abandoned

225 lines implementing a "volcano" detector: radial intensity profiles looking
for a pit-and-rim ring structure in the R and B channels, with matplotlib 3-D
surface plots. No import anywhere, no `__main__` guard, and its parameters
(`MAX_RIM_R = 220`, `CENTRE_TOL = 55`) are in pixel units that match neither the
480×640 nor the 960×1280 capture path. Reads as an early exploration of finding
the pupil/red-eye by ring structure, before the radial-ridge fitter in `cap.py`.

### 3.3 The Swirski detector — already commented out

[cap.py:1005-1356](../cap.py#L1005-L1356), ~350 lines, is enclosed in a
`'''…'''` string literal, headed
`# ── Swirski detector (commented out — Orlosky is used instead; see below) ──`.

Worth knowing because it **defeats grep**: `_swirski_coarse`,
`_swirski_support`, `_swirski_fit_ellipse`, `_swirski_segment`,
`swirski_detect_pupil`, `swirski_detect` and `_swirski_all_candidates` all appear
to be defined *and called* if you search the file, but none of that code exists
at runtime. Live functions with the same names are defined later, from line 1361.

The immediate consequence: several live functions are called **only** from
inside this string, so they are dead. That is most of §5.1.

### 3.4 The `_orlosky_*` contour path — dead

[cap.py:1418-1504](../cap.py#L1418-L1504). `_orlosky_contours` →
`_orlosky_seed` → `_orlosky_darkest_area` / `_mask_outside_square`. Nothing calls
`_orlosky_contours`, so the whole subtree is unreachable. The module docstring
still advertises this method: "The Orlosky method detects the pupil in the
ambient image" ([cap.py:35](../cap.py#L35)). The live detector is the radial
gradient-ridge fitter `detect_pupil` ([cap.py:1975](../cap.py#L1975)).

### 3.5 `flash_photo*.py` — the four-step descent chain

See §2.4. Superseded wholesale by `cap.py`.

### 3.6 Two detectors for the same LED cover

`_find_led_cover_mask` ([cap.py:557](../cap.py#L557), dead) and
`build_cover_mask` ([cap.py:652](../cap.py#L652), live) both locate the LED cover
as the largest near-black connected component touching a frame border, then
dilate it. They differ in threshold constant, in whether they blur first, and in
whether they keep one component or all of them. The dead one was the live-frame
version; the live one is the calibration version. **`build_cover_mask` does not
call `_find_led_cover_mask`** despite the names — a case where reading the code
and reading the name give different answers.

---

## 4. One structural note on `cap.py`

At 5 280 lines it holds four separable programs, which is worth knowing before
you tune anything:

| Lines (approx.) | What lives there |
|---|---|
| 119-540 | Configuration constants, GPIO, camera settings |
| 542-1000 | LED-cover masking and calibration |
| 1005-1930 | Pupil detection — including the dead Swirski and Orlosky blocks |
| 1931-3320 | Red-eye extraction, blink detection, pupil-size reference |
| 3320-4060 | Staged-session state machines (`LitStage`, `ApproachState`, `GazeScan`) |
| 4063-4560 | Session storage, mosaic entry, transfer |
| 4557-5950 | Capture and the interactive streaming loop |
| 5952-6092 | `main` and the command prompt |

`streaming_mode` alone is **882 lines** ([cap.py:4899-5780](../cap.py#L4899-L5780)).

Note also that `cap.py`'s own program does **not** use
`detect_and_annotate`, `detect_both`, `coarse_locate` or `classify_detection` —
those exist for `receiver.py` and `pupillab`. The Pi path calls `detect_pupil`
directly ([cap.py:5120](../cap.py#L5120), [5291](../cap.py#L5291),
[3236](../cap.py#L3236)).

---

## 5. Dead code

You asked mid-task to comment out dead code, overriding the brief's "do not
modify any source file in this pass", and confirmed after seeing this list.
**§5.1 has been applied.** Nothing else was touched.

### 5.1 Verified dead — COMMENTED OUT 2026-08-19

No reference from any module, any test, or any identifier-valued string constant.
**470 lines across 19 functions in 5 files.** Each block is prefixed with:

```
# DEAD CODE - no caller anywhere in the repo (AST reachability scan, 2026-08-19).
# Commented out, not deleted.  See docs/REPO_MAP.md section 5.1.
```

Line numbers below are the **pre-edit** positions. Verification after the edit:
`py_compile` clean on all five files; `import cap, guidance, mosaic,
eyevu_mosaic, receiver` clean; **206/206 offline tests pass**; a full
`run_session` over `Sessions/session_20260815_224337` reproduced the previous run
exactly — same 1 879 keypoints, same 9 accepted pairs with identical inlier
counts and NCC values, same components `[6, 3]`, same BA RMS 3.71 → 1.48 px, same
output dimensions and covered-pixel counts.

| Location | Lines | Function | Why it is dead |
|---|---|---|---|
| [cap.py:557](../cap.py#L557) | 38 | `_find_led_cover_mask` | Only caller is `_apply_exclusions`, also dead |
| [cap.py:597](../cap.py#L597) | 24 | `_apply_exclusions` | Only caller is inside the commented-out Swirski string (§3.3) |
| [cap.py:1411](../cap.py#L1411) | 5 | `_enhance_contrast` | No caller |
| [cap.py:1418](../cap.py#L1418) | 17 | `_orlosky_seed` | Only caller is `_orlosky_contours`, dead |
| [cap.py:1437](../cap.py#L1437) | 19 | `_orlosky_darkest_area` | Only caller is `_orlosky_seed`, dead |
| [cap.py:1458](../cap.py#L1458) | 11 | `_mask_outside_square` | Only caller is `_orlosky_seed`, dead |
| [cap.py:1471](../cap.py#L1471) | 34 | `_orlosky_contours` | No caller (§3.4) |
| [cap.py:1599](../cap.py#L1599) | 44 | `_segment_pupil_at` | Only caller is inside the Swirski string |
| [cap.py:1645](../cap.py#L1645) | 34 | `_swirski_support` | Only caller is `_swirski_fit_ellipse`, dead |
| [cap.py:1681](../cap.py#L1681) | 43 | `_swirski_fit_ellipse` | Only caller is inside the Swirski string |
| [cap.py:1828](../cap.py#L1828) | 28 | `_refine_center` | No caller |
| [cap.py:1858](../cap.py#L1858) | 30 | `_fit_ellipse_robust` | No caller — the live fit is `_fit_circle_robust` ([cap.py:1915](../cap.py#L1915)) |
| [cap.py:4391](../cap.py#L4391) | 28 | `show_session_shots` | No caller; not bound to any key in `streaming_mode` |
| [cap.py:4757](../cap.py#L4757) | 21 | `_draw_detections` | No caller |
| [eyevu_mosaic/core/features.py:176](../eyevu_mosaic/core/features.py#L176) | 3 | `set_landmark` | No caller — see §5.4 |
| [eyevu_mosaic/core/pairwise.py:293](../eyevu_mosaic/core/pairwise.py#L293) | 64 | `prosac` | No caller — see §5.4 |
| [pupillab/context.py:241](../pupillab/context.py#L241) | 7 | `context_for_folder` | No caller |
| [Optics Calc Tester.py:54](../Optics%20Calc%20Tester.py#L54) | 9 | `lens_backward` | No caller |
| [Optics Calc Tester.py:64](../Optics%20Calc%20Tester.py#L64) | 11 | `calc_eye_pos` | No caller |

Also fully dead, but already inert and therefore **left alone**:
**[cap.py:1005-1356](../cap.py#L1005-L1356)** (pre-edit numbering), the ~350-line
Swirski string. There is nothing to comment out; the open question is whether to
delete it, which I have not done.

One side effect worth knowing. The section header above `set_landmark`
([features.py:161-174](../eyevu_mosaic/core/features.py#L161-L174)) still reads
"The mechanism is here and is exercised by the tests" — and now sits directly
above a commented-out function. That claim was already wrong before the edit
(§5.4); the edit makes it conspicuous. I left the comment untouched because
rewriting prose was outside what you approved. Worth a one-line fix.

### 5.2 Reachable only from `pupillab` or `receiver.py` — do **not** touch

The scan flags these as unreachable from `cap.main`, but they have real callers.

| Function | Real caller |
|---|---|
| `cap.detect_redeye` ([1931](../cap.py#L1931)) | [pupillab/modules/ridge_baseline.py:59](../pupillab/modules/ridge_baseline.py#L59) |
| `cap._fuse_pupil` ([2272](../cap.py#L2272)) | [pupillab/modules/ridge_baseline.py:69](../pupillab/modules/ridge_baseline.py#L69) |
| `cap.detect_and_annotate` ([2306](../cap.py#L2306)) | [test_pupil_detection.py:32](../test_pupil_detection.py#L32) |
| `cap.coarse_locate` ([2364](../cap.py#L2364)) | [pupillab/app.py:144](../pupillab/app.py#L144), [pupillab/session.py:92](../pupillab/session.py#L92) |
| `cap.detect_both` ([2393](../cap.py#L2393)) | [receiver.py:181](../receiver.py#L181), [226](../receiver.py#L226) |
| `cap.classify_detection` ([2459](../cap.py#L2459)) | [receiver.py:182](../receiver.py#L182), [228](../receiver.py#L228) |

### 5.3 Scanner false positives — do **not** touch

Reached only through mechanisms an AST scan cannot see:

- `RidgeBaseline`, `BonteanuHough`, `RITnetSeg`, `NcnnSeg` — instantiated by the
  `@register` decorator in [pupillab/registry.py](../pupillab/registry.py).
- `UploadHandler.do_POST`, `UploadHandler.log_message` — called by
  `BaseHTTPRequestHandler`.
- The three `forward` methods in [ritnet_infer.py](../ritnet_infer.py) — called
  by `torch.nn.Module.__call__`.
- `BlinkDetector.suspect` ([cap.py:3091](../cap.py#L3091)), `LitStage.to_dwell`
  ([cap.py:3402](../cap.py#L3402)), `DetectionContext.restrict`
  ([pupillab/base.py:99](../pupillab/base.py#L99)),
  `MosaicConfig.to_json` ([config.py:357](../eyevu_mosaic/config.py#L357)) —
  these are method definitions on live classes. I have **not** individually
  verified that each is unused; treat them as unresolved, not as dead.

### 5.4 Two dead functions that contradict the documentation

These matter more than their line count, because they mean the pipeline does not
do what its README says.

**`prosac` is never called** ([pairwise.py:293-356](../eyevu_mosaic/core/pairwise.py#L293-L356)).
The module docstring lists "PROSAC" in its summary line and cites
"Chum & Matas, CVPR 2005 -- PROSAC" ([pairwise.py:22](../eyevu_mosaic/core/pairwise.py#L22)),
and the package README says the match stage runs "PROSAC". What
`fit_ladder` actually calls is `prosac_rotation`
([pairwise.py:386](../eyevu_mosaic/core/pairwise.py#L386)), a different function.
Levels 1-4 are fitted by `_fit_level` on the L0 inlier set with **no robust
estimator at all** ([pairwise.py:424](../eyevu_mosaic/core/pairwise.py#L424)).
I have not yet read `prosac_rotation` closely enough to say how much of PROSAC's
ordered sampling it retains — that goes in `ALGORITHMS.md`.

**`set_landmark` is never called and never tested**
([features.py:176](../eyevu_mosaic/core/features.py#L176)). The README describes
landmark injection as "wired and tested but unpopulated". The "unpopulated" half
is right. The "tested" half is not: the scan seeds every `test_*` function, and
`set_landmark` is still unreachable.

---

## What this scan misses

Stated so you can judge how much to trust §5:

- **Names are resolved globally, not per module.** A name defined in two files
  merges into one node. This can only make something look *more* reachable, so
  "unreachable" is a strong claim and "reachable" is a weak one.
- **Attribute access is untyped.** `x.foo()` counts as a reference to every
  `foo` in the repo.
- **Dynamic dispatch is invisible** except where the name appears as a bare
  string constant: `getattr(obj, name_from_config)`, plugin loading, and
  framework callbacks are not traced. §5.3 lists the cases I caught by hand.
- **Method-level reachability was not computed** — §5.1 covers module-level
  functions only. Methods inside live classes are unresolved.
- **`EyeTracking/`, `models/`, `Sessions/`, `Transfers/` were excluded.**
- **No git history.** `git` is not on PATH in this environment, so "superseded"
  judgements rest on code structure, imports and file mtimes, not commits.

---

*Step 1 of 6. The §5.1 dead-code edit has been applied and verified; steps 2-6
(ARCHITECTURE, PARAMETERS, ALGORITHMS, TOUR, FINDINGS) are not yet written.*
