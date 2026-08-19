# Architecture

Three levels of zoom, then a call graph generated from the code.

## Snapshot

The tree was being edited while this was written. Every line number below refers
to this snapshot:

| File | Lines | mtime | SHA-256 (first 12) |
|---|---|---|---|
| `cap.py` | 5 459 | 19/08 21:28:41 | `D475A0ACD86E` |
| `mosaic.py` | 993 | 19/08 20:46:10 | `6C0F94DBC9B5` |
| `guidance.py` | 501 | 15/08 16:51:16 | `5617DF0931AE` |
| `receiver.py` | 401 | 19/08 21:14:49 | `EA56F23EFB21` |
| `try_mosaic.py` | 375 | 19/08 21:25:29 | `843B853542BE` |
| `eyevu_mosaic/config.py` | 334 | 19/08 20:10:55 | `E74C2FEADB12` |
| `eyevu_mosaic/run_session.py` | 369 | 19/08 21:22:12 | `467AA8BA60A7` |
| `eyevu_mosaic/synth.py` | 341 | 19/08 21:35:37 | `B9D906D1F496` |
| `eyevu_mosaic/core/preprocess.py` | 293 | 19/08 19:17:34 | `4194687614EA` |
| `eyevu_mosaic/core/features.py` | 156 | 19/08 21:11:35 | `5E1CF83E7AA5` |
| `eyevu_mosaic/core/pairwise.py` | 646 | 19/08 21:11:35 | `56B4B352A42B` |
| `eyevu_mosaic/core/graph.py` | 230 | 19/08 18:18:49 | `5D9416E1E8DA` |
| `eyevu_mosaic/core/globalopt.py` | 512 | 19/08 20:29:03 | `E03598B89C62` |
| `eyevu_mosaic/core/compose.py` | 338 | 19/08 20:09:55 | `E38AA7BE5A30` |
| `eyevu_mosaic/core/models.py` | 321 | 19/08 17:58:53 | `D061BC045765` |
| `eyevu_mosaic/core/bundle.py` | 302 | 19/08 21:21:58 | `4122733A6401` |

`features.py` and `pairwise.py` at 21:11:35 are the dead-code commenting from
[REPO_MAP.md §5.1](REPO_MAP.md). `cap.py`, `run_session.py`, `bundle.py`,
`synth.py` were changed by you after that, adding the patient tier — which is
included below.

All sizes quoted are **measured from this working copy**, not estimated.

---

# Level 1 — system

11 nodes. Solid arrows carry image data; dashed arrows carry only geometry or
metadata. Sizes are medians measured over the 142 extracts and 6 bundles present.

```mermaid
graph TB
  SENSOR["Picamera2 sensor<br/>1280x960 RGB"]
  LIVE["cap.streaming_mode<br/>live loop, 480x640 display"]
  DET["cap.detect_pupil<br/>+ cap.redeye_extract"]
  CAPTURE["cap.capture_flash_pair<br/>dark + flash pair"]
  SESSION[("Sessions/session_*/<br/>redeye_NN_extract.png ~4.4 kB<br/>_mask.png ~1.5 kB<br/>session.json")]
  RECV["receiver.py<br/>HTTP :8000 on the laptop"]
  XFER[("Transfers/capture_*/<br/>flash 46 kB, dark 43 kB<br/>ambient 58 kB, both 60 kB")]
  MOSAIC["eyevu_mosaic.run_session.run<br/>1 session or N combined"]
  BUNDLE[("bundle/<br/>bundle.npz 458 kB<br/>bundle.json 197 kB")]
  OUT[("mosaic.png ~24 kB<br/>203x193 typical<br/>+ coverage, outlined, preview")]
  VIEW["cap.view_session_mosaic<br/>or try_mosaic.py"]

  SENSOR -->|"capture_array()<br/>(960,1280,3) uint8 RGB<br/>3.7 MB/frame"| LIVE
  LIVE -->|"resized (640,480,3) BGR"| DET
  DET -.->|"centre+radius as frame fractions"| CAPTURE
  SENSOR -->|"dark + flash arrays"| CAPTURE
  DET -->|"extract + mask, uint8<br/>mask area median 941 px"| SESSION
  CAPTURE -->|"redeye extract + mask"| SESSION
  CAPTURE -->|"HTTP POST raw JPEG/PNG"| RECV
  RECV --> XFER
  SESSION -->|"list of (path, bgr, mask)"| MOSAIC
  MOSAIC --> BUNDLE
  MOSAIC --> OUT
  OUT --> VIEW
```

**What crosses each boundary.**

| Boundary | Payload | Format and size |
|---|---|---|
| sensor → live loop | one frame | `(960, 1280, 3)` uint8 RGB in memory, 3.69 MB; rotated by `LIVE_ROTATION=1` ([cap.py:175](../cap.py#L175)) |
| live loop → detection | display copy | resized to `DISPLAY_W=480 x DISPLAY_H=640` ([cap.py:5119-5120](../cap.py#L5119-L5120)) |
| detection → session | extract + mask | two PNGs per capture, median 4.4 kB and 1.5 kB |
| Pi → laptop | capture folder | HTTP POST to `192.168.137.1:8000` ([cap.py:139](../cap.py#L139)) |
| session → mosaic | patches | `load_sessions` returns `[(path, bgr, mask)]` ([run_session.py:116](../eyevu_mosaic/run_session.py#L116)) |
| mosaic → bundle | everything | `bundle.npz` 458 kB for 10 patches, `bundle.json` 197 kB |

**Two facts worth carrying into tuning.**

*Extracts arrive at two resolutions, sometimes in the same session.* Measured
across `Sessions/`:

| session | 640x480 | 1280x960 |
|---|---|---|
| `..._113948` | 11 | 2 |
| `..._115547` | 12 | 2 |
| `..._214652` | 38 | 2 |
| `..._223238` | 0 | 12 |
| `..._224337` | 0 | 10 |

The two `save_session_frame` call sites differ: [cap.py:5422](../cap.py#L5422)
passes a full-resolution mask (its comment says so), while
[cap.py:5546](../cap.py#L5546) passes the display-sized `meas["mask"]`. The two
newest sessions are uniformly full-resolution; I did not trace exactly when that
changed. `preprocess.analysis_scale` is what normalises the difference — see
Level 2c.

*The mosaic reads the session folder, not the capture folder.* `Transfers/` is a
diagnostic sink. Nothing in `eyevu_mosaic` ever opens it.

---

# Level 2 — subsystems

Dataflow, not call order. Parameter entry points are marked `⚙`.

## 2a — live pupil detection, `cap.detect_pupil`

```mermaid
graph LR
  IMG["ambient BGR<br/>(640,480,3) uint8"]
  GREEN["green = img[:,:,1]<br/>(640,480) uint8"]
  REFLEX["cap._find_corneal_reflex<br/>cap.py:1565"]
  INP["cap._inpaint_reflections<br/>cap.py:1521"]
  RAYS["cap._ray_edges<br/>cap.py:1792"]
  CIRC["cap._fit_circle_robust<br/>cap.py:1939"]
  RIDGE["cap._ray_ridges<br/>cap.py:1746"]
  CONC["cap._fit_concentric<br/>cap.py:1802"]
  OUT["overlay dicts<br/>+ _LAST_PUPIL, _LAST_CONF"]
  COVER["cap.load_cover_mask<br/>cap.py:742"]

  IMG --> GREEN
  GREEN --> REFLEX
  REFLEX -->|"(x, y, r, mask)<br/>anchor at pupil EDGE"| INP
  GREEN --> INP
  INP -->|"GaussianBlur 7x7<br/>'work' uint8"| RAYS
  COVER -.->|"uint8 mask, rays inside are dropped"| RAYS
  RAYS -->|"angles, radii -> (N,2) float32 pts"| CIRC
  CIRC -->|"(cx, cy, r); loop x _RAD_RECENTRE_ITERS"| RAYS
  CIRC --> RIDGE
  RIDGE -->|"iris pts (M,2)"| CONC
  CIRC --> CONC
  CONC -->|"adopted only if iris_cover >= _IRIS_MIN_COVER"| OUT
  P1{{"⚙ _SW_MIN_R_FRAC, _SW_MAX_R_FRAC<br/>cap.py:2033-2034"}} -.-> RAYS
  P2{{"⚙ _RAD_RECENTRE_ITERS<br/>cap.py:2083"}} -.-> CIRC
  P3{{"⚙ _IRIS_RP_MIN, _IRIS_SEARCH_MAX<br/>_IRIS_MIN_COVER  cap.py:2124-2129"}} -.-> CONC
```

The recentring loop is the part to understand: the corneal reflex sits at the
pupil **edge**, so the first centre is wrong by roughly a pupil radius. Rays are
recast from each new circle centre until the centre stops moving by more than
2 px ([cap.py:2095](../cap.py#L2095)). A circle is fitted rather than an ellipse
because a circle is stable on a partial arc — the comment at
[cap.py:2071-2075](../cap.py#L2071-L2075) says the ellipse version "ran away into
the bright glow".

## 2b — red-eye extraction, `cap.redeye_extract`

```mermaid
graph LR
  F["flash BGR int16"]
  D["dark BGR int16"]
  A["ambient BGR<br/>optional"]
  RS["redshift =<br/>(fR-dR) - (fB-dB)"]
  ROI["circular ROI<br/>r = _REDEYE_ROI_MULT x radius"]
  OTSU["Otsu over redshift<br/>inside ROI"]
  SEL["sel = redshift>=floor<br/>AND warm AND roi"]
  SPEC["specular exclusion<br/>threshold + reflex + dilate"]
  GROUP["cap.select_redeye_group<br/>cap.py:2614"]
  RES["RedeyeResult<br/>.extract .mask .overlay .valid"]

  F --> RS
  D --> RS
  A -.->|"aluma < _REDEYE_AMBIENT_DARK<br/>adds _REDEYE_AMBIENT_BOOST"| RS
  A -.->|"strict_ambient: hard veto<br/>Otsu on ambient luma"| SEL
  RS --> OTSU
  ROI --> OTSU
  OTSU -->|"floor = max(_REDEYE_MIN_SHIFT, otsu)"| SEL
  RS --> SEL
  SEL -->|"uint8 mask x255"| SPEC
  SPEC -->|"MORPH_OPEN 5x5"| GROUP
  GROUP --> RES
  P1{{"⚙ _REDEYE_ROI_MULT<br/>_REDEYE_ROI_MIN_FRAC  cap.py:2747-2748"}} -.-> ROI
  P2{{"⚙ _REDEYE_MIN_SHIFT  cap.py:2783"}} -.-> OTSU
  P3{{"⚙ _INPAINT_BRIGHT_THRESH<br/>_REDEYE_SPECULAR_DILATE  cap.py:2795-2800"}} -.-> SPEC
```

Both thresholds here are **adaptive with a fixed floor** — Otsu over the ROI, but
never below `_REDEYE_MIN_SHIFT`. Speculars are *excluded, not inpainted*
([cap.py:2793](../cap.py#L2793)), which is why `eyevu_mosaic`'s own
`specular_mask` usually finds nothing on real captures.

## 2c — mosaic preprocessing, `preprocess.prepare`

```mermaid
graph LR
  IN["bgr (H,W,3) uint8<br/>mask (H,W) uint8"]
  BB["content_bbox<br/>preprocess.py:59"]
  SC["analysis_scale<br/>preprocess.py:69"]
  CROP["crop + INTER_CUBIC resize<br/>to (nh, nw)"]
  GREEN["green = up[:,:,1]"]
  SPECM["specular_mask<br/>preprocess.py:186"]
  QUAL["quality_metrics<br/>preprocess.py:204"]
  ERODE["cv2.erode<br/>elliptical, e px"]
  FLAT["flatten_illumination<br/>preprocess.py:138"]
  NATIVE["native_bgr / native_mask<br/>native_weight = distanceTransform"]
  P["Patch<br/>.image .mask .S .focal .radius"]

  IN --> BB
  BB -->|"(x0,y0,w,h) native px"| CROP
  SC -->|"s = upsample x ref_width / W<br/>= 3.0 at 480 wide, 1.5 at 960"| CROP
  CROP --> GREEN
  CROP --> SPECM
  SPECM -->|"umask[spec>0] = 0"| QUAL
  GREEN --> QUAL
  QUAL -->|"accepted / reason"| P
  QUAL --> ERODE
  ERODE -->|"mask, analysis px"| FLAT
  GREEN --> FLAT
  FLAT -->|"poly order 2, divide, then CLAHE"| P
  CROP --> NATIVE
  NATIVE --> P
  C1{{"⚙ analysis_upsample 3.0<br/>ref_frame_width 480<br/>analysis_pad_px 6.0<br/>config.py:71-73"}} -.-> SC
  C2{{"⚙ erode_frac_of_radius 0.05<br/>erode_min_px 2, erode_max_px 8<br/>config.py:88-90"}} -.-> ERODE
  C3{{"⚙ flatten_mode 'poly'<br/>flatten_poly_order 2<br/>clahe_clip 5.0, clahe_tile 8<br/>config.py:107-113"}} -.-> FLAT
  C4{{"⚙ min_mask_area_px 150<br/>min_focus_lapvar 3.0<br/>max_saturated_frac 0.35<br/>config.py:126-131"}} -.-> QUAL
```

Two things to notice. `analysis_scale` depends on **frame width only**
([preprocess.py:75](../eyevu_mosaic/core/preprocess.py#L75)) — I verified this on
a real bundle: `native_bgr_0` is `(93,110,3)` and `img_0` is `(140,165)`, a factor
of 1.505, which is `3.0 x 480 / 960`. And the quality gate converts back to
**native** px before thresholding
([preprocess.py:220](../eyevu_mosaic/core/preprocess.py#L220)), so the area gate
does not silently rescale by `s²`.

## 2d — pairwise matching, `pairwise.match_pair`

```mermaid
graph TB
  FA["Features A<br/>xy (N,2) f32, desc (N,128) f32"]
  FB["Features B"]
  MD["match_descriptors<br/>pairwise.py:99"]
  HS["hough_seed<br/>pairwise.py:142"]
  LAD["fit_ladder<br/>pairwise.py:361"]
  PR["prosac_rotation<br/>pairwise.py:222"]
  NCC["ncc_verify<br/>pairwise.py:482"]
  DIR["_maybe_direct -> direct_align<br/>pairwise.py:738 / 525"]
  REC["record dict<br/>accepted, level, ncc, matches"]

  FA --> MD
  FB --> MD
  MD -->|"ia, ib, dist sorted ASCENDING<br/>BFMatcher NORM_L2, mutual NN"| HS
  MD --> LAD
  HS -->|"seed_idx: winning 4-D bin"| LAD
  LAD --> PR
  PR -->|"batched 2-pt Kabsch<br/>(T,3,3) SVD, chunks of 512"| LAD
  LAD -->|"estimate + inlier mask<br/>+ est_l0 always"| NCC
  NCC -->|"ok, ncc, overlap_px"| REC
  MD -.->|"< min_putative"| DIR
  LAD -.->|"no L0 consensus"| DIR
  NCC -.->|"ncc < ncc_min"| DIR
  DIR --> REC
  C1{{"⚙ lowe_ratio 0.98, mutual_nn True<br/>min_putative 4  config.py:176-180"}} -.-> MD
  C2{{"⚙ hough_bins_xy 0.25, _scale 2.0<br/>_theta_deg 30, min_votes 3<br/>config.py:187-190"}} -.-> HS
  C3{{"⚙ ransac_reproj_px 6.0<br/>ransac_max_iters 1500<br/>prosac_growth 0.25  config.py:194-198"}} -.-> PR
  C4{{"⚙ ncc_min 0.50<br/>min_overlap_px 400<br/>min_overlap_frac 0.10  config.py:244-246"}} -.-> NCC
  C5{{"⚙ direct_ncc_min 0.66<br/>direct_screen_ncc 0.42<br/>direct_roll_range_deg 8.0  config.py:211-222"}} -.-> DIR
```

`ncc_verify` is the decisive gate — the inlier count is not. Note the three
separate routes into `_maybe_direct`: too few putatives, no L0 consensus, and a
failed photometric check. A direct edge carries `matches = None`, which is
exactly how `globalopt` later decides it needs an explicit rotation prior.

## 2e — graph and global optimisation

```mermaid
graph LR
  RECS["records[]<br/>accepted pairs"]
  BUILD["graph.build<br/>graph.py:40"]
  CYC["graph.cycle_filter<br/>graph.py:72"]
  COMP["graph.components<br/>graph.py:189"]
  MST["graph.max_spanning_tree<br/>graph.py:210"]
  INIT["graph.initial_rotations<br/>graph.py:231"]
  RA["globalopt.rotation_average<br/>iters=30"]
  BA["globalopt.bundle_adjust<br/>scipy least_squares"]
  REJ["globalopt._reject_outliers"]
  G["G: {patch -> 3x3 rotation}"]

  RECS --> BUILD
  BUILD -->|"edges both directions<br/>R = so3_exp(est_l0.rotvec)<br/>w = n_inliers x ncc / (1+resid)"| CYC
  CYC -->|"drops weakest edge of any<br/>triangle over threshold, one per pass"| COMP
  COMP -->|"components, largest first"| MST
  MST -->|"Kruskal on w"| INIT
  INIT -->|"G_j = G_i R_ij^T from highest-degree root"| RA
  RA --> BA
  BA -->|"3 params/patch, one fixed for gauge<br/>Huber, jac_sparsity"| REJ
  REJ -->|"drop worst, unless it is a bridge"| RA
  REJ --> G
  C1{{"⚙ cycle_max_error_px 18.0<br/>cycle_check True  config.py:264-265"}} -.-> CYC
  C2{{"⚙ ba_huber_delta 4.0, ba_max_iters 60<br/>ba_max_tracks 400  config.py:273-285"}} -.-> BA
  C3{{"⚙ global_outlier_max_px 25.0<br/>_max_drops 6, _rounds 2<br/>config.py:307-309"}} -.-> REJ
```

Cycle error is scored in **analysis pixels, not degrees**
([graph.py:183](../eyevu_mosaic/core/graph.py#L183)): pointing error is weighted
by the focal (~3600 px) and roll only by the patch radius (~70 px). The
solve/reject/re-solve order matters — rejection runs *after* the solve because
rotation averaging hides a bad edge by spreading its error
([globalopt.py:435-441](../eyevu_mosaic/core/globalopt.py#L435-L441)).

## 2f — compositing, `compose.composite`

```mermaid
graph LR
  P["patches + G"]
  PLAN["plan_canvas<br/>compose.py:117"]
  GAIN["exposure_gains<br/>compose.py:152"]
  BOX["_patch_output_bbox<br/>compose.py:178"]
  TILE["tile loop<br/>tile_px x tile_px"]
  SAMP["_sample -> cv2.remap<br/>compose.py:189"]
  MED["_weighted_median<br/>compose.py:215"]
  OUT["mosaic (H,W,3) uint8<br/>coverage (H,W) uint16"]

  P --> PLAN
  PLAN -->|"Projection(centre, f_out, mode)<br/>+ canvas W,H"| BOX
  P --> GAIN
  GAIN -->|"ref / patch mean, per patch"| SAMP
  BOX -->|"which patches touch which tile"| TILE
  TILE --> SAMP
  SAMP -->|"samples native_bgr DIRECTLY<br/>one resampling, not two"| MED
  SAMP -->|"native_weight = distanceTransform<br/>0 at the rim"| MED
  MED --> OUT
  C1{{"⚙ projection 'azimuthal_equidistant'<br/>output_scale 1.0, output_max_px 4000<br/>output_margin_frac 0.05  config.py:316-319"}} -.-> PLAN
  C2{{"⚙ tile_px 512, blend_mode<br/>'weighted_median'  config.py:320-321"}} -.-> TILE
  C3{{"⚙ exposure_normalise True<br/>config.py:325"}} -.-> GAIN
```

`_sample` remaps from `patch.native_bgr` — the **native crop**, not the analysis
image — so output pixels come from source pixels with one resampling
([compose.py:203](../eyevu_mosaic/core/compose.py#L203)). That is why
`analysis_upsample` affects matching accuracy but not output sharpness.

---

# Level 3 — sequence

## 3a — the live capture loop

The stage machine in `streaming_mode` ([cap.py:5070-5975](../cap.py#L5070-L5975)).
Stages come from `guidance` ([guidance.py:48-54](../guidance.py#L48-L54)).

```mermaid
sequenceDiagram
  participant Op as Operator
  participant SM as streaming_mode
  participant Cam as Picamera2
  participant Det as detect_pupil
  participant Sw as ApproachState / GazeScan
  participant Disk as Sessions/

  SM->>SM: _stage = START_STAGE (cap.py:5151)
  loop every frame
    Cam-->>SM: capture_array()
    SM->>Det: detect_pupil(gray, live=True)
    alt no reflex and no coarse seed
      Det-->>SM: [] (cap.py:2068) - early exit, frame skipped
    else pupil found
      Det-->>SM: overlays, _LAST_PUPIL, _LAST_CONF
    end
    alt _stage == FIND_COVER
      SM->>SM: CoverTracker; on success -> CENTRE (cap.py:5322)
    else _stage == CENTRE
      SM->>Sw: ApproachState(side, centre, radius) (cap.py:5361)
      SM->>SM: _stage = APPROACH (cap.py:5363)
    else _stage in (APPROACH, TARGET)
      SM->>Sw: submit(measurement)
      opt _sweep.worth_saving(meas)
        SM->>Disk: save_session_frame(extract, mask) (cap.py:5422, 5546)
      end
      alt blink detected
        SM->>SM: BlinkDetector gates the measurement
      end
      alt _sweep.done and stage == APPROACH
        SM->>Sw: GazeScan(...) (cap.py:5582)
        SM->>SM: _stage = TARGET (cap.py:5585)
      else scan finished
        SM->>SM: _stage = CAPTURE (cap.py:5590)
      else lost
        SM->>SM: _stage = CENTRE (cap.py:5571) - restart
      end
    else _stage == CAPTURE
      SM->>Cam: capture_flash_pair(session=_session) (cap.py:5646)
      SM->>SM: _stage = START_STAGE (cap.py:5659)
    end
    Op-->>SM: keypress
    alt o
      SM->>SM: build_session_mosaic (cap.py:5867)
    else O
      SM->>SM: build_patient_mosaic (cap.py:5893)
    else s
      SM-->>Op: leave streaming (cap.py:5707)
    end
  end
```

`capture_flash_pair` itself is strictly ordered because the pupil must dilate
([cap.py:4847-4923](../cap.py#L4847-L4923)):

```mermaid
sequenceDiagram
  participant CF as capture_flash_pair
  participant LED as GPIO
  participant Cam as Picamera2
  participant RX as _redeye_capture
  participant Net as receiver.py

  CF->>Cam: apply_camera_settings(FLASH_GAIN)
  CF->>LED: flash_off + ambient_off
  CF->>CF: sleep(AUTOCAP_DILATE) - pupil dilates in the dark
  CF->>Cam: _drain_frames() then capture_array() -> dark
  CF->>LED: flash_on
  CF->>CF: sleep(FLASH_PRE_DELAY)
  CF->>Cam: _drain_frames() then capture_array() -> flash
  CF->>LED: flash_off
  CF->>RX: _redeye_capture(None, flash, dark, centre_frac, radius_frac)
  Note over RX: NO pupil detection here.<br/>The region comes from the last live frame.
  alt cv2 raises
    RX-->>CF: traceback printed, img = process_image(flash) (cap.py:4898-4903)
  end
  opt redeye.valid
    CF->>CF: save_session_redeye BEFORE transfer (cap.py:4907)
  end
  CF->>Net: transfer_capture(...)
```

The `_drain_frames` calls are load-bearing: picamera2 buffers frames, so the
first array after an LED change can still be a stale exposure
([cap.py:163-167](../cap.py#L163-L167)).

## 3b — a full mosaicing run

`run_session.run` ([run_session.py:145](../eyevu_mosaic/run_session.py#L145)).
Every stage is inside one `try`, and the `finally` always writes a bundle.

```mermaid
sequenceDiagram
  participant C as caller
  participant R as run_session.run
  participant P as preprocess
  participant F as features
  participant M as pairwise
  participant G as graph
  participant O as globalopt
  participant X as compose
  participant B as bundle

  C->>R: run(dirs, out_dir, cfg)
  R->>R: load_sessions(dirs)
  alt no captures
    R-->>B: RuntimeError -> finally writes bundle with traceback
  end
  R->>P: prepare(i, img, mask, cfg) per capture
  P-->>R: Patch(accepted, reason)
  Note over R: rejected patches are logged,<br/>never silently dropped
  R->>F: detect(patch, cfg, SIFTDetector)
  F-->>R: Features
  R->>R: reject patches with < min_keypoints (run_session.py:227)
  alt fewer than 2 usable
    R-->>B: RuntimeError "need >= 2 usable captures"
  end
  R->>M: shortlist(good, cfg)
  loop each candidate pair
    R->>M: match_pair(...)
    alt feature path fails
      M->>M: _maybe_direct -> direct_align
      Note over M: needs scikit-image;<br/>absent -> (None, nan, 0), degrades
    end
    M-->>R: record (accepted or with a reason)
  end
  R->>G: build -> cycle_filter -> components
  alt no component survived
    R-->>B: RuntimeError "no pair survived verification"
  end
  loop each component
    R->>G: max_spanning_tree + initial_rotations
    R->>O: refine(...)
    Note over O: scipy absent -> BA skipped,<br/>rotation averaging only, recorded in bundle
    O-->>R: G, stats
  end
  R->>X: composite per component
  X-->>R: mosaic, coverage, projection, info
  opt more than one component
    R->>R: WARNING, write mosaic_component_N.png each
  end
  R->>B: write(everything + timings + error)
  alt bundle write itself fails
    B-->>C: prints "BUNDLE WRITE FAILED" + traceback (run_session.py:390)
  end
  R-->>C: {ok, out_dir, components, timings, error}
```

**Early exits and degradations, all of them.**

| Where | Condition | Result |
|---|---|---|
| [run_session.py:208](../eyevu_mosaic/run_session.py#L208) | no captures found | RuntimeError, bundle still written |
| [run_session.py:227](../eyevu_mosaic/run_session.py#L227) | `len(f) < min_keypoints` | patch rejected with reason |
| [run_session.py:240](../eyevu_mosaic/run_session.py#L240) | fewer than 2 usable patches | RuntimeError, bundle still written |
| [pairwise.py:553](../eyevu_mosaic/core/pairwise.py#L553) | scikit-image missing | direct alignment unavailable, `nan` recorded, feature path unaffected |
| [globalopt.py:471](../eyevu_mosaic/core/globalopt.py#L471) | `bundle_adjust` off, or scipy missing | falls back to rotation averaging, method recorded |
| [globalopt.py:473](../eyevu_mosaic/core/globalopt.py#L473) | more than half of edges above L1 | rotation averaging instead of BA |
| [run_session.py:281](../eyevu_mosaic/run_session.py#L281) | no component survived | RuntimeError, bundle still written |
| [run_session.py:352](../eyevu_mosaic/run_session.py#L352) | >1 component | separate mosaic each, warning; **no relative placement invented** |
| [run_session.py:368](../eyevu_mosaic/run_session.py#L368) | `fail_soft` False | re-raises after writing |
| [run_session.py:390](../eyevu_mosaic/run_session.py#L390) | bundle write fails | traceback to stdout, does not raise |

On the `cap.py` side there is one more layer: `build_session_mosaic` catches
`ImportError` and falls back to `mosaic.py`
([cap.py:4247-4249](../cap.py#L4247-L4249)), and catches any other exception and
falls back again ([cap.py:4256-4260](../cap.py#L4256-L4260)) — even though `run`
is already fail-soft. `build_patient_mosaic` has the ImportError branch but
**no** legacy fallback ([cap.py:4316-4319](../cap.py#L4316-L4319)); it returns
`(None, {})`.

---

# Call graph — generated

**Method.** `pyan3` (installed cleanly via `pip install pyan3`) run as:

```
python -m pyan eyevu_mosaic/core/*.py eyevu_mosaic/run_session.py \
       eyevu_mosaic/config.py --root . --uses --no-defines --dot
```

Its DOT output (235 edges, 159 nodes) was converted to Mermaid by a script that
keeps only function-level edges and drops module bookkeeping nodes. After
filtering: **139 functions, 201 call edges**.

**What it misses.** Stated so you know how far to trust it:

- **Dynamic dispatch.** `features.detect` takes a `detector` argument and calls
  `det.detect_and_describe` ([features.py:130](../eyevu_mosaic/core/features.py#L130)).
  pyan resolves this to `SIFTDetector` only because that is the sole
  implementation; a learned detector dropped in later would not appear.
- **Callbacks.** `composite(progress=...)` and the `residuals`/`unpack` closures
  passed to `scipy.optimize.least_squares` are edges pyan cannot follow into
  scipy and back.
- **Library calls are not shown at all.** Every `cv2.*`, `numpy.*` and `scipy.*`
  call is invisible here, which hides where the real work happens — `cv2.SIFT`,
  `cv2.warpPerspective`, `cv2.remap`, `scipy.optimize.least_squares`.
- **Lazy imports inside functions** (`from .models import make_K` at
  [preprocess.py:346](../eyevu_mosaic/core/preprocess.py#L346),
  `from skimage.registration import ...` at
  [pairwise.py:552](../eyevu_mosaic/core/pairwise.py#L552)) are resolved for the
  first-party case and simply absent for the third-party one.
- **`cap.py` is not included.** pyan choked on nothing, but at 5 459 lines with a
  882-line `streaming_mode` the result was unreadable; the Level 3 diagrams cover
  that path by hand instead, with line references.

## Module level

Edge labels are the number of distinct function-to-function call edges.

Node ids are suffixed `_m` because `graph` is a Mermaid keyword and cannot be a
node id.

```mermaid
graph LR
  run_m["run_session"]
  graph_m["graph"]
  compose_m["compose"]
  features_m["features"]
  config_m["config"]
  globalopt_m["globalopt"]
  pairwise_m["pairwise"]
  preprocess_m["preprocess"]
  bundle_m["bundle"]
  models_m["models"]

  run_m -->|6| graph_m
  run_m -->|4| compose_m
  run_m -->|3| features_m
  run_m -->|3| config_m
  run_m -->|2| globalopt_m
  run_m -->|2| pairwise_m
  run_m -->|2| preprocess_m
  run_m -->|1| bundle_m
  pairwise_m -->|25| models_m
  globalopt_m -->|6| models_m
  pairwise_m -->|5| preprocess_m
  bundle_m -->|4| preprocess_m
  graph_m -->|3| models_m
  bundle_m -->|2| models_m
  bundle_m -->|2| compose_m
  compose_m -->|2| preprocess_m
  preprocess_m -->|2| models_m
  compose_m -->|1| features_m
  compose_m -->|1| models_m
  globalopt_m -->|1| preprocess_m
  pairwise_m -->|1| features_m
  bundle_m -->|1| config_m
  bundle_m -->|1| features_m
```

`models` is the sink: nothing in the package imports back into `pairwise` or
`run_session`. `preprocess` sits below almost everything because `patch_K` is
defined there rather than in `models`.

## The main path, function level

Only the edges `run_session.run` itself makes, in execution order.

```mermaid
graph TB
  RUN["run_session.run"]
  RUN --> LS["run_session.load_sessions"]
  RUN --> PREP["preprocess.prepare"]
  RUN --> SIFT["features.SIFTDetector"]
  RUN --> DET["features.detect"]
  RUN --> SL["pairwise.shortlist"]
  RUN --> MP["pairwise.match_pair"]
  RUN --> GB["graph.build"]
  RUN --> GC["graph.cycle_filter"]
  RUN --> GK["graph.components"]
  RUN --> GM["graph.max_spanning_tree"]
  RUN --> GI["graph.initial_rotations"]
  RUN --> RF["globalopt.refine"]
  RUN --> BT["globalopt.build_tracks"]
  RUN --> CP["compose.composite"]
  RUN --> PO["compose.patch_outlines"]
  RUN --> DO["compose.draw_outlines"]
  RUN --> BW["bundle.write"]
```

## Inside the two modules you will actually tune

```mermaid
graph TB
  MP["match_pair"] --> MD["match_descriptors"]
  MP --> HS["hough_seed"]
  MP --> FL["fit_ladder"]
  MP --> NV["ncc_verify"]
  MP --> MDIR["_maybe_direct"]
  FL --> PR["prosac_rotation"]
  FL --> FLV["_fit_level"]
  FL --> BL["_baseline"]
  PR --> PS["_prosac_samples"]
  MDIR --> DA["direct_align"]
  DA --> TR["_try_roll"]
  TR --> E2R["_euclid_to_rotation"]
  TR --> NV
  NV --> INV["_inv"]
```

```mermaid
graph TB
  RF["refine"] --> RA["rotation_average"]
  RF --> SO["_solve"]
  RF --> RJ["_reject_outliers"]
  RF --> ER["_edge_residuals"]
  SO --> BT["build_tracks"]
  SO --> BA["bundle_adjust"]
  BA --> UP["unpack (closure)"]
  BA --> RES["residuals (closure)"]
  RES --> SLB["_so3_log_batch"]
  UP --> SEB["_so3_exp_batch"]
  RJ --> IB["_is_bridge"]
  RJ --> ER
  RJ --> RA
  BT --> UF["_UF (union-find)"]
```

One thing the graph shows that the prose does not: `ncc_verify` is reached from
**two** places — the normal feature path via `match_pair`, and inside
`_try_roll`, once per roll hypothesis. That is why `direct_align` is the
expensive stage, and why `direct_screen_ncc` exists to cut the roll sweep short.

---

*Step 2 of 6. Next: `docs/PARAMETERS.md`.*
