"""Every tunable threshold in the mosaicing pipeline, in one place.

Nothing else in the package defines a magic number.  The exact `MosaicConfig`
used for a run is serialised into the session bundle, so a result can always be
traced back to the settings that produced it and re-run offline.

Where a default was chosen by MEASUREMENT rather than by taste, the measurement
is recorded beside it.  Those numbers come from the 141 real patches in
`Sessions/` (11 sessions, Aug 2026) and should not be changed casually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields


# ── the model escalation ladder ──
# Levels are (name, dof, min_correspondences).  A pair enters at L0 and is
# promoted only when BOTH an inlier-count gate and a residual-improvement gate
# are passed -- see `promote_gates` below and models.LADDER.
#
# The inlier gates in the original design brief (8 / 12 / 20 / 35) were written
# for ~640px apertures.  On the real patches they are unreachable: measured over
# a 14-capture session, TRUE pairs yielded 3-5 inliers and FALSE pairs 5-8.  The
# gates below are scaled to that reality.  L3/L4 remain implemented and tested
# but are effectively unreachable at current patch size; they exist for when
# capture moves to full sensor resolution.
@dataclass
class LadderConfig:
    enabled_max_level: int = 2      # highest level a pair may be promoted TO.
                                    # 2 = affine.  Raise to 4 once patches are
                                    # large enough for 20+ inliers to be normal.
    l1_min_inliers: int = 6         # rotation + shared focal
    l2_min_inliers: int = 10        # affine
    l3_min_inliers: int = 20        # homography
    l4_min_inliers: int = 35        # 12-param quadratic
    # Higher-DOF models ALWAYS fit better in-sample, so residual improvement
    # alone can never justify promotion.  Both gates must pass.
    residual_gain: float = 0.20     # median residual must drop by this fraction
    l4_min_baseline_px: float = 40.0  # L4 also needs a genuinely wide baseline


@dataclass
class MosaicConfig:
    # ── geometry / intrinsics ──────────────────────────────────────────────
    # There are NO calibrated intrinsics for this rig.  `focal_ref_px` is the
    # focal length in pixels at `ref_frame_width`, and it is a placeholder: at
    # the current patch size the projective terms of K R K^-1 go as (x/f)^2 ~
    # 3e-4, so L0 is numerically a Euclidean transform and the value barely
    # matters.  It starts to matter as patches grow; L1 refines it from data.
    focal_ref_px: float = 1200.0
    focal_is_calibrated: bool = False
    ref_frame_width: int = 480      # frame width the reference focal belongs to

    # ── analysis space ────────────────────────────────────────────────────
    # Detection, matching and optimisation all happen in "analysis space": each
    # patch cropped to its content and rescaled to a COMMON angular sampling.
    #
    # The scale factor depends only on the frame resolution, never on the
    # patch's own size:      s_i = analysis_upsample * ref_frame_width / W_i
    #
    # This matters.  Scaling each patch so its own bbox hits a fixed size (the
    # previous implementation) gives a 35px patch x6.86 and a 45px patch x5.33
    # -- ~29% of relative scale invented between two overlapping views.  A
    # similarity model hides it by spending a DOF on it; a rotation model
    # cannot.  Normalising by resolution keeps true relative scale at 1.0.
    # Measured on synthetic sessions: 3.0 and 5.0 recover the same number of
    # true pairs, and 2.0 clearly fewer.  3.0 is the cheaper of the two that
    # work, which matters on the Pi.
    analysis_upsample: float = 3.0
    analysis_pad_px: float = 6.0    # context kept around the patch, ANALYSIS px
    analysis_max_px: int = 900      # hard cap on the long edge, memory backstop

    # ── preprocessing ─────────────────────────────────────────────────────
    # Mask erosion is deliberately SMALL.  Measured over two real sessions,
    # eroding hard to keep keypoints off the illumination boundary costs real
    # registrations, because ~77% of all keypoints lie within 12 analysis-px of
    # the rim -- at this patch size "clean of the boundary" and "has any
    # features at all" are in direct conflict:
    #
    #     erode 5 analysis-px          6/6 true pairs recovered
    #     erode ~15 analysis-px        4/6
    #     erode ~30 analysis-px        0/6   (total collapse)
    #
    # The design brief's "erode by 5% of the aperture radius" would be ~4px
    # here, which is consistent with the top row.  Expressed directly:
    erode_frac_of_radius: float = 0.05   # of the mask's equivalent radius
    erode_min_px: int = 2                # ...but never less than this
    erode_max_px: int = 8                # ...and never more (see table above)

    # Illumination flattening.  "poly" divides out a low-order polynomial fit to
    # the illumination over the mask; the Gaussian modes high-pass instead.
    #
    # MEASURED, on synthetic sessions with ground-truth rotations: the median
    # NCC over pairs that genuinely overlap was
    #
    #     subtract, sigma 0.125 R    0.075        poly, order 2    0.93
    #     subtract, sigma 0.400 R    0.173
    #     subtract, sigma 0.600 R    0.213
    #     no flattening at all       0.244
    #
    # Every Gaussian setting is worse than doing nothing, because at ~15 native
    # px of patch radius the illumination and the anatomy occupy the same
    # spatial frequencies -- see the note in preprocess.flatten_illumination.
    # Keep "poly" unless patches get much larger.
    flatten_mode: str = "poly"           # "poly" | "subtract" | "divide" | "none"
    flatten_poly_order: int = 2          # vignetting is quadratic in position
    flatten_sigma_frac: float = 0.125    # Gaussian modes only
    # Measured: 5.0 recovered 18 solvable pairs of 36 against 14 at 2.5 and 8 at
    # 0.0 (CLAHE off).  Aggressive, but this is very low-contrast material.
    clahe_clip: float = 5.0
    clahe_tile: int = 8

    # Speculars are ALREADY excluded upstream by cap.redeye_extract (thresholded
    # and dilated out of the mask before the extract is written), so this is a
    # second line of defence for anything that survived, not the primary one.
    specular_thresh: int = 250           # near-saturation on the raw patch
    specular_dilate_px: int = 3

    # ── quality gate ──────────────────────────────────────────────────────
    # A patch failing any of these is FLAGGED and excluded from the graph, but
    # never silently dropped: the metric and the reason are always recorded.
    # Bounds come from the real distribution (median mask area 943px, range
    # 49-7484; the 49px and 7x9px outliers are what min_mask_area rejects).
    min_mask_area_px: int = 150
    max_mask_area_px: int = 60000
    min_focus_lapvar: float = 3.0        # variance of Laplacian inside the mask
    max_saturated_frac: float = 0.35
    min_mean_luma: float = 8.0
    max_mean_luma: float = 250.0

    # ── features ──────────────────────────────────────────────────────────
    # 600 keypoints (the brief's figure) is for a 640px aperture.  A ~225px
    # analysis patch realistically yields 30-150; the cap is a safety rail.
    max_keypoints: int = 300
    # Much looser than SIFT's 0.04 default: this is low-contrast tissue, and
    # measured, 0.01 recovers noticeably more true pairs than 0.02 or 0.04.
    # Below 0.01 the extra detections are noise and recall stops improving.
    sift_contrast_threshold: float = 0.01
    sift_edge_threshold: float = 12.0
    rootsift: bool = True
    # Spatial spread.  Raw response ranking piles keypoints onto whatever few
    # high-contrast structures a patch has; spreading forces coverage.
    # Measured: ANMS beat grid bucketing (16 vs 14 solvable pairs of 36).
    spread_mode: str = "anms"               # "anms" | "grid" | "none"
    spread_grid: int = 6                    # grid_n x grid_n buckets
    spread_per_bucket: int = 12
    min_keypoints: int = 6                  # fewer than this = unmatchable

    # ── pair shortlisting ─────────────────────────────────────────────────
    # With no gaze prior available, shortlisting is descriptor-driven only.
    # Below `shortlist_all_below_n` patches, all-pairs is cheap enough to just
    # do -- 25 patches is 300 pairs of ~100 keypoints, which is seconds.
    shortlist_all_below_n: int = 25
    shortlist_top_k: int = 12               # best-ranked candidates per patch
    shortlist_random_k: int = 4             # plus a random sample of the rest,
                                            # so unexpected overlaps stay findable
    shortlist_seed: int = 0
    global_desc_size: int = 16              # NxN downsampled masked thumbnail

    # ── pairwise matching ─────────────────────────────────────────────────
    # MUCH looser than the usual 0.7, and this is the single most consequential
    # threshold in the file.  The ratio test exists to make putative matches
    # trustworthy enough for a high-DOF model fitted by plain RANSAC.  Here the
    # model needs TWO correspondences, PROSAC orders its draws by descriptor
    # distance anyway, and acceptance is decided photometrically -- so a strict
    # ratio buys nothing and costs true correspondences this material cannot
    # spare.  Measured, solvable pairs (>= 3 GT-consistent matches) went
    #
    #     ratio 0.75   8%        ratio 0.92   33%
    #     ratio 0.85  19%        ratio 0.98   42%
    #
    # with precision staying at 1.00 throughout, because the NCC gate absorbs
    # the extra outliers exactly as intended.
    lowe_ratio: float = 0.98
    # Mutual NN costs nothing in recall here and roughly halves the putative set,
    # which halves PROSAC's work -- worth keeping on for the Pi.
    mutual_nn: bool = True
    min_putative: int = 4

    # Similarity-space Hough voting (Lowe 2004; Brown & Lowe 2007).  Each SIFT
    # match carries scale and orientation, so ONE correspondence hypothesises a
    # 4-DOF similarity.  Voting finds consensus from 3-4 matches where
    # positional RANSAC alone finds nothing -- which is this data's regime.
    hough_enabled: bool = True
    hough_bins_xy: float = 0.25             # bin width as a fraction of patch size
    hough_bins_scale: float = 2.0           # multiplicative bin factor
    hough_bins_theta_deg: float = 30.0
    hough_min_votes: int = 3

    # PROSAC: draw samples in ascending descriptor distance, which converges far
    # faster than plain RANSAC at low inlier ratios (Chum & Matas 2005).
    ransac_reproj_px: float = 6.0           # ANALYSIS px (~2 native px at x3)
    ransac_max_iters: int = 1500            # hypotheses are batched, so this is
                                            # a few vectorised calls, not a loop
    ransac_confidence: float = 0.995
    prosac_growth: float = 0.25

    # ── direct alignment fallback ─────────────────────────────────────────
    # Tried only when the feature path fails.  A true overlap here yields a
    # median of 2-3 correct descriptor matches; below three nothing can be
    # fitted, and measured edge recall from features alone is ~0.20 -- yet those
    # same pairs agree at NCC 0.8+ once warped together.  They are not
    # ambiguous, they are just undescribable.
    #
    # This is tractable ONLY because the model is 3-DOF: the search is two
    # translations and a roll.  An 8-DOF homography could not be searched this
    # way, so the reduced-DOF model pays off a second time here.
    direct_align: bool = True
    direct_roll_range_deg: float = 8.0
    direct_roll_step_deg: float = 2.0
    direct_downscale: float = 0.5           # FFT correlation resolution
    # Zero-roll screen: most candidate pairs do not overlap at all, and paying
    # for a full roll sweep to find that out is where the time goes.  Set well
    # below direct_ncc_min so the screen can never be what decides a pair.
    direct_screen_ncc: float = 0.42
    # Stricter than ncc_min, because a direct search optimises the very quantity
    # it is then judged on -- the same number does not mean the same thing.
    # Measured: every false edge the fallback produced scored 0.62-0.64, every
    # genuine one 0.68 or above, so the band is clean and 0.66 sits in it.
    direct_ncc_min: float = 0.66

    # ── edge acceptance ───────────────────────────────────────────────────
    # Inlier count is the WEAK test on this material.  Measured over a real
    # 14-capture session: true pairs 3-5 inliers, false pairs 5-8 -- the
    # distributions do not separate.  Photometric agreement DOES separate, with
    # no overlap at all (true 0.53-0.91, false <0.46; synthetic 0.79 vs 0.19).
    # So the inlier floor is set at the algebraic minimum plus one, and NCC is
    # what actually decides.
    min_inliers: int = 3
    # Deliberately near-zero.  With a permissive ratio test the putative set is
    # large and mostly wrong BY DESIGN, so inlier FRACTION is even less
    # discriminating than inlier count: a perfectly good pair with 4 inliers out
    # of 100 putative scores 0.04.  The old 0.25 gate silently rejected every
    # true pair once the ratio test was loosened.  Kept only as a guard against
    # pathological fits.
    min_inlier_frac: float = 0.02
    # Measured sweep of this gate, on synthetic sessions with ground truth:
    #   0.35 -> precision 0.89 (2 false pairs)     0.55 -> precision 1.00
    #   0.45 -> precision 1.00                     0.65 -> precision 1.00, recall down
    # 0.50 sits in the middle of the clean band and matches the value measured
    # independently on real sessions (true 0.53-0.91, false < 0.46).
    ncc_min: float = 0.50
    min_overlap_px: int = 400               # ANALYSIS px
    min_overlap_frac: float = 0.10          # of the smaller patch's mask
    max_rotation_deg: float = 60.0          # a sane bound on eye rotation/pair

    # ── match graph ───────────────────────────────────────────────────────
    # Cycle consistency: for every triangle (i,j,k), R_ij R_jk R_ki should be
    # identity.  Cheap, and it catches plausible-looking bad edges that inlier
    # count never will.  The single highest-value filter in the pipeline.
    #
    # The error is measured in IMAGE DISPLACEMENT (analysis px), not in degrees.
    # A rotation's three DOF affect the image by wildly different amounts here:
    # pointing error scales with the focal (~3600 px), roll only with the patch
    # radius (~70 px), so an angular threshold is dominated by roll -- which is
    # simultaneously the worst-observed and least consequential parameter.
    # Measured: on a session where every accepted edge was independently
    # verified correct, median edge error was 2.12 deg of which 2.12 was roll
    # and 0.030 pointing; a 2-degree gate threw away 3 of the 4 triangles, all
    # good.  In pixels those same edges sit at a median 3.4, while a genuinely
    # false edge lands orders of magnitude away.
    cycle_check: bool = True
    cycle_max_error_px: float = 18.0        # analysis px of loop closure
    cycle_min_triangles: int = 1            # edges in no triangle are untested,
                                            # not rejected -- they keep NCC only

    # ── global optimisation ───────────────────────────────────────────────
    bundle_adjust: bool = True
    ba_refine_focal: bool = False           # only meaningful once patches are
                                            # big enough for f to be observable
    ba_huber_delta: float = 4.0             # ANALYSIS px
    ba_max_iters: int = 60
    ba_min_track_len: int = 2
    # Convergence tolerances for a problem measured in PIXELS.  scipy's 1e-8
    # defaults chase precision orders of magnitude below the noise floor and
    # simply exhaust the evaluation budget.
    ba_ftol: float = 1e-6
    ba_xtol: float = 1e-6
    ba_gtol: float = 1e-6
    # Cost grows with track count (3 parameters and 2 residuals per observation
    # each), while the marginal value of the thousandth track is nil once the
    # rotations are already determined.  Longest tracks are kept first.
    ba_max_tracks: int = 400
    # If most edges settled ABOVE L1 the graph is not a pure rotation graph and
    # bundling it as one is wrong; fall back to rotation averaging plus per-edge
    # residual correction instead.
    ba_max_level_for_rotation_ba: int = 1
    # Feed every accepted edge into BA as a relative-rotation constraint, not
    # just the correspondences.  Edges found by direct alignment contribute no
    # correspondences at all, so without this a BA driven purely by reprojection
    # silently drifts away from them -- measured, it left them violated by 4-7 px
    # and the global pointing error 18x worse than on a feature-carried session.
    ba_use_edge_priors: bool = True
    # Global outlier rejection.  Cycle consistency only tests edges that lie in
    # a triangle, and these graphs are barely denser than trees -- 32 edges over
    # 30 patches gave 9 triangles -- so most edges are never cross-checked, and
    # one bad edge corrupts everything downstream of it.  Measured: two false
    # edges took a 30-patch session from ~1 px of median misalignment to 11 px.
    # An edge whose removal would disconnect the component is always kept.
    # Rejection runs AFTER the solve, not before: rotation averaging spreads a
    # bad edge's error over the component and hides it, while bundle adjustment
    # pulls onto the good structure and leaves it standing out.  Measured, the
    # two false edges sat at 117 px against a median of 0.2 px after BA.
    global_outlier_reject: bool = True
    global_outlier_max_px: float = 25.0
    global_outlier_max_drops: int = 6
    global_outlier_rounds: int = 2          # solve / reject / re-solve cycles

    # ── compositing ───────────────────────────────────────────────────────
    # Azimuthal equidistant about the mean gaze direction: pixel spacing stays
    # proportional to arc length on the retinal sphere, which is the clinically
    # meaningful quantity, and it degrades far more gracefully than gnomonic at
    # wide field.
    projection: str = "azimuthal_equidistant"   # | "gnomonic"
    output_scale: float = 1.0               # output px per analysis px
    output_max_px: int = 4000               # hard cap on the long edge
    output_margin_frac: float = 0.05
    tile_px: int = 512                      # compositing tile size
    blend_mode: str = "weighted_median"     # | "weighted_mean"
    # Median compositing rejects outliers that survived the specular exclusion
    # and any residual exposure mismatch, since a bad contribution is unlikely
    # to be the weighted middle of the stack.
    exposure_normalise: bool = True
    preview_max_px: int = 1200
    # Per-patch boundary overlay: which capture contributed which region.  The
    # geometry always goes into bundle.json; this only controls whether a
    # rendered copy is written beside the mosaic.
    write_outlined_mosaic: bool = True
    outline_thickness: int = 1
    outline_dim: float = 0.35               # darken the mosaic under the lines

    # ── runtime / resource ────────────────────────────────────────────────
    cv_num_threads: int = 4
    descriptor_dtype: str = "float16"       # stored dtype in the bundle
    lru_cache_patches: int = 8              # full-res patches held at once
    fail_soft: bool = True                  # always write a bundle, even on error

    ladder: LadderConfig = field(default_factory=LadderConfig)

    # ── serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MosaicConfig":
        d = dict(d or {})
        lad = d.pop("ladder", None)
        known = {f.name for f in fields(cls)} - {"ladder"}
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        if lad:
            lk = {f.name for f in fields(LadderConfig)}
            cfg.ladder = LadderConfig(**{k: v for k, v in lad.items() if k in lk})
        return cfg

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "MosaicConfig":
        return cls.from_dict(json.loads(s))


DEFAULT = MosaicConfig()
