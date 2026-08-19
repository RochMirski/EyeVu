"""Pair shortlisting, descriptor matching, Hough seeding, PROSAC, the ladder.

The hard problem this module exists for: a true overlap between two of these
patches typically yields THREE TO FIVE inlier correspondences.  Measured over a
real 14-capture session, true pairs scored 3-5 inliers and false pairs 5-8 --
the inlier-count distributions do not separate at all.  Two consequences run
through everything below.

*   Estimation must work at 3-5 correspondences.  That is what the 3-DOF model
    (2 correspondences) and Hough seeding (consensus from 3-4 matches, using the
    scale and orientation each SIFT match already carries) are for.

*   Acceptance must NOT be keyed on inlier count, because it cannot be.  The
    decisive test is photometric: warp one patch onto the other and correlate
    the pixels where they actually land together.  Measured, that separates
    cleanly with no overlap -- true 0.53-0.91, false below 0.46 (0.79 vs 0.19 on
    synthetic captures).  `ncc_verify` is the gate that matters here.

References
    Lowe, IJCV 2004                  -- Hough voting on similarity parameters
    Brown & Lowe, IJCV 2007          -- match graph, Hough seeding
    Chum & Matas, CVPR 2005          -- PROSAC
"""

from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from . import models as M
from .features import shared_landmarks
from .preprocess import patch_K


# ══ shortlisting ══════════════════════════════════════════════════════════

def global_descriptor(patch, cfg):
    """A cheap whole-patch signature for ranking candidate pairs.

    The masked, illumination-flattened patch squashed to NxN and normalised.
    Crude, but it only has to rank -- anything it wrongly demotes is still
    reachable through the random sample below.
    """
    n = int(cfg.global_desc_size)
    img = patch.image.astype(np.float32)
    m = (patch.mask > 0).astype(np.float32)
    num = cv2.resize(img * m, (n, n), interpolation=cv2.INTER_AREA)
    den = cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA)
    d = (num / np.maximum(den, 1e-6)).ravel()
    d -= d.mean()
    return d / max(float(np.linalg.norm(d)), 1e-6)


def shortlist(patches, cfg):
    """Which pairs to attempt, and why the rest were skipped.

    All-pairs matching is O(n^2).  At the real session sizes (up to 40 patches,
    780 pairs of ~100 keypoints each) that is seconds, so below
    `shortlist_all_below_n` we simply do all of them and skip the heuristics
    entirely -- a shortlist that saves nothing can only lose overlaps.

    Above it: rank by global-descriptor similarity, keep the top k per patch,
    and add a random sample of the remainder so an unexpected overlap between
    two patches that look nothing alike is still discoverable.

    Returns (pairs, skipped) with pairs a sorted list of (i, j), i < j.
    """
    idx = [p.index for p in patches]
    all_pairs = [(a, b) for ii, a in enumerate(idx) for b in idx[ii + 1:]]
    if len(patches) < cfg.shortlist_all_below_n:
        return all_pairs, []

    by_index = {p.index: p for p in patches}
    G = {i: global_descriptor(by_index[i], cfg) for i in idx}
    keep = set()
    for i in idx:
        others = [j for j in idx if j != i]
        sim = np.array([float(G[i] @ G[j]) for j in others])
        for j in np.argsort(-sim)[:int(cfg.shortlist_top_k)]:
            keep.add((min(i, others[j]), max(i, others[j])))

    rng = np.random.default_rng(int(cfg.shortlist_seed))
    rest = [p for p in all_pairs if p not in keep]
    if rest and cfg.shortlist_random_k > 0:
        n = min(len(rest), int(cfg.shortlist_random_k) * len(idx))
        for k in rng.choice(len(rest), size=n, replace=False):
            keep.add(rest[int(k)])

    skipped = [{"pair": list(p), "reason": "not in descriptor shortlist"}
               for p in all_pairs if p not in keep]
    return sorted(keep), skipped


# ══ descriptor matching ═══════════════════════════════════════════════════

def match_descriptors(fa, fb, cfg):
    """Mutual nearest neighbour with a Lowe ratio test.

    Returns (ia, ib, dist) sorted by ASCENDING descriptor distance -- the order
    PROSAC needs, and the order Hough voting benefits from.

    The ratio threshold is looser than the usual 0.7 (default 0.8) because a
    3-DOF model absorbs outliers cheaply and the real gate downstream is
    photometric.  Tightening it here only throws away true correspondences that
    this material cannot spare.
    """
    if len(fa) == 0 or len(fb) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(fa.desc, fb.desc, k=min(2, len(fb)))
    fwd = {}
    for pair in knn:
        if not pair:
            continue
        m = pair[0]
        if len(pair) == 2 and m.distance >= cfg.lowe_ratio * pair[1].distance:
            continue
        fwd[m.queryIdx] = (m.trainIdx, m.distance)

    if cfg.mutual_nn and fwd:
        rev = {}
        for pair in bf.knnMatch(fb.desc, fa.desc, k=1):
            if pair:
                rev[pair[0].queryIdx] = pair[0].trainIdx
        fwd = {a: (b, d) for a, (b, d) in fwd.items() if rev.get(b) == a}

    if not fwd:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    ia = np.fromiter(fwd.keys(), int, len(fwd))
    ib = np.fromiter((v[0] for v in fwd.values()), int, len(fwd))
    dd = np.fromiter((v[1] for v in fwd.values()), float, len(fwd))
    o = np.argsort(dd)
    return ia[o], ib[o], dd[o]


# ══ similarity-space Hough seeding ════════════════════════════════════════

def hough_seed(fa, fb, ia, ib, shape_a, cfg):
    """Vote for a coarse similarity, and return the winning bin's matches.

    Every SIFT match carries a scale and an orientation, so ONE correspondence
    already hypothesises a 4-DOF similarity (Lowe 2004; Brown & Lowe 2007).
    Accumulating those hypotheses finds consensus from three or four matches in
    cases where positional RANSAC, which must first guess a consistent sample,
    finds nothing at all.  On this data that is the common case, not the corner
    case.

    Each match votes into the two nearest bins in each of the four dimensions
    (16 cells) so a hypothesis straddling a bin boundary is not lost.
    """
    if len(ia) < cfg.hough_min_votes:
        return np.zeros(0, int)

    h, w = shape_a[:2]
    ca = np.array([w * 0.5, h * 0.5])
    sa, sb = fa.size[ia], fb.size[ib]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(sa > 1e-6, sb / np.maximum(sa, 1e-6), 1.0)
    s = np.clip(s, 1e-3, 1e3)
    th = np.radians(fb.angle[ib] - fa.angle[ia])

    # Where patch A's centre would land under each match's own similarity.
    d = ca[None, :] - fa.xy[ia]
    cos, sin = np.cos(th), np.sin(th)
    px = fb.xy[ib][:, 0] + s * (cos * d[:, 0] - sin * d[:, 1])
    py = fb.xy[ib][:, 1] + s * (sin * d[:, 0] + cos * d[:, 1])

    bxy = max(1.0, cfg.hough_bins_xy * max(w, h))
    bls = math.log(max(cfg.hough_bins_scale, 1.0001))
    bth = math.radians(cfg.hough_bins_theta_deg)
    coords = np.stack([px / bxy, py / bxy, np.log(s) / bls,
                       np.arctan2(np.sin(th), np.cos(th)) / bth], axis=1)

    # Each match votes into 16 cells; find the fullest cell without a Python
    # loop by encoding the 4-D bin index and using np.unique.
    base = np.floor(coords).astype(np.int64)
    cells = (base[:, None, :] + _OFFSETS16[None, :, :]).reshape(-1, 4)
    who = np.repeat(np.arange(len(ia)), len(_OFFSETS16))
    keys, inv = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(keys))
    top = int(np.argmax(counts))
    if counts[top] < cfg.hough_min_votes:
        return np.zeros(0, int)
    return np.unique(who[inv == top])


_OFFSETS16 = np.array([[a, b, c, d]
                       for a in (0, 1) for b in (0, 1)
                       for c in (0, 1) for d in (0, 1)], int)


# ══ PROSAC ════════════════════════════════════════════════════════════════

def _prosac_samples(n, m, cfg, rng, n_prior):
    """Sample indices for every PROSAC iteration, as a (T, m) array.

    The pool grows with the iteration count, so early draws come only from the
    best-ranked correspondences and later ones reach further down the list --
    which is the whole point of PROSAC over uniform RANSAC at low inlier ratios.
    Landmark priors, if present, occupy the leading rows and are therefore drawn
    first automatically.
    """
    T = int(cfg.ransac_max_iters)
    pool = np.minimum(n, np.maximum(m, n_prior + m
                                    + (np.arange(1, T + 1) * cfg.prosac_growth
                                       ).astype(int)))
    a = (rng.random(T) * pool).astype(int)
    b = (rng.random(T) * (pool - 1)).astype(int)
    b += (b >= a)
    out = np.stack([a, b], axis=1)
    # Exhaust the landmark priors first when there are enough of them.
    if n_prior >= m:
        pairs = [(i, j) for i in range(n_prior) for j in range(i + 1, n_prior)]
        out[:len(pairs)] = np.array(pairs[:T], int)
    return np.clip(out, 0, n - 1)


def prosac_rotation(src, dst, Ka, Kb, thresh, cfg, rng=None, n_prior=0):
    """PROSAC for the 3-DOF conjugate rotation, fitted in BATCH.

    Every hypothesis is a 2-point Kabsch solve, and numpy will do thousands of
    3x3 SVDs in one call, so the entire iteration loop collapses into a handful
    of vectorised operations.  This is not a micro-optimisation: the scalar
    version spends ~0.15 s per pair, which on a Pi Zero 2 W over 400+ pairs is
    the difference between a few minutes and most of an hour.

    Returns (H, inlier_mask) or (None, None).
    """
    n = len(src)
    if n < 2:
        return None, None
    rng = rng or np.random.default_rng(0)
    u = M.rays(src, Ka)
    v = M.rays(dst, Kb)
    samples = _prosac_samples(n, 2, cfg, rng, n_prior)

    Kai = np.linalg.inv(Ka)
    src_h = np.hstack([np.asarray(src, np.float64), np.ones((n, 1))])
    dst_a = np.asarray(dst, np.float64)
    best_H, best_inl, best_score = None, None, -1.0

    # Chunked so peak memory stays flat regardless of iteration count.
    for s0 in range(0, len(samples), 512):
        sm = samples[s0:s0 + 512]
        # Batched Kabsch: correlation matrix per hypothesis.  matmul rather
        # than einsum throughout -- both go through BLAS, and on this shape
        # einsum falls back to its own slow path and dominates the profile.
        C = np.matmul(np.transpose(v[sm], (0, 2, 1)), u[sm])
        ok = np.all(np.isfinite(C), axis=(1, 2))
        if not ok.any():
            continue
        C = C[ok]
        try:
            U, _S, Vt = np.linalg.svd(C)
        except np.linalg.LinAlgError:
            continue
        D = np.repeat(np.eye(3)[None], len(C), axis=0)
        D[:, 2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
        H = Kb @ R @ Kai                                  # (T', 3, 3)

        p = np.matmul(H, src_h.T)                         # (T', 3, n)
        w = p[:, 2, :]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        dx = p[:, 0, :] / w - dst_a[None, :, 0]
        dy = p[:, 1, :] / w - dst_a[None, :, 1]
        err = np.hypot(dx, dy)
        inl = err < thresh
        k = inl.sum(axis=1)
        score = k - (np.where(inl, err, 0.0).sum(axis=1) / (thresh * n + 1e-9))
        t = int(np.argmax(score))
        if k[t] >= 2 and score[t] > best_score:
            best_H, best_inl, best_score = H[t], inl[t], float(score[t])

    if best_H is None:
        return None, None
    # Refit on the full consensus: the 2 points that found it are rarely the
    # best estimate of it.
    if int(best_inl.sum()) > 2:
        R2 = M.fit_rotation(np.asarray(src)[best_inl], dst_a[best_inl], Ka, Kb)
        if R2 is not None:
            H2 = M.H_from_rotation(R2, Ka, Kb)
            inl2 = M.transfer_error(H2, src, dst) < thresh
            if int(inl2.sum()) >= int(best_inl.sum()):
                best_H, best_inl = H2, inl2
    return best_H, best_inl


# DEAD CODE - no caller anywhere in the repo (AST reachability scan, 2026-08-19).
# Commented out, not deleted.  See docs/REPO_MAP.md section 5.1.
# def prosac(src, dst, m, fit, thresh, cfg, rng=None, n_prior=0):
#     """Robust fit drawing samples in ascending descriptor-distance order.

#     Plain RANSAC samples uniformly, which at low inlier ratios wastes almost
#     every draw.  PROSAC (Chum & Matas 2005) exploits the fact that the matches
#     are already sorted by a quality proxy -- descriptor distance -- and grows the
#     sampling pool outward from the best ones, so it converges far faster in
#     exactly the regime this data sits in.

#     `fit(src_sample, dst_sample)` returns a 3x3 or None.  `n_prior` marks how
#     many leading correspondences are landmark priors, which are always included
#     in the sample when present.

#     Returns (H, inlier_mask) or (None, None).
#     """
#     n = len(src)
#     if n < m:
#         return None, None
#     rng = rng or np.random.default_rng(0)
#     best_H, best_inl, best_score = None, None, -1.0
#     pool = max(m, min(n, n_prior + m))
#     iters, t = int(cfg.ransac_max_iters), 0

#     while t < iters:
#         t += 1
#         pool = min(n, max(m, n_prior + m + int(t * cfg.prosac_growth)))
#         if n_prior >= m:
#             sample = rng.choice(n_prior, size=m, replace=False)
#         elif n_prior > 0:
#             extra = rng.choice(np.arange(n_prior, pool), size=m - n_prior,
#                                replace=False)
#             sample = np.concatenate([np.arange(n_prior), extra])
#         else:
#             sample = rng.choice(pool, size=m, replace=False)

#         H = fit(src[sample], dst[sample])
#         if H is None or not np.all(np.isfinite(H)):
#             continue
#         err = M.transfer_error(H, src, dst)
#         inl = err < thresh
#         k = int(inl.sum())
#         if k < m:
#             continue
#         # Rank by inlier count, breaking ties on total inlier residual.
#         score = k - float(err[inl].sum()) / (thresh * n + 1e-9)
#         if score > best_score:
#             best_H, best_inl, best_score = H, inl, score
#             w = max(k / float(n), 1e-6)
#             if w < 1.0:
#                 need = math.log(max(1e-12, 1.0 - cfg.ransac_confidence)) / \
#                     math.log(max(1e-12, 1.0 - w ** m))
#                 iters = int(min(cfg.ransac_max_iters, max(m * 4, need)))

#     if best_H is None:
#         return None, None
#     # Final refit on the full inlier set: the sample that found the consensus is
#     # rarely the best estimate of it.
#     if int(best_inl.sum()) > m:
#         H2 = fit(src[best_inl], dst[best_inl])
#         if H2 is not None and np.all(np.isfinite(H2)):
#             inl2 = M.transfer_error(H2, src, dst) < thresh
#             if int(inl2.sum()) >= int(best_inl.sum()):
#                 best_H, best_inl = H2, inl2
#     return best_H, best_inl


# ══ the model ladder ══════════════════════════════════════════════════════

def fit_ladder(src, dst, pa, pb, cfg, rng=None, n_prior=0, seed_idx=None):
    """Fit src->dst, starting at L0 and promoting only on hard evidence.

    Escalation follows the dual-bootstrap principle (Stewart et al., TMI 2003):
    enter at the lowest order the data supports and raise the order only when
    the evidence justifies it.  A higher-DOF model ALWAYS fits better in-sample,
    so residual improvement can never justify promotion on its own -- both the
    inlier-count gate and the residual gate must pass, every time.

    Returns (estimate, inlier_mask, info, estimate_l0).  The L0 estimate is
    returned alongside the winner whatever level the pair settles at, because
    the graph, cycle consistency and rotation averaging all need a rotation per
    edge -- and an affine or homographic winner has none.  Keeping both is what
    lets a mixed-level graph still be reasoned about on SO(3).
    """
    info = {"levels_tried": [], "level": None}
    Ka, Kb = patch_K(pa), patch_K(pb)
    thresh = float(cfg.ransac_reproj_px)
    rng = rng or np.random.default_rng(0)

    # ── L0: the entry point.  Two correspondences suffice. ──
    def fit_rot(s, d):
        R = M.fit_rotation(s, d, Ka, Kb)
        return None if R is None else M.H_from_rotation(R, Ka, Kb)

    H0, inl0 = prosac_rotation(src, dst, Ka, Kb, thresh, cfg, rng, n_prior)

    # A Hough consensus is an independent hypothesis; take whichever wins.
    if seed_idx is not None and len(seed_idx) >= M.LEVEL_MIN_CORR[0]:
        Hs = fit_rot(src[seed_idx], dst[seed_idx])
        if Hs is not None and np.all(np.isfinite(Hs)):
            inls = M.transfer_error(Hs, src, dst) < thresh
            if inl0 is None or int(inls.sum()) > int(inl0.sum()):
                H0, inl0 = Hs, inls
                info["seeded_by_hough"] = True

    if H0 is None or inl0 is None or int(inl0.sum()) < cfg.min_inliers:
        info["fail"] = "no L0 consensus"
        return None, None, info, None

    R0 = M.fit_rotation(src[inl0], dst[inl0], Ka, Kb)
    if R0 is None:
        info["fail"] = "L0 refit failed"
        return None, None, info, None
    best = M.make_estimate(0, R=R0, focal=pa.focal,
                           matrix=M.H_from_rotation(R0, Ka, Kb))
    est_l0 = dict(best)
    best_inl = inl0
    best_med = float(np.median(M.estimate_residuals(best, src[inl0], dst[inl0])))
    info["levels_tried"].append({"level": 0, "inliers": int(inl0.sum()),
                                 "median_residual": best_med})

    lad = cfg.ladder
    gates = (None, lad.l1_min_inliers, lad.l2_min_inliers,
             lad.l3_min_inliers, lad.l4_min_inliers)

    for lvl in range(1, min(int(lad.enabled_max_level), 4) + 1):
        n_in = int(best_inl.sum())
        if n_in < gates[lvl] or n_in < M.LEVEL_MIN_CORR[lvl]:
            break
        if lvl == 4 and _baseline(src[best_inl], dst[best_inl]) < lad.l4_min_baseline_px:
            break

        cand = _fit_level(lvl, src[best_inl], dst[best_inl], pa, pb, Ka, Kb)
        if cand is None:
            break
        res_all = M.estimate_residuals(cand, src, dst)
        inl = res_all < thresh
        if int(inl.sum()) < n_in:                 # a promotion must not lose data
            break
        med = float(np.median(res_all[inl]))
        info["levels_tried"].append({"level": lvl, "inliers": int(inl.sum()),
                                     "median_residual": med})
        # BOTH gates, never one.
        if med > (1.0 - lad.residual_gain) * best_med:
            break
        best, best_inl, best_med = cand, inl, med

    info["level"] = best["level"]
    info["median_residual"] = best_med
    info["n_inliers"] = int(best_inl.sum())
    return best, best_inl, info, est_l0


def _fit_level(lvl, s, d, pa, pb, Ka, Kb):
    if lvl == 1:
        r = M.fit_rotation_focal(s, d, Ka, Kb, pa.centre[0], pa.centre[1],
                                 pb.centre[0], pb.centre[1], pa.focal)
        if r is None:
            return None
        R, f = r
        Ka2 = M.make_K(f, pa.centre[0], pa.centre[1])
        Kb2 = M.make_K(f, pb.centre[0], pb.centre[1])
        return M.make_estimate(1, R=R, focal=f,
                               matrix=M.H_from_rotation(R, Ka2, Kb2))
    if lvl == 2:
        A = M.fit_affine(s, d)
        return None if A is None else M.make_estimate(2, matrix=A)
    if lvl == 3:
        H = M.fit_homography(s, d)
        return None if H is None else M.make_estimate(3, matrix=H)
    if lvl == 4:
        q = M.fit_quadratic(s, d)
        if q is None:
            return None
        coeffs, T = q
        # Keep a linear companion so warping and bbox code still have a matrix.
        lin = M.fit_affine(s, d)
        return M.make_estimate(4, coeffs=coeffs, T=T, matrix=lin)
    return None


def _baseline(src, dst):
    """How far the correspondences actually moved -- a wide-baseline proxy."""
    return float(np.median(np.linalg.norm(np.asarray(dst) - np.asarray(src), axis=1)))


# ══ photometric verification ══════════════════════════════════════════════

def ncc_verify(pa, pb, H, cfg):
    """Do the two patches agree where this transform puts them together?

    Warps B into A's frame and correlates over the region where both masks land.
    THIS IS THE DECISIVE ACCEPTANCE TEST.  Descriptors can coincide by chance on
    self-similar tissue and inlier counts do not separate true from false pairs
    on this material, but the pixels themselves will not agree unless the
    overlap is real.

    Returns (ok, ncc, overlap_px).
    """
    ha, wa = pa.image.shape[:2]
    Hi = _inv(H)
    if Hi is None:
        return False, 0.0, 0
    wm = cv2.warpPerspective(pb.mask, Hi, (wa, ha), flags=cv2.INTER_NEAREST)
    ov = (wm > 0) & (pa.mask > 0)
    n = int(ov.sum())
    smaller = max(1, min(pa.area, int(np.count_nonzero(wm))))
    if n < cfg.min_overlap_px or n < cfg.min_overlap_frac * smaller:
        return False, 0.0, n
    wb = cv2.warpPerspective(pb.image, Hi, (wa, ha))
    x = pa.image[ov].astype(np.float32)
    y = wb[ov].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    den = float(np.sqrt(float((x * x).sum()) * float((y * y).sum())))
    if den <= 1e-6:
        return False, 0.0, n
    ncc = float((x * y).sum() / den)
    return ncc >= cfg.ncc_min, ncc, n


def _inv(H):
    try:
        Hi = np.linalg.inv(np.asarray(H, np.float64))
        return Hi if np.all(np.isfinite(Hi)) else None
    except np.linalg.LinAlgError:
        return None


# ══ direct alignment fallback ═════════════════════════════════════════════

def direct_align(pa, pb, cfg):
    """Align two patches on PIXELS when descriptors fail to produce a consensus.

    Why this is worth having.  A true overlap on this material yields a median
    of two or three descriptor matches that are actually correct; below three,
    no amount of robust fitting recovers a model, and measured edge recall from
    the feature path alone is about 0.20.  The rest of the overlaps are not
    ambiguous -- warp the patches together and they agree at NCC 0.8+ -- they
    simply have no descriptors to find each other with.

    That is affordable to fix here ONLY because the model is 3-DOF.  The search
    is two translations and a roll; an 8-DOF homography could not be searched
    this way.  So the reduced-DOF model pays off twice: once by needing fewer
    correspondences, and again by making a direct search tractable at all.

    Translation is solved in one shot per roll hypothesis by masked FFT
    normalised cross-correlation (Padfield 2012), which handles the irregular
    masks correctly -- ordinary correlation would lock onto blob shape rather
    than anatomy.

    Returns (R, ncc, overlap_px) or (None, 0.0, 0).
    """
    # scikit-image may genuinely be absent (the Pi installer's pip block is
    # commented out), so its absence must degrade rather than break: the feature
    # path still works, there are simply fewer edges.  Reported explicitly so
    # the bundle does not claim the alignment was ATTEMPTED and failed.
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:                                     # pragma: no cover
        return None, float("nan"), 0

    scale = float(cfg.direct_downscale)
    A = _shrink(pa.image, scale)
    Am = _shrink(pa.mask, scale)
    h, w = pb.image.shape[:2]
    centre = (w * 0.5, h * 0.5)
    best = (None, -1.0, 0)

    # Screen at zero roll first, and only search roll for pairs that already
    # look plausible.  Most candidate pairs do not overlap at all, and paying
    # for a full roll sweep to establish that is where the time goes: on a
    # 40-patch session it is ~730 failed pairs times nine hypotheses.  Roll
    # between two captures of the same eye is a few degrees at most, so a real
    # overlap still scores respectably at roll zero -- the screen threshold is
    # set well below the acceptance threshold precisely so it cannot become the
    # thing that decides.
    rolls = [0.0]
    screened = _try_roll(pa, pb, 0.0, A, Am, scale, centre, cfg,
                         phase_cross_correlation)
    if screened[1] < cfg.direct_screen_ncc:
        return screened if screened[0] is not None else (None, 0.0, 0)
    best = screened
    rolls = [r for r in np.arange(-cfg.direct_roll_range_deg,
                                  cfg.direct_roll_range_deg + 1e-9,
                                  cfg.direct_roll_step_deg) if abs(r) > 1e-9]

    for roll in rolls:
        cand = _try_roll(pa, pb, float(roll), A, Am, scale, centre, cfg,
                         phase_cross_correlation)
        if cand[1] > best[1]:
            best = cand
    return best


def _try_roll(pa, pb, roll, A, Am, scale, centre, cfg, pcc):
    """One roll hypothesis: FFT-solve the translation, then score it."""
    h, w = pb.image.shape[:2]
    if abs(roll) > 1e-9:
        Mrot = cv2.getRotationMatrix2D(centre, roll, 1.0)
        Bi = cv2.warpAffine(pb.image, Mrot, (w, h))
        Bm = cv2.warpAffine(pb.mask, Mrot, (w, h), flags=cv2.INTER_NEAREST)
    else:
        Bi, Bm = pb.image, pb.mask
    B = _shrink(Bi, scale)
    Bms = _shrink(Bm, scale)
    if not Am.any() or not Bms.any():
        return None, -1.0, 0
    # Pad both to a common shape so the FFT correlation is well defined.
    H = max(A.shape[0], B.shape[0])
    W = max(A.shape[1], B.shape[1])
    try:
        shift, _e, _p = pcc(_pad(A, H, W), _pad(B, H, W),
                            reference_mask=_pad(Am, H, W) > 0,
                            moving_mask=_pad(Bms, H, W) > 0)
    except (ValueError, RuntimeError):
        return None, -1.0, 0
    dy, dx = float(shift[0]) / scale, float(shift[1]) / scale
    # Correspondences implied by (roll, dx, dy), fitted back into a rotation so
    # the rest of the pipeline sees an ordinary L0 estimate.
    R = _euclid_to_rotation(pa, pb, roll, dx, dy)
    if R is None:
        return None, -1.0, 0
    Hm = M.H_from_rotation(R, patch_K(pa), patch_K(pb))
    _ok, ncc, ov = ncc_verify(pa, pb, Hm, cfg)
    return R, float(ncc), int(ov)


def _shrink(img, scale):
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _pad(img, H, W):
    return cv2.copyMakeBorder(img, 0, H - img.shape[0], 0, W - img.shape[1],
                              cv2.BORDER_CONSTANT, value=0)


def _euclid_to_rotation(pa, pb, roll_deg, dx, dy):
    """Turn a (roll, dx, dy) alignment into the equivalent 3-DOF rotation.

    Synthesises correspondences from the implied 2-D transform and fits the
    rotation to them, so the result is an ordinary L0 estimate that composes,
    cycle-checks and bundle-adjusts exactly like a feature-derived one.
    """
    h, w = pa.image.shape[:2]
    r = max(8.0, pa.radius)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    src = np.stack([pa.centre[0] + r * np.cos(ang),
                    pa.centre[1] + r * np.sin(ang)], axis=1)
    th = np.radians(roll_deg)
    c, s = np.cos(th), np.sin(th)
    cb = np.array([pb.image.shape[1] * 0.5, pb.image.shape[0] * 0.5])
    # Inverse of the warp applied to B, then the measured shift.
    d = src - cb
    dst = np.stack([cb[0] + c * d[:, 0] + s * d[:, 1] - dx,
                    cb[1] - s * d[:, 0] + c * d[:, 1] - dy], axis=1)
    return M.fit_rotation(src, dst, patch_K(pa), patch_K(pb))


# ══ one pair, end to end ══════════════════════════════════════════════════

def match_pair(pa, pb, fa, fb, cfg, rng=None):
    """Match patch A to patch B.  Returns a record dict; never raises.

    The record is written to bundle.json whether the pair was accepted or not,
    with the reason -- a rejected pair is evidence about the session, not noise.
    """
    rec = {"pair": [pa.index, pb.index], "n_putative": 0, "n_inliers": 0,
           "level": None, "ncc": 0.0, "overlap_px": 0, "median_residual": None,
           "accepted": False, "reason": "", "estimate": None,
           "matches": None, "inlier_mask": None}

    ia, ib, dd = match_descriptors(fa, fb, cfg)
    rec["n_putative"] = int(len(ia))
    if len(ia) < cfg.min_putative:
        rec["reason"] = f"only {len(ia)} putative matches"
        return _maybe_direct(rec, pa, pb, cfg)

    src = fa.xy[ia].astype(np.float64)
    dst = fb.xy[ib].astype(np.float64)

    # Landmark priors go first so PROSAC always draws them (see features.py --
    # nothing populates landmarks yet, but the path is live and tested).
    ls, ld, lnames = shared_landmarks(fa, fb)
    n_prior = len(ls)
    if n_prior:
        src = np.vstack([ls, src])
        dst = np.vstack([ld, dst])
        ia = np.concatenate([-np.ones(n_prior, int), ia])
        ib = np.concatenate([-np.ones(n_prior, int), ib])
        rec["landmarks"] = lnames

    seed = (hough_seed(fa, fb, ia[n_prior:], ib[n_prior:], pa.image.shape, cfg)
            if cfg.hough_enabled else np.zeros(0, int))
    if len(seed):
        seed = seed + n_prior
    rec["n_hough_votes"] = int(len(seed))

    est, inl, info, est_l0 = fit_ladder(src, dst, pa, pb, cfg, rng, n_prior,
                                        seed if len(seed) else None)
    rec["ladder"] = info
    if est is None:
        rec["reason"] = info.get("fail", "no consensus")
        return _maybe_direct(rec, pa, pb, cfg)
    rec["estimate_l0"] = est_l0

    n_in = int(inl.sum())
    rec["n_inliers"] = n_in
    rec["level"] = est["level"]
    rec["median_residual"] = info.get("median_residual")
    if n_in < cfg.min_inliers:
        rec["reason"] = f"{n_in} inliers < {cfg.min_inliers}"
        return _maybe_direct(rec, pa, pb, cfg)
    if n_in / max(1, len(src)) < cfg.min_inlier_frac:
        rec["reason"] = f"inlier fraction {n_in / len(src):.2f} too low"
        return _maybe_direct(rec, pa, pb, cfg)

    R = M.estimate_rotation(est)
    if R is not None:
        ang = M.rotation_angle_deg(R)
        rec["rotation_deg"] = ang
        if ang > cfg.max_rotation_deg:
            rec["reason"] = f"rotation {ang:.1f} deg implausible"
            return rec

    ok, ncc, ov = ncc_verify(pa, pb, est["matrix"], cfg)
    rec["ncc"], rec["overlap_px"] = ncc, ov
    if not ok:
        rec["reason"] = (f"photometric check failed (ncc {ncc:.2f}, "
                         f"overlap {ov}px)")
        return _maybe_direct(rec, pa, pb, cfg)

    rec["accepted"] = True
    rec["source"] = "features"
    rec["estimate"] = est
    rec["matches"] = np.stack([ia, ib], axis=1)
    rec["inlier_mask"] = inl
    return rec


def _maybe_direct(rec, pa, pb, cfg):
    """Last resort for a pair the feature path could not solve.

    Only ever runs after the feature path has failed, and its result is held to
    a STRICTER photometric threshold than a feature-derived edge: a direct
    search optimises the very quantity it is then judged on, so the same gate
    would not mean the same thing.  It also contributes no correspondences, so
    it takes no part in bundle adjustment -- it constrains the graph through
    rotation averaging only, which `globalopt.refine` already handles.
    """
    if not cfg.direct_align:
        return rec
    R, ncc, ov = direct_align(pa, pb, cfg)
    rec["direct_ncc"] = round(float(ncc), 4)
    if R is None or ncc < cfg.direct_ncc_min or ov < cfg.min_overlap_px:
        rec["reason"] = f"{rec['reason']}; direct align ncc {ncc:.2f}"
        return rec
    ang = M.rotation_angle_deg(R)
    if ang > cfg.max_rotation_deg:
        rec["reason"] = f"{rec['reason']}; direct align rotation {ang:.1f} deg"
        return rec
    est = M.make_estimate(0, R=R, focal=pa.focal,
                          matrix=M.H_from_rotation(R, patch_K(pa), patch_K(pb)))
    rec.update({"accepted": True, "source": "direct", "estimate": est,
                "estimate_l0": est, "ncc": float(ncc), "overlap_px": int(ov),
                "level": 0, "reason": "", "n_inliers": 0,
                "matches": None, "inlier_mask": None})
    return rec
