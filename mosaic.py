"""Stitch the red-eye (fundus) extracts of one capture session into a mosaic.

Each capture isolates the pixels the flash brought back through the pupil — a
small, irregular patch of retina.  Different gaze directions expose different
patches, so several captures of the same eye can be registered and blended into a
wider view of the fundus than any single flash can reach.

Design notes
────────────
* **Feature-based**, SIFT preferred with an ORB fallback: the patches overlap only
  partially and sit at different scales/rotations as the eye rolls, so a plain
  translation model is not enough.  SIFT is markedly better on this low-contrast,
  low-texture material; ORB is there because SIFT is absent from some OpenCV
  builds (notably older ARM ones on the Pi).
* Features are detected on a **contrast-boosted** copy of the patch, but only
  inside its own mask — the black surround is not real image content and its
  border generates strong, meaningless corners.
* Registration is **graph-based**, not sequential.  Every pair of captures is
  matched once, and the resulting graph decides both where to start and what to
  add: the seed is the best-connected image (most inlier matches to the others,
  inside the largest connected component), not whichever happened to be captured
  first.  A poor first capture therefore no longer caps the whole mosaic.
* The mosaic then grows greedily — at each step the unplaced image with the
  strongest link to what is already placed joins, so a capture that overlaps ANY
  placed image lands, not only one that overlaps its predecessor.  When it
  overlaps several, its position is fitted jointly to all of them: the inlier
  correspondences from every placed neighbour are mapped into mosaic coordinates
  and one homography is fitted to the pooled set (RANSAC), which is the standard
  resection and beats chaining through an arbitrary single neighbour.
* Anything the graph could not attach gets one more chance against the RENDERED
  mosaic, whose blended overlap region carries features no single patch has.
* Blending is a masked running average, which hides the seams without needing a
  full multi-band pyramid.

Usable with no hardware: `python mosaic.py <session_dir> [-o out.png]`.
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

# ── tuning ──
# Three separate gates, easily confused, in the order a pair passes through them:
#   MIN_MATCHES  good descriptor matches needed before a model is even FITTED
#   MIN_INLIERS  RANSAC inliers needed to accept that fit
#   MIN_NCC      photometric agreement over the overlap — the decisive one
# All three used to sit at 8, which was far too strict at both of the first two.
# Measured on a real 14-capture session, the pair 1-2 produces 5 good matches, a
# 4-inlier similarity, and an NCC of 0.80 over 24k px of overlap — an unambiguous
# true overlap that an 8-match gate discards before ever looking at the pixels.
# A similarity needs two correspondences and a homography four, so four good
# matches is the algebraic floor; below that there is nothing to fit.  Everything
# above it is decided by whether the patches actually agree, not by how many
# descriptors happened to survive the ratio test on 30x25px of low-contrast tissue.
MIN_MATCHES = 4           # fewest good descriptor matches worth fitting a model to
MIN_KEYPOINTS = 4         # ...and fewest keypoints for a patch to be matchable
# Fewest RANSAC inliers to accept a fit.  Deliberately low, because inlier count is
# the WEAK test on this material and photometric agreement (MIN_NCC) is the strong
# one.  Measured over a real 14-capture session (extracts of 31x26 to 42x33 px,
# 32-55 keypoints each): every true pairing had only 3-5 inliers but scored NCC
# 0.53-0.91, while wrong pairings reached 5-8 inliers and scored below 0.46.  An
# inlier floor high enough to exclude the latter excludes all of the former too.
# A similarity is determined by two correspondences, so three leaves one to spare.
MIN_INLIERS = 3
LOWE_RATIO = 0.78         # Lowe ratio test for descriptor matching
RANSAC_REPROJ = 4.0       # px reprojection tolerance for findHomography
MIN_INLIER_FRAC = 0.30    # inliers/matches below this = a bogus alignment
# The canvas is sized to the union of everything the graph placed, so this margin
# is only headroom for the rescue pass (and for looking at the result).
CANVAS_MARGIN = 0.35      # pad the union by this fraction of its longest side
MAX_DIM = 2600            # cap the canvas so a runaway homography cannot explode it
# A capture is a patch of the SAME fundus seen from a slightly different gaze, so a
# homography that changes its area by more than this is not a registration.
MAX_AREA_RATIO = 6.0
# A homography has 8 degrees of freedom.  Fitted to the dozen-odd matches these
# small patches yield, it happily "explains" them with a violent projective warp
# that folds the patch over its own horizon — an alignment that looks excellent by
# inlier count and is nonsense.  So the model is chosen by the evidence: a
# similarity (4 DOF: rotate, uniform scale, translate) is what a small gaze change
# over a small patch actually produces, and the full homography is only allowed in
# when there are plenty of matches AND it earns its extra freedom in inliers.
HOMOGRAPHY_MIN_MATCHES = 25
HOMOGRAPHY_GAIN = 1.15    # homography needs this many times the similarity's inliers

# Inlier counts alone do NOT establish a match here.  Every capture is the same kind
# of blotchy, self-similar tissue, so two patches that barely touch — or do not
# overlap at all — still turn up a handful of mutually consistent descriptors, and a
# transform fitted to them stacks the patches on top of each other.  So every
# candidate is verified photometrically: warp one patch onto the other and correlate
# the pixels where they actually land together.  This is the decisive test —
# on a real session, true and false pairings separate at 0.46/0.53 with no overlap
# in between, and on synthetic captures at 0.19/0.79.
MIN_OVERLAP_FRAC = 0.12   # overlap must be this much of the smaller patch
MIN_OVERLAP_PX = 200      # ...and this many analysis-space pixels
MIN_NCC = 0.50            # zero-mean normalised correlation over that overlap
CLAHE_CLIP = 3.0
CLAHE_TILE = 8

# A red-eye extract is a SMALL patch (measured: 28x23 to 93x56 px) adrift in an
# otherwise black 480x640 frame.  Run at that scale, SIFT finds 0-7 keypoints —
# far too few to match.  Cropping to the patch and enlarging it to ANALYSIS_SIZE
# takes the same patches to 32-95 keypoints, which is what makes feature-based
# stitching viable here at all.  Detection therefore happens in a per-image
# "analysis space", and the resulting homography is mapped back afterwards.
ANALYSIS_SIZE = 240       # longest side of the crop fed to the detector, px
ANALYSIS_PAD = 4          # px of context kept around the patch when cropping
ANALYSIS_ERODE = 5        # mask erosion IN ANALYSIS SPACE (~1px of original).
                          # Measured: dropping it to 3 costs real registrations,
                          # and raising it costs more — see KP_SUPPORT_CLEAR.

# Keeping the detector off the patch's own black border, properly.
#
# Eroding the mask is the blunt version of this: it forbids keypoint CENTRES near
# the rim, uniformly, whatever their scale — so a small, perfectly clean feature
# 6px in is deleted to protect a large one whose support genuinely straddles the
# edge.  The per-keypoint test below is the right shape instead: a keypoint is
# contaminated only if ITS OWN support region reaches the boundary, which is what
# `dist_to_edge >= KP_SUPPORT_CLEAR * kp.size / 2` says.
#
# DEFAULT OFF, because measurement says so.  Over two real sessions (14 and 13
# captures), against the NCC-verified true pairs of the first:
#
#   erode=5, off     1303 kp    6/6 true pairs, 0 false   <- current behaviour
#   erode=3, x0.5    1446 kp    6/6 true pairs, 0 false
#   erode=5, x0.5    1034 kp    4/6 true pairs
#   erode=1, x1.0    1392 kp    4/6 true pairs
#   erode=1, x1.5     769 kp    3/6 true pairs
#   erode=1, x2.0     343 kp    0/6 true pairs  — total collapse
#
# The reason is scale.  A 31x26px extract has no well-interior region relative to
# SIFT's descriptor support, so on this material "clean of the boundary" and "has
# any features at all" are in direct conflict — the boundary-adjacent keypoints
# ARE the ones doing the matching, and ~77% of all keypoints are within 12
# analysis-px of the rim.  Set it above 0 once the scan captures at full sensor
# resolution, where the same patch carries four times the interior area and there
# is genuinely something to keep; the numbers above are what tightening costs
# before that point.
KP_SUPPORT_CLEAR = 0.0
MOSAIC_UPSCALE = 3.0      # render the mosaic this much larger than the source, so
                          # the stitched fundus is not limited to ~50px of detail
MAX_PASSES = 6            # backstop on the agglomeration loop.  It already stops
                          # when a pass merges nothing, and each pass at least
                          # halves the count when it does merge, so this is only
                          # ever reached if something pathological is happening.


def _detector():
    """SIFT if this OpenCV build has it, else ORB.  Returns (detector, norm)."""
    if hasattr(cv2, "SIFT_create"):
        try:
            return cv2.SIFT_create(nfeatures=1500), cv2.NORM_L2, "SIFT"
        except cv2.error:
            pass
    return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING, "ORB"


def content_bbox(mask):
    """(x0, y0, w, h) of the non-zero content, or None."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _analysis(bgr, mask=None):
    """Crop to the patch and enlarge it for feature detection.

    Returns (gray, mask, S) where S is the 3x3 transform taking ORIGINAL image
    coordinates into this analysis image, so a homography found here can be mapped
    back.  Without this the patches are far too small for SIFT to describe.
    """
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mask is None:
        mask = (gray > 0).astype(np.uint8) * 255
    bb = content_bbox(mask)
    if bb is None:
        return None, None, None
    x0, y0, bw, bh = bb
    p = ANALYSIS_PAD
    x0 = max(0, x0 - p); y0 = max(0, y0 - p)
    x1 = min(gray.shape[1], x0 + bw + 2 * p)
    y1 = min(gray.shape[0], y0 + bh + 2 * p)
    crop = gray[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None, None

    s = ANALYSIS_SIZE / float(max(crop.shape[0], crop.shape[1]))
    s = max(s, 1.0)                       # never shrink; small patches need scale
    nw, nh = max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))
    up = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    umask = cv2.resize(cmask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                            tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    up = clahe.apply(up)
    # Keep the detector off the patch boundary, whose hard edge against the black
    # surround is a strong but meaningless feature.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (ANALYSIS_ERODE, ANALYSIS_ERODE))
    umask = cv2.erode(umask, k)

    S = np.array([[s, 0.0, -s * x0],
                  [0.0, s, -s * y0],
                  [0.0, 0.0, 1.0]], np.float64)
    return up, umask, S


def _inv(M):
    """Matrix inverse, or None for a singular/degenerate transform."""
    if M is None or not np.all(np.isfinite(M)):
        return None
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None


def _corners(bbox):
    """The four corners of a content bbox, as a perspectiveTransform input."""
    x0, y0, w, h = bbox
    return np.float32([[x0, y0], [x0 + w, y0],
                       [x0 + w, y0 + h], [x0, y0 + h]]).reshape(-1, 1, 2)


def _sane(H, bbox, scale=1.0):
    """Is `H` a plausible registration for a patch of this size?

    Every capture shows the SAME fundus from a slightly different gaze, so a
    transform that inflates, collapses or folds the patch is a mis-registration
    however good its inlier count looked.  `scale` is the expected linear scaling
    built into H (the mosaic canvas is rendered upscaled), which is divided out
    before the area test.
    """
    if H is None or bbox is None or not np.all(np.isfinite(H)):
        return False
    q = cv2.perspectiveTransform(_corners(bbox), H).astype(np.float32)
    if not np.all(np.isfinite(q)):
        return False
    a_src = cv2.contourArea(_corners(bbox), True)
    a_dst = cv2.contourArea(q, True)
    if a_src == 0 or a_dst == 0:
        return False
    if (a_src > 0) != (a_dst > 0):          # orientation flipped => mirrored
        return False
    ratio = abs(a_dst / a_src) / max(1e-9, scale * scale)
    if not (1.0 / MAX_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
        return False
    return bool(cv2.isContourConvex(q))     # a folded quad is never a real view


def _models(src, dst):
    """Candidate transforms for src->dst, best-supported first.

    Both models are fitted and BOTH are returned, in preference order, so a caller
    that rejects one as geometrically implausible can fall back to the other rather
    than throwing away a pair that genuinely matches.  The homography only leads
    when there are enough matches to constrain its 8 DOF and it beats the
    similarity by HOMOGRAPHY_GAIN — see the note at that constant.

    Each candidate is (M 3x3, inlier_mask, model_name).
    """
    out = []
    if len(src) < 4:
        return out
    # Guarded like the SIFT/ORB choice above: estimateAffinePartial2D arrived in
    # OpenCV 3.4.4, and the Pi has been known to carry an older ARM build.  Without
    # it the homography is all there is — worse on sparse matches, but the sanity
    # and NCC checks still throw out what it gets wrong.
    if hasattr(cv2, "estimateAffinePartial2D"):
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=RANSAC_REPROJ,
                                             maxIters=3000, confidence=0.995)
        if A is not None and inl is not None:
            out.append((np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64),
                        inl.ravel().astype(bool), "similarity"))
    H, hinl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    if H is not None and hinl is not None:
        hk = hinl.ravel().astype(bool)
        cand = (H.astype(np.float64), hk, "homography")
        lead = (not out or (len(src) >= HOMOGRAPHY_MIN_MATCHES
                            and hk.sum() >= HOMOGRAPHY_GAIN * out[0][1].sum()))
        if lead:
            out.insert(0, cand)
        else:
            out.append(cand)
    return out


def _boundary_clear(kp, a_mask, factor=None):
    """Split keypoints into (clear_of_the_boundary, contaminated_by_it).

    A keypoint is judged on its OWN support radius — half its `size`, scaled by
    KP_SUPPORT_CLEAR — against the distance from its centre to the nearest
    non-patch pixel.  So a small feature may sit close to the rim and a large one
    may not, which is exactly the distinction a mask erosion cannot make.
    """
    f = KP_SUPPORT_CLEAR if factor is None else factor
    if not kp or f <= 0:
        return list(range(len(kp))), []
    dist = cv2.distanceTransform(a_mask, cv2.DIST_L2, 3)
    h, w = dist.shape
    keep, drop = [], []
    for i, k in enumerate(kp):
        x, y = int(k.pt[0]), int(k.pt[1])
        if 0 <= y < h and 0 <= x < w and dist[y, x] >= f * k.size * 0.5:
            keep.append(i)
        else:
            drop.append(i)
    return keep, drop


def _feat(det, bgr, mask=None):
    """Detect features for one image.  Returns {kp, desc, S} or None.

    Detection happens in analysis space (see `_analysis`); `S` is kept so the
    resulting geometry can be taken back to the image's own coordinates.
    """
    a_img, a_mask, S = _analysis(bgr, mask)
    if a_img is None:
        return None
    kp, desc = det.detectAndCompute(a_img, a_mask)
    if desc is None or len(kp) < MIN_KEYPOINTS:
        return None
    # Drop the ones whose support region reaches the black surround: their
    # descriptors partly describe the patch's own outline, which is a different
    # shape in every capture and therefore matches nothing real.
    keep, _drop = _boundary_clear(kp, a_mask)
    if len(keep) >= MIN_KEYPOINTS:
        kp = [kp[i] for i in keep]
        desc = desc[keep]
    # The analysis image/mask are kept, not just the descriptors: a candidate
    # alignment is confirmed against the PIXELS (see `_verify`), which is the only
    # thing that separates a real overlap from a lucky descriptor coincidence.
    return {"kp": kp, "desc": desc, "S": S, "img": a_img, "mask": a_mask,
            "area": int(cv2.countNonZero(a_mask))}


def _verify(fa, fb, H_an):
    """Do the two patches actually agree where this alignment puts them together?

    Warps B onto A in analysis space and correlates the two over their common
    region.  Returns (ok, ncc, overlap_px).  This is what rejects the false
    pairings that inlier counts cannot: descriptors can coincide on self-similar
    tissue, but the pixels themselves will not.
    """
    ha, wa = fa["img"].shape[:2]
    wm = cv2.warpPerspective(fb["mask"], H_an, (wa, ha), flags=cv2.INTER_NEAREST)
    ov = (wm > 0) & (fa["mask"] > 0)
    n = int(ov.sum())
    smaller = max(1, min(fa["area"], int(cv2.countNonZero(wm))))
    if n < MIN_OVERLAP_PX or n < MIN_OVERLAP_FRAC * smaller:
        return False, 0.0, n
    wb = cv2.warpPerspective(fb["img"], H_an, (wa, ha))
    x = fa["img"][ov].astype(np.float32)
    y = wb[ov].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-6:
        return False, 0.0, n
    ncc = float((x * y).sum() / denom)
    return ncc >= MIN_NCC, ncc, n


def _match_pair(norm, fa, fb, bbox_b, bbox_a=None, scale=1.0):
    """Match image B onto image A.  Returns a dict, or None if unreliable.

    `H` maps B's ORIGINAL coordinates onto A's, composed back through the two
    crop/scale transforms:  H = inv(S_A) . H_analysis . S_B.

    `src`/`dst` are the RANSAC INLIER correspondences, also in the two images'
    original coordinates.  Keeping them is what allows a later image to be fitted
    against several already-placed neighbours at once instead of being chained
    through just one of them.

    Each candidate model is checked for geometric plausibility before it is
    accepted, and `bbox_a` additionally demands plausibility in the REVERSE
    direction — which image becomes the seed decides which way the transform gets
    used, and a fit can be well-behaved over one patch while folding over the
    other's.  Failing that test demotes to the simpler model rather than dropping
    the pair.
    """
    if fa is None or fb is None:
        return None
    ka, da, Sa = fa["kp"], fa["desc"], fa["S"]
    kb, db, Sb = fb["kp"], fb["desc"], fb["S"]
    matcher = cv2.BFMatcher(norm)
    try:
        knn = matcher.knnMatch(db, da, k=2)
    except cv2.error:
        return None
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None
    src_an = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_an = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    ia, ib = _inv(Sa), _inv(Sb)
    if ia is None or ib is None:
        return None
    for H_an, keep, model in _models(src_an, dst_an):
        n_in = int(keep.sum())
        if n_in < MIN_INLIERS or n_in / len(good) < MIN_INLIER_FRAC:
            continue
        H = ia @ H_an @ Sb
        if not _sane(H, bbox_b, scale):
            continue
        if bbox_a is not None:
            back = _inv(H)
            if back is None or not _sane(back, bbox_a, 1.0 / max(1e-9, scale)):
                continue
        ok, ncc, ov = _verify(fa, fb, H_an)
        if not ok:
            continue
        return {"H": H, "inliers": n_in, "matches": len(good), "model": model,
                "ncc": ncc, "overlap": ov,
                "src": cv2.perspectiveTransform(src_an[keep], ib),
                "dst": cv2.perspectiveTransform(dst_an[keep], ia)}
    return None


def _reverse(m):
    """The same pairing seen from the other image (B->A becomes A->B)."""
    Hi = _inv(m["H"])
    if Hi is None:
        return None
    return {"H": Hi, "inliers": m["inliers"], "matches": m["matches"],
            "model": m.get("model", ""), "ncc": m.get("ncc", 0.0),
            "overlap": m.get("overlap", 0), "src": m["dst"], "dst": m["src"]}


def match_graph(norm, feats, bboxes, idxs, verbose=True):
    """Match every pair of captures once.  Returns {(i, j): pairing}, both ways.

    `(i, j)` holds the transform taking image *j* onto image *i*.  Doing this up
    front — rather than matching each capture to its predecessor — is what makes it
    possible to CHOOSE where to start, and to attach a capture to whichever placed
    image it actually overlaps.
    """
    pairs = {}
    for a, i in enumerate(idxs):
        for j in idxs[a + 1:]:
            m = _match_pair(norm, feats[i], feats[j], bboxes[j], bboxes[i])
            if m is None:
                continue
            rev = _reverse(m)
            if rev is None:
                continue
            pairs[(i, j)] = m
            pairs[(j, i)] = rev
    if verbose:
        linked = sorted(((i, j, m["inliers"], m.get("ncc", 0.0))
                         for (i, j), m in pairs.items() if i < j),
                        key=lambda t: -t[2])
        print(f"    graph: {len(linked)} linked pair(s) of "
              f"{len(idxs) * (len(idxs) - 1) // 2}"
              + ("  " + ", ".join(f"{i}-{j}:{n}in/{c:.2f}ncc"
                                  for i, j, n, c in linked[:8]) if linked else ""))
    return pairs


def _components(idxs, pairs):
    """Connected components of the match graph, as a list of index sets."""
    seen, comps = set(), []
    for start in idxs:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            k = stack.pop()
            if k in comp:
                continue
            comp.add(k)
            stack += [j for (i, j) in pairs if i == k and j not in comp]
        seen |= comp
        comps.append(comp)
    return comps


def choose_seed(idxs, pairs, verbose=True):
    """Pick the capture to build the mosaic around.

    The seed defines the mosaic's coordinate frame, so every other image's error is
    accumulated relative to it — starting from a weakly-matched capture is what
    limits a mosaic to two frames.  Preference order:

      1. the largest connected component (it bounds how many can ever be placed),
      2. within it, the image with the most total inlier matches to the others,
      3. then the most partners, then the earliest capture.

    Returns (seed_index, component).
    """
    comps = _components(idxs, pairs)
    best_comp = max(comps, key=lambda c: (len(c), -min(c)))

    def score(i):
        links = [pairs[(i, j)]["inliers"] for j in best_comp if (i, j) in pairs]
        return (sum(links), len(links), -i)

    seed = max(best_comp, key=score)
    if verbose:
        s = score(seed)
        print(f"    seed: image {seed} ({s[0]} inliers across {s[1]} partner(s)); "
              f"largest group has {len(best_comp)} of {len(idxs)} capture(s)")
    return seed, best_comp


def _resection(j, placed, pairs, bbox_j, verbose=True):
    """Where image `j` sits in mosaic coordinates, given what is already placed.

    `placed` maps image index -> its transform into the reference (seed) frame.

    One neighbour: compose through it.  SEVERAL: pool every neighbour's inlier
    correspondences — each neighbour's points pushed into reference coordinates by
    its own transform — and fit a single homography to the lot with RANSAC.  That is
    the standard way to bring in an image that overlaps more than one of its
    predecessors: it uses all the evidence at once and cannot inherit whichever
    single chain happened to be picked.

    Returns (H, contributing_neighbours, n_points) or (None, [], 0).
    """
    nbrs = sorted((i for i in placed if (i, j) in pairs),
                  key=lambda i: -pairs[(i, j)]["inliers"])
    if not nbrs:
        return None, [], 0
    if len(nbrs) == 1:
        i = nbrs[0]
        H = placed[i] @ pairs[(i, j)]["H"]
        return (H if _sane(H, bbox_j) else None), nbrs, pairs[(i, j)]["inliers"]

    src = np.vstack([pairs[(i, j)]["src"] for i in nbrs])
    dst = np.vstack([cv2.perspectiveTransform(pairs[(i, j)]["dst"], placed[i])
                     for i in nbrs])
    for H, inl, _model in _models(src, dst):
        # RANSAC over the pooled set also ARBITRATES: a neighbour that disagrees
        # with the rest is outvoted here rather than being averaged in.
        if int(inl.sum()) >= MIN_INLIERS and _sane(H, bbox_j):
            return H, nbrs, int(inl.sum())
    # The joint fit can fail when two neighbours disagree (one of them is
    # mis-registered).  Fall back to the strongest single link rather than
    # dropping an image that plainly matches something.
    i = nbrs[0]
    H = placed[i] @ pairs[(i, j)]["H"]
    if verbose:
        print(f"    [{j}] joint fit over {len(nbrs)} neighbours failed; "
              f"chaining through {i}")
    return (H if _sane(H, bbox_j) else None), [i], pairs[(i, j)]["inliers"]


def mask_for(img, mask=None):
    """The selection mask for a patch — supplied, or derived from non-black pixels."""
    if mask is not None:
        return mask
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (g > 0).astype(np.uint8) * 255


def _grow(idxs, pairs, bboxes, seed, verbose):
    """Place a component's images in the seed's frame, strongest link first.

    Order is chosen rather than inherited from capture order: an image placed off a
    confident overlap becomes a trustworthy anchor for the next, whereas the weakest
    link, placed early, poisons everything after it.

    Returns (G, order, unplaced): G maps index -> 3x3 into seed coordinates.
    """
    G = {seed: np.eye(3, dtype=np.float64)}
    order = [(seed, [], None)]
    left = [i for i in idxs if i != seed]
    unplaced = []
    while left:
        def strength(j, _G=G):
            return sum(pairs[(i, j)]["inliers"] for i in _G if (i, j) in pairs)
        j = max(left, key=lambda k: (strength(k), -k))
        if strength(j) == 0:
            break                                 # nothing left touches the mosaic
        left.remove(j)
        H, nbrs, n_in = _resection(j, G, pairs, bboxes[j], verbose)
        if H is None:
            unplaced.append((j, f"implausible transform from {len(nbrs)} neighbour(s)"))
            if verbose:
                print(f"    [{j}] SKIPPED - transform rejected as implausible")
            continue
        G[j] = H
        order.append((j, nbrs, n_in))
    unplaced += [(j, "no match with any placed capture") for j in left]
    return G, order, unplaced


def _render(imgs, masks, G, bboxes, upscale, verbose=False):
    """Warp every placed image onto one canvas and blend.  Returns (bgr, mask, px, T).

    `T` maps the seed frame to canvas pixels, so a caller can compose it with each
    input's own transform and know where every ORIGINAL patch ended up.

    The canvas is sized to the UNION of what was placed, in seed coordinates: the
    extracts are small patches adrift in a mostly-black frame, so a frame-sized
    canvas would be nearly empty and render at patch resolution.
    """
    quad = np.vstack([cv2.perspectiveTransform(_corners(bboxes[i]), G[i]).reshape(-1, 2)
                      for i in G])
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    span = max(1.0, float(max(x1 - x0, y1 - y0)))
    pad = CANVAS_MARGIN * span
    span_w, span_h = float(x1 - x0) + 2 * pad, float(y1 - y0) + 2 * pad
    u = min(upscale, MAX_DIM / max(span_w, span_h))
    cw = max(1, int(round(span_w * u)))
    ch = max(1, int(round(span_h * u)))
    T = np.array([[u, 0.0, -u * (x0 - pad)],
                  [0.0, u, -u * (y0 - pad)],
                  [0.0, 0.0, 1.0]], np.float64)

    acc = np.zeros((ch, cw, 3), np.float32)       # running sum for the blend
    cnt = np.zeros((ch, cw), np.float32)          # contributions per pixel
    px = {}
    for i in G:
        H = T @ G[i]
        warp = cv2.warpPerspective(imgs[i], H, (cw, ch))
        wm = cv2.warpPerspective(masks[i], H, (cw, ch), flags=cv2.INTER_NEAREST)
        sel = wm > 0
        acc[sel] += warp[sel].astype(np.float32)
        cnt[sel] += 1.0
        px[i] = int(sel.sum())
    out = np.zeros((ch, cw, 3), np.uint8)
    nz = cnt > 0
    out[nz] = (acc[nz] / cnt[nz][:, None]).astype(np.uint8)
    return out, (nz.astype(np.uint8) * 255), px, T


def stitch_pass(pieces, upscale=1.0, verbose=True):
    """ONE agglomeration pass.  `pieces` is a list of (bgr, mask, members, xforms).

    Every piece is matched against every other, the match graph is split into
    connected components, and each component is collapsed into a single blended
    mosaic.  Pieces that matched nothing come through untouched, so nothing is ever
    discarded — it just waits for a later pass, when the others have grown.

    `xforms` maps each ORIGINAL image index to its 3x3 into this piece's own pixel
    coordinates, and is composed forward on every merge.  That is what lets the
    final mosaic be re-rendered once, at full upscale, straight from the source
    patches — rather than resampling a mosaic of a mosaic at every pass.

    Returns (new_pieces, detector_name, pair_records, placements).
    """
    det, norm, name = _detector()
    n = len(pieces)
    idxs = list(range(n))
    bboxes = {i: content_bbox(pieces[i][1]) for i in idxs}
    idxs = [i for i in idxs if bboxes[i] is not None]
    feats = {i: _feat(det, pieces[i][0], pieces[i][1]) for i in idxs}
    pairs = match_graph(norm, feats, bboxes, idxs, verbose)
    records = sorted(
        ({"a": i, "b": j, "inliers": m["inliers"], "ncc": m.get("ncc", 0.0),
          "model": m.get("model", "")}
         for (i, j), m in pairs.items() if i < j),
        key=lambda p: -p["inliers"])

    out = []
    placements = []
    for comp in _components(idxs, pairs):
        if len(comp) == 1:
            i = comp.pop()
            out.append(pieces[i])                 # nothing to merge it with yet
            continue
        sub = sorted(comp)
        seed, _ = choose_seed(sub, pairs, verbose)
        G, order, unplaced = _grow(sub, pairs, bboxes, seed, verbose)
        imgs = {i: pieces[i][0] for i in G}
        msks = {i: pieces[i][1] for i in G}
        mos, mmask, px, T = _render(imgs, msks, G, bboxes, upscale)
        members = [m for i in G for m in pieces[i][2]]
        # Compose each ORIGINAL patch's transform forward into the merged canvas.
        xforms = {m: T @ G[i] @ X
                  for i in G for m, X in pieces[i][3].items()}
        for j, nbrs, n_in in order:
            placements.append({"index": min(pieces[j][2]), "members": pieces[j][2],
                               "px": px[j], "inliers": n_in, "via": list(nbrs)})
            if verbose:
                if not nbrs:
                    print(f"    [{j}] SEED, {px[j]}px on a "
                          f"{mos.shape[1]}x{mos.shape[0]} canvas")
                else:
                    how = f"via {nbrs[0]}" if len(nbrs) == 1 else f"via {nbrs} jointly"
                    print(f"    [{j}] placed {how}, {px[j]}px, {n_in} inliers")
        out.append((mos, mmask, members, xforms))
        for j, why in unplaced:                   # keep it; a later pass may take it
            if verbose:
                print(f"    [{j}] not merged this pass - {why}")
            out.append(pieces[j])
    # Pieces whose content bbox was empty never entered the graph.
    out += [pieces[i] for i in range(n) if bboxes.get(i) is None]
    return out, name, records, placements


def stitch(images, masks=None, verbose=True):
    """Register and blend red-eye extracts into a mosaic, agglomeratively.

    `images` are BGR patches on black (cap.RedeyeResult.extract); `masks` are the
    matching selections, derived from non-black pixels when omitted.

    NOT built around a single root.  One pass matches every pair, splits the graph
    into connected components and collapses each into its own mosaic; captures that
    matched nothing survive untouched.  The pass then REPEATS over those results —
    and this is the point of doing it that way: a blended mosaic of three patches
    carries more texture than any of the three alone, so two groups that could not
    be linked through any single capture often link once each has been merged.
    Passes continue until a pass stops reducing the count, i.e. nothing more can be
    joined.  The largest surviving mosaic is returned.

    Returns (mosaic_bgr, info); info has used, skipped, detector, placements, seed,
    order, pairs, passes and pieces.  Indices refer to positions in `images`.
    """
    info = {"used": 0, "skipped": [], "detector": "", "placements": [],
            "seed": None, "order": [], "pairs": [], "passes": 0, "pieces": []}
    imgs = list(images)
    msks = list(masks) if masks is not None else []
    msks += [None] * (len(imgs) - len(msks))

    src_mask = {i: mask_for(imgs[i], msks[i])
                for i, im in enumerate(imgs) if im is not None}
    src_bbox = {i: content_bbox(m) for i, m in src_mask.items()}
    keep = [i for i in src_mask if src_bbox[i] is not None]
    pieces = [(imgs[i], src_mask[i], [i], {i: np.eye(3)}) for i in keep]
    if not pieces:
        return None, info
    if len(pieces) == 1:
        info["used"], info["seed"] = 1, pieces[0][2][0]
        info["pieces"] = [list(pieces[0][2])]
        return pieces[0][0].copy(), info

    if verbose:
        print(f"  mosaic: {len(pieces)} image(s)")

    prev_n = len(pieces) + 1
    while len(pieces) < prev_n and len(pieces) > 1 and info["passes"] < MAX_PASSES:
        prev_n = len(pieces)
        info["passes"] += 1
        if verbose:
            print(f"  --- pass {info['passes']}: {prev_n} piece(s) ---")
        # Every pass renders at NATIVE scale.  Upscaling here would put an enlarged
        # mosaic next to un-enlarged singletons, and the plausibility check compares
        # the two patches' areas — so every cross-scale pair would be thrown out as
        # a wild transform, which is exactly the merge a later pass exists to find.
        pieces, name, records, placed = stitch_pass(pieces, 1.0, verbose)
        info["detector"] = name
        info["placements"] += [dict(p, **{"pass": info["passes"]}) for p in placed]
        if info["passes"] == 1:
            info["pairs"] = records
        if verbose:
            print(f"      -> {len(pieces)} piece(s): "
                  f"{[sorted(p[2]) for p in pieces]}")

    pieces.sort(key=lambda p: -len(p[2]))
    best = pieces[0]
    info["pieces"] = [sorted(p[2]) for p in pieces]
    info["used"] = len(best[2])
    info["order"] = sorted(best[2])
    info["seed"] = info["order"][0] if info["order"] else None

    # Re-render the winner ONCE, at full upscale, from the ORIGINAL patches.  Each
    # one is resampled a single time from its own pixels rather than through a
    # mosaic-of-a-mosaic, and the sub-pixel registration the passes established is
    # what the extra resolution is actually carrying.
    out = best[0]
    if len(best[2]) > 1 and MOSAIC_UPSCALE > 1.0:
        G = best[3]
        out, _m, _px, _T = _render({i: imgs[i] for i in G},
                                   {i: src_mask[i] for i in G},
                                   G, src_bbox, MOSAIC_UPSCALE)
    for p in pieces[1:]:
        for i in p[2]:
            info["skipped"].append(
                {"index": i, "reason": "could not be joined to the main mosaic"})
    if verbose and info["skipped"]:
        print(f"    unplaced: {[s['index'] for s in info['skipped']]}")
    return out, info


def keypoint_view(img, mask=None, det=None):
    """One patch AS THE DETECTOR SEES IT, with its keypoints drawn.

    Returns (bgr, n_keypoints), or (None, 0) if the patch has no content.

    The view is in ANALYSIS space — cropped to the patch, enlarged to
    ANALYSIS_SIZE and contrast-boosted (see `_analysis`) — because that is the
    image the matcher actually describes, and because at the source scale of
    ~31x26px the keypoints would be one solid overlapping blob.  The eroded mask
    boundary is outlined too: the detector is confined inside it, so a patch whose
    keypoints all crowd that edge is one whose features are the patch's own
    border rather than fundus texture.
    """
    if det is None:
        det, _, _ = _detector()
    a_img, a_mask, _S = _analysis(img, mask)
    if a_img is None:
        return None, 0
    kp = det.detect(a_img, a_mask)
    keep, drop = _boundary_clear(kp, a_mask)
    vis = cv2.cvtColor(a_img, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(a_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (140, 90, 40), 1)
    # RICH keypoints: each circle's size is the scale it was found at and the
    # radius line its orientation — both of which have to agree between two
    # patches for the descriptors to match, so they are worth seeing.  Dim red =
    # discarded because its support reached the black surround; green = used.
    rich = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    if drop:
        vis = cv2.drawKeypoints(vis, [kp[i] for i in drop], None,
                                color=(60, 60, 190), flags=rich)
    vis = cv2.drawKeypoints(vis, [kp[i] for i in keep], None,
                            color=(0, 255, 0), flags=rich)
    return vis, len(keep)


def keypoint_sheet(images, masks=None, labels=None, cols=4, tile=260,
                   verbose=True):
    """Grid of every extract with its keypoints.  Returns (sheet, labels).

    How many keypoints each patch yields is the single number that decides whether
    a session can be stitched at all: below MIN_MATCHES there is nothing to match
    with, and the difference between a session that mosaics and one that does not
    is usually visible here long before it shows up in the pair graph.
    """
    det, _, name = _detector()
    masks = list(masks) if masks is not None else [None] * len(images)
    masks += [None] * (len(images) - len(masks))
    views, labs, counts = [], [], []
    for i, im in enumerate(images):
        vis, n = keypoint_view(im, masks[i], det)
        if vis is None:
            vis = np.zeros((tile, tile, 3), np.uint8)
            n = 0
        views.append(vis)
        counts.append(n)
        base = labels[i] if labels and i < len(labels) else str(i)
        labs.append(f"{base} {n}kp")
    if verbose and counts:
        weak = [i for i, n in enumerate(counts) if n < MIN_KEYPOINTS]
        print(f"  keypoints ({name}): total {sum(counts)}, "
              f"min {min(counts)}, median {sorted(counts)[len(counts) // 2]}, "
              f"max {max(counts)}")
        if weak:
            print(f"    too few to match anything (<{MIN_KEYPOINTS}): {weak}")
    return contact_sheet(views, cols=cols, tile=tile, labels=labs), labs


def show_keypoints(session_dir, out_path=None, show=True, verbose=True):
    """Build (and optionally display/save) the keypoint sheet for a session."""
    imgs, masks, paths = load_session(session_dir)
    if not imgs:
        if verbose:
            print(f"  no captures in {session_dir}")
        return None
    labels = [os.path.basename(p).replace("_extract.png", "")[7:] for p in paths]
    sheet, _ = keypoint_sheet(imgs, masks, labels, verbose=verbose)
    if sheet is None:
        return None
    if out_path:
        cv2.imwrite(out_path, sheet)
        if verbose:
            print(f"  keypoint sheet written: {out_path}")
    if show:
        vis = sheet
        if vis.shape[0] > 900:
            s = 900.0 / vis.shape[0]
            vis = cv2.resize(vis, (int(vis.shape[1] * s), 900))
        cv2.imshow("Red-eye keypoints", vis)
        cv2.waitKey(1)
    return sheet


def crop_to_content(img, pad=8):
    """Trim the black border left by the oversized canvas."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ys, xs = np.nonzero(gray)
    if not len(xs):
        return img
    x0 = max(0, xs.min() - pad); x1 = min(img.shape[1], xs.max() + pad + 1)
    y0 = max(0, ys.min() - pad); y1 = min(img.shape[0], ys.max() + pad + 1)
    return img[y0:y1, x0:x1]


def load_session(session_dir):
    """Load a session's red-eye extracts (+ masks) in capture order.

    Returns (images, masks, paths)."""
    imgs, masks, paths = [], [], []
    for p in sorted(glob.glob(os.path.join(session_dir, "redeye_*_extract.png"))):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        mp = p.replace("_extract.png", "_mask.png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp) else None
        if m is not None and m.shape[:2] != img.shape[:2]:
            m = None
        imgs.append(img)
        masks.append(m)
        paths.append(p)
    return imgs, masks, paths


def contact_sheet(images, cols=3, tile=260, labels=None):
    """Grid of the constituent patches, for eyeballing what went into a mosaic."""
    if not images:
        return None
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        s = min(tile / max(1, w), tile / max(1, h))
        rs = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        r, c = divmod(i, cols)
        y, x = r * tile, c * tile
        sheet[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
        cv2.rectangle(sheet, (x, y), (x + tile - 1, y + tile - 1), (40, 40, 40), 1)
        lab = labels[i] if labels and i < len(labels) else str(i)
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(sheet, lab, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, th, cv2.LINE_AA)
    return sheet


def build(session_dir, out_path=None, verbose=True):
    """Stitch a session directory.  Returns (mosaic, info, paths)."""
    imgs, masks, paths = load_session(session_dir)
    if len(imgs) < 2:
        if verbose:
            print(f"  mosaic: need >= 2 extracts, found {len(imgs)} in {session_dir}")
        return None, {"used": len(imgs), "skipped": [], "detector": "",
                      "placements": []}, paths
    mos, info = stitch(imgs, masks, verbose=verbose)
    mos = crop_to_content(mos)
    if out_path and mos is not None:
        cv2.imwrite(out_path, mos)
        if verbose:
            print(f"  mosaic written: {out_path}")
    return mos, info, paths


def browse(session_dir, window="Session captures"):
    """Step through a session's captures one at a time, at full size.

        SPACE / n / right   next        m   extract <-> mask
        p / left            previous    ESC / q     close
    """
    imgs, masks, paths = load_session(session_dir)
    if not imgs:
        print(f"  no captures in {session_dir}")
        return False
    i, show_mask = 0, False
    print(f"  {len(imgs)} capture(s) - SPACE/n/right=next, p/left=prev, m=mask, "
          f"ESC/q=close")
    try:
        while True:
            if show_mask and masks[i] is not None:
                vis = cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR)
            else:
                vis = imgs[i].copy()
            if vis.shape[0] > 900:
                s = 900.0 / vis.shape[0]
                vis = cv2.resize(vis, (int(vis.shape[1] * s), 900))
            px = int(cv2.countNonZero(masks[i])) if masks[i] is not None else -1
            label = (f"[{i + 1}/{len(imgs)}] "
                     f"{os.path.basename(paths[i]).replace('_extract.png', '')}"
                     + (f"  {px}px" if px >= 0 else "")
                     + ("  MASK" if show_mask else ""))
            for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(vis, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, th, cv2.LINE_AA)
            cv2.imshow(window, vis)
            k = cv2.waitKeyEx(0)
            if k in (27, ord('q')):
                break
            if k in (32, ord('n'), 65363, 2555904):
                i = (i + 1) % len(imgs)
            elif k in (ord('p'), 65361, 2424832):
                i = (i - 1) % len(imgs)
            elif k == ord('m'):
                show_mask = not show_mask
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    return True


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: python mosaic.py <session_dir> [-o out.png] [--sheet s.png]"
              " [--keypoints [kp.png]] [--browse]")
        return 2
    session = argv[1]
    if "--browse" in argv:
        return 0 if browse(session) else 1
    if "--keypoints" in argv:
        # Optional path: --keypoints on its own writes next to the session.
        k = argv.index("--keypoints") + 1
        kp_out = argv[k] if k < len(argv) and not argv[k].startswith("-") \
            else os.path.join(session, "keypoints.png")
        return 0 if show_keypoints(session, kp_out, show=False) is not None else 1
    out = None
    sheet = None
    if "-o" in argv:
        out = argv[argv.index("-o") + 1]
    else:
        out = os.path.join(session, "mosaic.png")
    if "--sheet" in argv:
        sheet = argv[argv.index("--sheet") + 1]
    mos, info, paths = build(session, out)
    if sheet:
        imgs, _, _ = load_session(session)
        s = contact_sheet(imgs, labels=[os.path.basename(p)[:14] for p in paths])
        if s is not None:
            cv2.imwrite(sheet, s)
            print(f"  contact sheet written: {sheet}")
    if mos is None:
        return 1
    print(f"  used {info['used']}/{len(paths)}, skipped {len(info['skipped'])}, "
          f"detector {info['detector']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
