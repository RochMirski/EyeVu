"""Keypoint detection, RootSIFT description, spatial spread, landmark hooks.

SIFT on a raw retinal frame is close to useless.  On a properly normalised green
channel -- illumination flattened, locally equalised, masked (see preprocess) --
it is respectable, and RootSIFT makes it measurably better for one line of code.

The detector sits behind `Detector`, deliberately.  Learned retinal keypoint
detectors (GLAMpoints, ICCV 2019; SuperRetina, ECCV 2022) genuinely outperform
SIFT on this imagery and are the accuracy ceiling here, but they need a runtime
this deployment does not have (no PyTorch, no ONNX, no GPU, 512 MB).  Anything
implementing `detect_and_describe` can be dropped in later without the rest of
the pipeline noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Features:
    """Keypoints and descriptors for one patch, in analysis coordinates."""
    index: int
    xy: np.ndarray = None               # (N, 2) float32
    size: np.ndarray = None             # (N,) float32, SIFT diameter
    angle: np.ndarray = None            # (N,) float32, degrees
    response: np.ndarray = None         # (N,) float32
    desc: np.ndarray = None             # (N, 128) float32, RootSIFT
    landmarks: dict = field(default_factory=dict)   # name -> (x, y)

    def __len__(self):
        return 0 if self.xy is None else len(self.xy)


class Detector:
    """Interface a keypoint backend must satisfy.

    `detect_and_describe(image, mask)` takes a uint8 single-channel analysis
    image and a uint8 mask, and returns (cv2.KeyPoint list, descriptors) with
    descriptors as float32 (N, D), L2-comparable.  That is the entire contract.
    """

    def detect_and_describe(self, image, mask):        # pragma: no cover
        raise NotImplementedError


class SIFTDetector(Detector):
    """OpenCV SIFT, with thresholds loosened for low-contrast tissue."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._sift = cv2.SIFT_create(
            nfeatures=0,
            contrastThreshold=float(cfg.sift_contrast_threshold),
            edgeThreshold=float(cfg.sift_edge_threshold))

    def detect_and_describe(self, image, mask):
        return self._sift.detectAndCompute(image, mask)


def rootsift(desc):
    """L1-normalise each descriptor, then take the elementwise square root.

    One line, and it reliably improves SIFT matching: the Hellinger/Bhattacharyya
    distance it induces is far better behaved on the long-tailed histograms that
    low-contrast tissue produces than plain L2 on raw counts.
    """
    if desc is None or len(desc) == 0:
        return desc
    d = np.asarray(desc, np.float32)
    d /= np.maximum(d.sum(axis=1, keepdims=True), 1e-7)
    return np.sqrt(d, out=d)


def _grid_spread(xy, response, mask_shape, cfg):
    """Keep the strongest few keypoints per grid cell.

    Raw response ranking piles everything onto whatever high-contrast structure
    a patch happens to contain -- on retinal imagery, the optic disc and the
    main arcades -- leaving the rest of the patch undescribed and unmatchable.
    Bucketing forces the coverage that a global ranking will not give.
    """
    h, w = mask_shape
    n = max(1, int(cfg.spread_grid))
    gx = np.clip((xy[:, 0] / max(w, 1) * n).astype(int), 0, n - 1)
    gy = np.clip((xy[:, 1] / max(h, 1) * n).astype(int), 0, n - 1)
    cell = gy * n + gx
    keep = []
    per = max(1, int(cfg.spread_per_bucket))
    for c in np.unique(cell):
        idx = np.nonzero(cell == c)[0]
        if len(idx) > per:
            idx = idx[np.argsort(-response[idx])[:per]]
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.arange(0)


def _anms(xy, response, k):
    """Adaptive non-maximal suppression (Brown, Szeliski & Winder 2005).

    Each keypoint's suppression radius is the distance to the nearest
    significantly stronger keypoint; keeping the largest radii gives a spatially
    even set without imposing a grid.  O(N^2), which is nothing at N ~ 200.
    """
    if len(xy) <= k:
        return np.arange(len(xy))
    order = np.argsort(-response)
    xy, response = xy[order], response[order]
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    stronger = response[None, :] > response[:, None] * 1.11
    d2 = np.where(stronger, d2, np.inf)
    radius = np.sqrt(d2.min(axis=1))
    radius[0] = np.inf                              # the global maximum survives
    return np.sort(order[np.argsort(-radius)[:k]])


def detect(patch, cfg, detector=None):
    """Detect and describe one prepared patch.  Returns a `Features`.

    Runs on the flattened, masked analysis image and inside the ERODED mask, so
    no keypoint centre sits on the illumination boundary.
    """
    f = Features(index=patch.index)
    if patch.image is None or patch.mask is None:
        return f
    det = detector or SIFTDetector(cfg)
    kp, desc = det.detect_and_describe(patch.image, patch.mask)
    if desc is None or not len(kp):
        return f

    xy = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
    size = np.array([k.size for k in kp], np.float32)
    angle = np.array([k.angle for k in kp], np.float32)
    resp = np.array([k.response for k in kp], np.float32)
    desc = np.asarray(desc, np.float32)

    if cfg.spread_mode == "grid":
        sel = _grid_spread(xy, resp, patch.mask.shape[:2], cfg)
    elif cfg.spread_mode == "anms":
        sel = _anms(xy, resp, int(cfg.max_keypoints))
    else:
        sel = np.arange(len(xy))
    xy, size, angle, resp, desc = (xy[sel], size[sel], angle[sel],
                                   resp[sel], desc[sel])

    # Cap AFTER spreading, so the cap trims a spatially even set rather than
    # undoing the spread by re-ranking on response alone.
    if len(xy) > cfg.max_keypoints:
        keep = np.sort(np.argsort(-resp)[:int(cfg.max_keypoints)])
        xy, size, angle, resp, desc = (xy[keep], size[keep], angle[keep],
                                       resp[keep], desc[keep])

    f.xy, f.size, f.angle, f.response = xy, size, angle, resp
    f.desc = rootsift(desc) if cfg.rootsift else desc
    return f


# ══ named landmarks ═══════════════════════════════════════════════════════
#
# A landmark is a point a detector can find in more than one patch by identity
# rather than by descriptor similarity -- the optic disc centre being the
# obvious one.  Injected as a high-confidence prior correspondence, it anchors
# any pair where both patches contain it, which is worth a lot when a pair
# otherwise has three matches.
#
# NOTHING POPULATES THIS YET, and that is a deliberate, honest gap: the repo has
# no optic disc detector (every "disc" in cap.py is the PUPIL dark-disc, a
# different thing), and at a median patch size of 45x32 native px the optic disc
# would span a handful of pixels -- there is no defensible way to detect it at
# this scale.  The mechanism is here and is exercised by the tests, so a
# detector can be added without touching the matcher.

# DEAD CODE - no caller anywhere in the repo (AST reachability scan, 2026-08-19).
# Commented out, not deleted.  See docs/REPO_MAP.md section 5.1.
# def set_landmark(features, name, xy):
#     """Record a named landmark for a patch, in analysis coordinates."""
#     features.landmarks[str(name)] = (float(xy[0]), float(xy[1]))


def shared_landmarks(fa, fb):
    """Landmarks present in both patches, as (src_xy, dst_xy) arrays.

    Returned as prior correspondences for `pairwise`: they are appended ahead of
    the descriptor matches so PROSAC draws them first.
    """
    names = sorted(set(fa.landmarks) & set(fb.landmarks))
    if not names:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64), []
    src = np.array([fa.landmarks[n] for n in names], np.float64)
    dst = np.array([fb.landmarks[n] for n in names], np.float64)
    return src, dst, names
