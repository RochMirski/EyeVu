"""Masking, illumination flattening, CLAHE and per-patch quality metrics.

Everything downstream works in "analysis space": each patch cropped to its own
content and rescaled to a COMMON angular sampling.  Two properties of that
rescaling matter enough to state plainly.

1.  It UPSAMPLES.  The real patches are a median 45x32 px of retina adrift in a
    480x640 or 960x1280 frame (0.28% fill).  At native scale SIFT finds almost
    nothing on them; enlarged, the same patches become describable.  This is not
    an optimisation, it is what makes feature matching possible here at all.

2.  The scale factor depends ONLY on the frame resolution, never on the patch's
    own bounding box.  Sizing each patch to a fixed analysis size -- the obvious
    thing, and what the previous implementation did -- gives a 35px patch x6.86
    and a 45px patch x5.33, inventing ~29% of relative scale between two views
    of the same retina.  A similarity model hides that by spending a DOF on it.
    The 3-DOF rotation model has no such DOF and would simply fail.  Sessions
    really do mix 480x640 and 960x1280 captures, so this has to be right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Patch:
    """One capture, prepared for feature work.  Pixel data is analysis-scale."""
    index: int
    path: str = ""
    frame_shape: tuple = (0, 0)         # (h, w) of the ORIGINAL frame
    bbox: tuple = ()                    # (x0, y0, w, h) content bbox, native px
    S: np.ndarray = None                # 3x3: original coords -> analysis coords
    scale: float = 1.0                  # the s inside S
    image: np.ndarray = None            # uint8 analysis image, flattened green
    raw: np.ndarray = None              # uint8 analysis green, before flattening
    mask: np.ndarray = None             # uint8 analysis mask, eroded, specular-free
    mask_full: np.ndarray = None        # uint8 analysis mask, NOT eroded
    # Native-resolution crop, kept for compositing.  At most ~125x101x3 = 38 kB,
    # so holding a whole session costs under 2 MB and the compositing pass needs
    # no disk IO and no second resampling -- output pixels come straight from
    # source pixels.  The crop origin is bbox[:2]; analysis coords relate to
    # crop-local coords by the single factor `scale`.
    native_bgr: np.ndarray = None       # uint8 (h, w, 3) crop of the original
    native_mask: np.ndarray = None      # uint8 (h, w) matching mask
    native_weight: np.ndarray = None    # float32 blend weight, 0 at the rim
    radius: float = 0.0                 # equivalent radius, analysis px
    area: int = 0                       # eroded mask area, analysis px
    focal: float = 0.0                  # focal length in analysis px
    centre: tuple = (0.0, 0.0)          # principal point in analysis coords
    quality: dict = field(default_factory=dict)
    accepted: bool = True
    reason: str = ""


def content_bbox(mask):
    """(x0, y0, w, h) of the non-zero content, or None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def analysis_scale(frame_width, cfg):
    """Analysis pixels per native pixel, for a frame of this width.

    Depends on resolution alone, so two patches of the same retina land at the
    same angular sampling whatever size their own blobs happen to be.
    """
    return float(cfg.analysis_upsample) * float(cfg.ref_frame_width) / float(frame_width)


def equivalent_radius(mask):
    """Radius of the disc with the same area as this mask.

    The illuminated region here is NOT a circle -- it is whatever shape the
    red-shift selection returned, gated to the pupil -- so there is no aperture
    radius to read off.  Area-equivalent radius is the honest stand-in, and it
    is what the erosion and flattening scales are expressed against.
    """
    a = float(np.count_nonzero(mask))
    return float(np.sqrt(a / np.pi)) if a > 0 else 0.0


def _masked_blur(img, mask, sigma):
    """Gaussian blur that only ever sees valid pixels (normalised convolution).

    A plain blur near the patch rim averages in the black surround and produces
    a false illumination gradient exactly where the patch is thinnest -- which
    is where most of the keypoints are.  Blurring image and mask separately and
    dividing keeps the estimate inside the data.
    """
    m = (mask > 0).astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def _poly_illumination(green, mask, order=2):
    """Least-squares low-order polynomial fit to the illumination, over the mask.

    Returns the fitted surface, strictly positive.
    """
    sel = mask > 0
    ys, xs = np.nonzero(sel)
    if len(xs) < 24:
        return np.full(green.shape, max(float(green[sel].mean()) if sel.any()
                                        else 1.0, 1.0), np.float32)
    h, w = green.shape[:2]
    # Normalised coordinates keep the normal equations well conditioned.
    nx = (xs - w * 0.5) / max(w * 0.5, 1.0)
    ny = (ys - h * 0.5) / max(h * 0.5, 1.0)
    terms = [np.ones_like(nx), nx, ny]
    if order >= 2:
        terms += [nx * nx, nx * ny, ny * ny]
    A = np.stack(terms, axis=1)
    b = green[sel].astype(np.float64)
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(green.shape, max(float(b.mean()), 1.0), np.float32)

    gy, gx = np.mgrid[0:h, 0:w]
    NX = (gx - w * 0.5) / max(w * 0.5, 1.0)
    NY = (gy - h * 0.5) / max(h * 0.5, 1.0)
    full = [np.ones_like(NX, np.float64), NX, NY]
    if order >= 2:
        full += [NX * NX, NX * NY, NY * NY]
    surf = sum(ci * ti for ci, ti in zip(c, full))
    return np.maximum(surf, 1.0).astype(np.float32)


def flatten_illumination(green, mask, radius, cfg):
    """Remove the illumination falloff, then equalise locally.

    Strong radial falloff and variable exposure between patches are the two
    things that stop the same piece of retina describing the same way in two
    captures, so removing them is what makes descriptors comparable at all.

    THE OBVIOUS METHOD DOES NOT WORK HERE, and it is worth saying why.
    Subtracting a heavily blurred copy is a high-pass, and a high-pass can only
    separate illumination from anatomy if the two live in different spatial
    frequency bands.  On a 640px aperture they do.  On THESE patches -- ~15
    native px of equivalent radius -- they do not: the illumination varies
    appreciably over the same few pixels the anatomy does.  Measured on
    synthetic sessions with known ground truth, every Gaussian sigma from
    0.125 to 1.0 of the radius left true overlapping pairs correlating at only
    0.08-0.25, because a sigma small enough to remove the falloff also removes
    the retina.

    A LOW-ORDER PARAMETRIC MODEL separates them by FORM instead of by
    frequency: vignetting really is a smooth quadratic in position, and a
    6-coefficient surface has nowhere to hide vessel structure however small the
    patch is.  Division is the right combiner because the falloff is
    multiplicative.  That is the default; the Gaussian modes are kept for larger
    patches, where the frequency argument does hold.
    """
    g = green.astype(np.float32)
    if cfg.flatten_mode == "none":
        out = g
    elif cfg.flatten_mode == "poly":
        surf = _poly_illumination(g, mask, int(cfg.flatten_poly_order))
        ref = float(g[mask > 0].mean()) if np.any(mask > 0) else 128.0
        out = g / surf * max(ref, 1.0)
    else:
        sigma = max(1.0, cfg.flatten_sigma_frac * max(radius, 1.0))
        lo = _masked_blur(g, mask, sigma)
        if cfg.flatten_mode == "divide":
            out = 128.0 * g / np.maximum(lo, 1.0)
        else:                                   # "subtract"
            out = g - lo + 128.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    if cfg.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip),
                                tileGridSize=(int(cfg.clahe_tile),) * 2)
        out = clahe.apply(out)
    out[mask == 0] = 0
    return out


def specular_mask(bgr_or_gray, cfg):
    """Near-saturated pixels, dilated -- an exclusion zone for feature work.

    Mostly a second line of defence: cap.redeye_extract already thresholds and
    dilates speculars out of the mask before the extract is written, so on real
    captures this usually finds nothing.  It matters for synthetic sessions and
    for any future capture path that does not pre-exclude them.
    """
    g = (bgr_or_gray if bgr_or_gray.ndim == 2
         else cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY))
    _, spec = cv2.threshold(g, int(cfg.specular_thresh), 255, cv2.THRESH_BINARY)
    k = int(max(1, cfg.specular_dilate_px))
    if k > 1:
        spec = cv2.dilate(spec, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)))
    return spec


def quality_metrics(raw_green, mask, radius, cfg, scale=1.0):
    """Per-patch quality numbers, and whether they pass.

    Returns (metrics, accepted, reason).  A failing patch is FLAGGED, never
    silently discarded -- the metrics are recorded for accepted and rejected
    patches alike so the dashboard can show why a capture did not contribute.

    Area is gated in NATIVE pixels, not analysis pixels.  The thresholds come
    from the measured distribution of real captures (median 943 px, range
    49-7484), which is a native-resolution fact about the optics; expressing the
    gate in analysis pixels would silently rescale it by `analysis_upsample`
    squared -- a factor of nine at the default -- and let through exactly the
    degenerate captures it exists to catch.
    """
    sel = mask > 0
    n_analysis = int(sel.sum())
    n = int(round(n_analysis / max(scale * scale, 1e-9)))
    m = {"mask_area_px": n, "mask_area_analysis_px": n_analysis,
         "equiv_radius_px": float(radius),
         "focus_lapvar": 0.0, "saturated_frac": 0.0, "mean_luma": 0.0}
    if n_analysis == 0:
        return m, False, "empty mask"

    vals = raw_green[sel].astype(np.float32)
    m["mean_luma"] = float(vals.mean())
    m["saturated_frac"] = float((vals >= cfg.specular_thresh).mean())
    lap = cv2.Laplacian(raw_green, cv2.CV_32F, ksize=3)
    m["focus_lapvar"] = float(lap[sel].var())

    if n < cfg.min_mask_area_px:
        return m, False, f"mask too small ({n}px < {cfg.min_mask_area_px})"
    if n > cfg.max_mask_area_px:
        return m, False, f"mask implausibly large ({n}px)"
    if m["focus_lapvar"] < cfg.min_focus_lapvar:
        return m, False, f"out of focus (lapvar {m['focus_lapvar']:.2f})"
    if m["saturated_frac"] > cfg.max_saturated_frac:
        return m, False, f"saturated ({m['saturated_frac']:.0%})"
    if not (cfg.min_mean_luma <= m["mean_luma"] <= cfg.max_mean_luma):
        return m, False, f"exposure out of range (mean {m['mean_luma']:.1f})"
    return m, True, ""


def prepare(index, bgr, mask, cfg, path=""):
    """Crop, rescale, flatten and score one capture.  Returns a `Patch`.

    A Patch is always returned, even when the capture is unusable: `accepted`
    and `reason` carry the verdict.  Pixel data is analysis-scale and small
    (a few hundred px square), so a whole session's Patches fit comfortably in
    memory -- the full-resolution frames are never held.
    """
    h, w = bgr.shape[:2]
    if mask is None:
        g = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = (g > 0).astype(np.uint8) * 255

    p = Patch(index=index, path=path, frame_shape=(h, w))
    bb = content_bbox(mask)
    if bb is None:
        p.accepted, p.reason = False, "empty mask"
        p.quality = {"mask_area_px": 0}
        return p

    s = analysis_scale(w, cfg)
    pad = int(round(cfg.analysis_pad_px / max(s, 1e-6)))
    x0 = max(0, bb[0] - pad)
    y0 = max(0, bb[1] - pad)
    x1 = min(w, bb[0] + bb[2] + pad)
    y1 = min(h, bb[1] + bb[3] + pad)
    p.bbox = (x0, y0, x1 - x0, y1 - y0)

    crop = bgr[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    nw = max(1, int(round(crop.shape[1] * s)))
    nh = max(1, int(round(crop.shape[0] * s)))
    # Backstop only; at real patch sizes this never binds.
    lim = int(cfg.analysis_max_px)
    if max(nw, nh) > lim:
        k = lim / float(max(nw, nh))
        nw, nh, s = max(1, int(nw * k)), max(1, int(nh * k)), s * k

    up = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    umask = cv2.resize(cmask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    umask = (umask > 127).astype(np.uint8) * 255

    # Green channel only: it carries the vessel contrast on retinal imagery and
    # is the least affected by the red-dominated flash return.
    green = up if up.ndim == 2 else up[:, :, 1]

    umask[specular_mask(up, cfg) > 0] = 0
    p.mask_full = umask
    p.radius = equivalent_radius(umask)

    q, ok, why = quality_metrics(green, umask, p.radius, cfg, s)
    p.quality, p.accepted, p.reason = q, ok, why

    # Erosion keeps the detector off the illumination boundary -- but only just.
    # See config.erode_* for the measurement that fixes the ceiling.
    e = int(round(cfg.erode_frac_of_radius * p.radius))
    e = int(np.clip(e, cfg.erode_min_px, cfg.erode_max_px))
    if e > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2)
        p.mask = cv2.erode(umask, k)
    else:
        p.mask = umask.copy()

    # Native crop for compositing.  The specular exclusion found in analysis
    # space is carried back down so the blend never samples a specular either.
    p.native_bgr = (crop if crop.ndim == 3
                    else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)).copy()
    nm = cv2.resize(umask, (crop.shape[1], crop.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    p.native_mask = (nm > 127).astype(np.uint8) * 255
    # Blend weight falls to zero at the mask boundary, so patches feather into
    # one another instead of showing their own outlines as seams.
    dt = cv2.distanceTransform(p.native_mask, cv2.DIST_L2, 3).astype(np.float32)
    p.native_weight = dt / max(float(dt.max()), 1e-6)

    p.raw = green
    p.image = flatten_illumination(green, umask, p.radius, cfg)
    p.image[p.mask == 0] = 0
    p.area = int(np.count_nonzero(p.mask))
    p.scale = s
    p.S = np.array([[s, 0.0, -s * x0],
                    [0.0, s, -s * y0],
                    [0.0, 0.0, 1.0]], np.float64)

    # Intrinsics for this patch, in ANALYSIS coordinates: the frame's principal
    # point (its centre) pushed through S, and the focal rescaled to match.
    from .models import analysis_focal
    pc = p.S @ np.array([w * 0.5, h * 0.5, 1.0])
    p.centre = (float(pc[0]), float(pc[1]))
    p.focal = analysis_focal(cfg, w, s)

    native_area = int(round(p.area / max(s * s, 1e-9)))
    if p.accepted and native_area < cfg.min_mask_area_px:
        p.accepted = False
        p.reason = f"mask too small after erosion ({native_area}px native)"
    return p


def patch_K(patch):
    """Intrinsic matrix for a prepared patch, in analysis coordinates."""
    from .models import make_K
    return make_K(patch.focal, patch.centre[0], patch.centre[1])
