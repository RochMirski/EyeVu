"""Projection, tiled warping, exposure normalisation and median blending.

The output is an AZIMUTHAL EQUIDISTANT projection about the mean gaze direction.
That keeps pixel spacing proportional to arc length on the retinal sphere --
which is the clinically meaningful quantity, since it is what makes a lesion's
measured size mean the same thing at the posterior pole and out at the equator.
It also degrades far more gracefully than gnomonic at wide field, where the
tangent term runs away.

Blending is a per-pixel WEIGHTED MEDIAN over contributing patches, weighted by
distance from each patch's own mask boundary.  The median matters because a
single bad contribution -- a residual specular, a misregistered patch, a badly
exposed capture -- is very unlikely to be the weighted middle of the stack,
whereas a mean would let it through in proportion to its weight.

Memory
------
Compositing samples the NATIVE-resolution crop of each patch, held on the Patch
itself (at most ~125x101x3 = 38 kB each, so a whole session is well under 2 MB).
That means one resampling from source pixels straight to output pixels, rather
than source -> analysis -> output, and no disk IO in the compositing pass at
all.  Output is still built in tiles so peak working memory stays flat as the
canvas grows.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import patch_K


class Projection:
    """Sphere <-> output plane, azimuthal equidistant or gnomonic.

    `f_out` is output pixels per radian of visual angle.  `origin` is the plane
    coordinate of output pixel (0, 0), so the canvas can be placed to bound the
    data without disturbing the projection itself.
    """

    def __init__(self, centre, f_out, mode="azimuthal_equidistant",
                 origin=(0.0, 0.0)):
        c = np.asarray(centre, np.float64)
        self.c = c / max(float(np.linalg.norm(c)), 1e-12)
        # Any basis orthogonal to the centre will do; pick the stabler seed.
        seed = np.array([0.0, 1.0, 0.0]) if abs(self.c[0]) > 0.9 else \
            np.array([1.0, 0.0, 0.0])
        e1 = seed - self.c * (seed @ self.c)
        self.e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
        self.e2 = np.cross(self.c, self.e1)
        self.f = float(f_out)
        self.mode = mode
        self.origin = np.asarray(origin, np.float64)

    def from_dir(self, d):
        """Unit directions (N, 3) -> output pixel coordinates (N, 2)."""
        d = np.asarray(d, np.float64).reshape(-1, 3)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        ct = np.clip(d @ self.c, -1.0, 1.0)
        th = np.arccos(ct)
        r = self.f * (np.tan(np.minimum(th, 1.4)) if self.mode == "gnomonic" else th)
        a1, a2 = d @ self.e1, d @ self.e2
        n = np.maximum(np.hypot(a1, a2), 1e-12)
        return np.stack([r * a1 / n, r * a2 / n], axis=1) - self.origin

    def to_dir(self, xy):
        """Output pixel coordinates (N, 2) -> unit directions (N, 3)."""
        p = np.asarray(xy, np.float64).reshape(-1, 2) + self.origin
        r = np.hypot(p[:, 0], p[:, 1])
        th = (np.arctan(r / self.f) if self.mode == "gnomonic" else r / self.f)
        u = np.where(r > 1e-12, p[:, 0] / np.maximum(r, 1e-12), 1.0)
        v = np.where(r > 1e-12, p[:, 1] / np.maximum(r, 1e-12), 0.0)
        st, ct = np.sin(th), np.cos(th)
        return (ct[:, None] * self.c[None, :]
                + (st * u)[:, None] * self.e1[None, :]
                + (st * v)[:, None] * self.e2[None, :])


def mean_direction(G):
    """Mean gaze direction over placed patches, as a unit vector.

    Each patch looks along its own optical axis; rotated into the reference
    frame and averaged, that is where the session as a whole was pointed, and it
    is the natural centre for the projection.
    """
    z = np.array([0.0, 0.0, 1.0])
    d = np.zeros(3)
    for R in G.values():
        d += np.asarray(R, np.float64) @ z
    n = float(np.linalg.norm(d))
    return (d / n) if n > 1e-9 else z


def _boundary_points(patch, n=12):
    """Points around the patch's mask bbox, in analysis coordinates."""
    ys, xs = np.nonzero(patch.mask_full)
    if not len(xs):
        h, w = patch.mask_full.shape[:2]
        x0, y0, x1, y1 = 0, 0, w - 1, h - 1
    else:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    t = np.linspace(0.0, 1.0, n)
    top = np.stack([x0 + t * (x1 - x0), np.full(n, y0)], 1)
    bot = np.stack([x0 + t * (x1 - x0), np.full(n, y1)], 1)
    lef = np.stack([np.full(n, x0), y0 + t * (y1 - y0)], 1)
    rig = np.stack([np.full(n, x1), y0 + t * (y1 - y0)], 1)
    return np.vstack([top, bot, lef, rig])


def _patch_dirs(patch, pts):
    """Analysis-space points -> reference-frame-ready unit rays in patch frame."""
    from .models import rays
    return rays(pts, patch_K(patch))


def plan_canvas(patches, G, cfg):
    """Choose the projection and canvas size that bound every placed patch.

    Returns (projection, width, height).
    """
    by_index = {p.index: p for p in patches}
    centre = mean_direction(G)
    f_out = float(cfg.output_scale) * float(
        np.median([by_index[n].focal for n in G]))
    proj = Projection(centre, f_out, cfg.projection)

    pts = []
    for n, R in G.items():
        p = by_index[n]
        v = _patch_dirs(p, _boundary_points(p)) @ np.asarray(R, np.float64).T
        pts.append(proj.from_dir(v))
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = float(cfg.output_margin_frac) * float(span.max())
    lo -= pad
    hi += pad

    w = int(np.ceil(hi[0] - lo[0]))
    h = int(np.ceil(hi[1] - lo[1]))
    lim = int(cfg.output_max_px)
    if max(w, h) > lim:                       # keep a runaway solution bounded
        k = lim / float(max(w, h))
        proj = Projection(centre, f_out * k, cfg.projection)
        lo, hi = lo * k, hi * k
        w, h = max(1, int(w * k)), max(1, int(h * k))
    proj.origin = lo
    return proj, max(1, w), max(1, h)


def exposure_gains(patches, G, cfg):
    """Per-patch gain bringing every patch to a common mean level.

    Variable exposure between captures is one of the two things (with radial
    falloff) that makes the same retina look different in two patches.  Feature
    matching is immune to it because it works on flattened images; the composite
    is not, and an unnormalised mosaic shows every patch boundary as a step.
    """
    if not cfg.exposure_normalise:
        return {n: 1.0 for n in G}
    means = {}
    for n in G:
        p = _by(patches, n)
        m = p.native_mask > 0
        means[n] = float(p.native_bgr[m].mean()) if m.any() else 0.0
    ref = float(np.median([v for v in means.values() if v > 0] or [1.0]))
    return {n: (ref / means[n] if means[n] > 1e-3 else 1.0) for n in G}


def _by(patches, index):
    for p in patches:
        if p.index == index:
            return p
    raise KeyError(index)


def _patch_output_bbox(patch, R, proj, w, h):
    """Where this patch lands on the canvas, as an integer bbox (or None)."""
    v = _patch_dirs(patch, _boundary_points(patch, 24)) @ np.asarray(R, np.float64).T
    xy = proj.from_dir(v)
    lo = np.floor(xy.min(axis=0)).astype(int) - 1
    hi = np.ceil(xy.max(axis=0)).astype(int) + 1
    x0, y0 = max(0, lo[0]), max(0, lo[1])
    x1, y1 = min(w, hi[0]), min(h, hi[1])
    return None if (x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)


def _sample(patch, R, proj, gain, x0, y0, tw, th):
    """Sample one patch over a tile.  Returns (bgr float32, weight float32)."""
    gx, gy = np.meshgrid(np.arange(tw) + x0, np.arange(th) + y0)
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    d = proj.to_dir(xy)
    # Reference frame -> this patch's frame:  v_i = G_i^T d.
    v = d @ np.asarray(R, np.float64)
    z = v[:, 2]
    ok = z > 1e-6
    z = np.where(ok, z, 1.0)
    ax = patch.centre[0] + patch.focal * v[:, 0] / z
    ay = patch.centre[1] + patch.focal * v[:, 1] / z
    # Analysis coords -> native crop coords is a pure scale (S is crop + scale,
    # and the crop offset is already baked into the stored native crop).
    mx = (ax / patch.scale).astype(np.float32).reshape(th, tw)
    my = (ay / patch.scale).astype(np.float32).reshape(th, tw)
    okm = ok.reshape(th, tw)

    bgr = cv2.remap(patch.native_bgr, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    wgt = cv2.remap(patch.native_weight, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wgt = np.where(okm, wgt, 0.0).astype(np.float32)
    return bgr.astype(np.float32) * float(gain), wgt


def _weighted_median(vals, wts):
    """Per-pixel weighted median over the stack axis.

    `vals` is (K, H, W, 3), `wts` is (K, H, W).  Returns (H, W, 3).
    """
    K = vals.shape[0]
    out = np.zeros(vals.shape[1:], np.float32)
    for ch in range(vals.shape[3]):
        v = vals[..., ch]
        order = np.argsort(v, axis=0)
        vs = np.take_along_axis(v, order, axis=0)
        ws = np.take_along_axis(wts, order, axis=0)
        cw = np.cumsum(ws, axis=0)
        half = cw[-1] * 0.5
        idx = (cw < half).sum(axis=0)
        idx = np.clip(idx, 0, K - 1)
        out[..., ch] = np.take_along_axis(vs, idx[None], axis=0)[0]
    return out


def composite(patches, G, cfg, progress=None):
    """Warp and blend every placed patch into one mosaic.

    Returns (bgr uint8, coverage uint16, projection, info).
    """
    proj, W, H = plan_canvas(patches, G, cfg)
    gains = exposure_gains(patches, G, cfg)
    boxes = {}
    for n, R in G.items():
        bb = _patch_output_bbox(_by(patches, n), R, proj, W, H)
        if bb is not None:
            boxes[n] = bb

    out = np.zeros((H, W, 3), np.uint8)
    cov = np.zeros((H, W), np.uint16)
    tile = max(64, int(cfg.tile_px))
    n_tiles = 0
    peak_stack = 0

    for ty in range(0, H, tile):
        th = min(tile, H - ty)
        for tx in range(0, W, tile):
            tw = min(tile, W - tx)
            here = [n for n, (x0, y0, x1, y1) in boxes.items()
                    if not (x1 <= tx or x0 >= tx + tw
                            or y1 <= ty or y0 >= ty + th)]
            n_tiles += 1
            if not here:
                continue
            vals, wts = [], []
            for n in here:
                b, w = _sample(_by(patches, n), G[n], proj, gains[n],
                               tx, ty, tw, th)
                if w.max() <= 0:
                    continue
                vals.append(b)
                wts.append(w)
            if not vals:
                continue
            peak_stack = max(peak_stack, len(vals))
            V = np.stack(vals)                      # (K, th, tw, 3)
            Wt = np.stack(wts)                      # (K, th, tw)
            any_w = Wt.sum(axis=0)
            if cfg.blend_mode == "weighted_mean":
                px = (V * Wt[..., None]).sum(0) / np.maximum(any_w, 1e-6)[..., None]
            else:
                px = _weighted_median(V, Wt)
            sel = any_w > 1e-6
            tile_out = np.zeros((th, tw, 3), np.float32)
            tile_out[sel] = px[sel]
            out[ty:ty + th, tx:tx + tw] = np.clip(tile_out, 0, 255).astype(np.uint8)
            cov[ty:ty + th, tx:tx + tw] = (Wt > 1e-6).sum(axis=0).astype(np.uint16)

    info = {"width": W, "height": H, "n_tiles": n_tiles,
            "tile_px": tile, "max_stack_depth": int(peak_stack),
            "projection": cfg.projection,
            "focal_out_px_per_rad": proj.f,
            "centre_direction": proj.c.tolist(),
            "blend_mode": cfg.blend_mode,
            "exposure_gains": {str(k): round(v, 4) for k, v in gains.items()},
            "covered_px": int((cov > 0).sum())}
    if progress:
        progress(info)
    return out, cov, proj, info


def patch_palette(n):
    """`n` visually distinct BGR colours, evenly spaced around the hue wheel."""
    hues = (np.arange(max(n, 1)) * (180.0 / max(n, 1))).astype(np.uint8)
    hsv = np.stack([hues, np.full(max(n, 1), 230, np.uint8),
                    np.full(max(n, 1), 255, np.uint8)], axis=1).reshape(-1, 1, 3)
    return [tuple(int(c) for c in bgr)
            for bgr in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)]


def patch_outlines(patches, G, proj, shape, simplify=2.0):
    """Where each patch's own boundary lands on the canvas.

    Returns {patch_index: (N, 2) int32 polygon in output pixels}.  Pure geometry
    -- no drawing -- so the dashboard can hit-test, colour or animate these
    however it likes rather than being handed a flattened picture.

    The boundary is taken from the UNERODED mask, because the question this
    answers is "which patch contributed these pixels", and the eroded mask is a
    feature-detection concern that would understate the real footprint.
    """
    h, w = shape[:2]
    out = {}
    for n, R in sorted(G.items()):
        p = _by(patches, n)
        if p.mask_full is None:
            continue
        cnts, _ = cv2.findContours(p.mask_full, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if simplify > 0:
            cnt = cv2.approxPolyDP(cnt, float(simplify), True)
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            continue
        v = _patch_dirs(p, pts) @ np.asarray(R, np.float64).T
        xy = proj.from_dir(v)
        if not np.all(np.isfinite(xy)):
            continue
        # Clamp rather than clip: a patch may extend past the canvas edge, and a
        # clipped polygon would misreport where its boundary actually is.
        xy[:, 0] = np.clip(xy[:, 0], -1e4, w + 1e4)
        xy[:, 1] = np.clip(xy[:, 1], -1e4, h + 1e4)
        out[n] = np.round(xy).astype(np.int32)
    return out


def draw_outlines(bgr, outlines, thickness=1, label=True, dim=0.0,
                  colours=None):
    """Draw patch boundaries onto a copy of a mosaic.

    `dim` darkens the underlying mosaic so thin outlines stay readable over
    bright fundus; 0 leaves it alone.  Each patch gets its own colour and, if
    `label`, its index at the boundary's centroid.
    """
    out = bgr.copy()
    if dim > 0:
        out = (out.astype(np.float32) * (1.0 - float(dim))).astype(np.uint8)
    keys = sorted(outlines)
    cols = colours or patch_palette(len(keys))
    for i, n in enumerate(keys):
        poly = outlines[n]
        col = cols[i % len(cols)]
        cv2.polylines(out, [poly], True, col, int(thickness), cv2.LINE_AA)
        if label:
            c = poly.mean(axis=0)
            org = (int(c[0]) - 6, int(c[1]) + 4)
            # Outline the text too, or an index vanishes against the retina.
            for colour, th in (((0, 0, 0), 3), (col, 1)):
                cv2.putText(out, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            colour, th, cv2.LINE_AA)
    return out


def coverage_image(cov):
    """Colour-mapped contribution count, for the dashboard."""
    if cov.max() == 0:
        return np.zeros(cov.shape + (3,), np.uint8)
    norm = (cov.astype(np.float32) / float(cov.max()) * 255).astype(np.uint8)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    img[cov == 0] = 0
    return img


def preview(bgr, cfg):
    """Downscaled copy for fast dashboard loading."""
    h, w = bgr.shape[:2]
    m = int(cfg.preview_max_px)
    if max(h, w) <= m:
        return bgr.copy()
    k = m / float(max(h, w))
    return cv2.resize(bgr, (max(1, int(w * k)), max(1, int(h * k))),
                      interpolation=cv2.INTER_AREA)
