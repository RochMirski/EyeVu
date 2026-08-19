"""Synthetic session generator: known rotations, real-looking degradations.

Takes one wide retinal image (or generates a procedural fundus), treats it as a
map of the retinal sphere, and samples patches from it under KNOWN rotations.
That gives ground truth for the whole pipeline -- which is what makes it
possible to say "registration is accurate" rather than "registration produced
something", and it is what the dashboard work needs to develop against.

The sampling is the exact inverse of `core.compose`: patch pixel -> ray through
K -> rotate by R_i -> project into the map.  Using the same projection in both
directions means a perfect pipeline recovers the input rotations exactly, so any
error measured against this generator is the pipeline's, not the harness's.

Degradations applied, in the order the real optics apply them:
    irregular illuminated blob   (the real masks are NOT circles -- they are
                                  red-shift selections gated to the pupil)
    radial illumination falloff  (strong, and different per patch)
    exposure variation           (per-patch gain)
    specular highlights          (in different places in each frame)
    sensor noise

Sizes default to the measured reality of `Sessions/`: a ~45x32 px blob inside a
480x640 frame, median mask area ~943 px.

    python -m eyevu_mosaic.synth out_dir --n 20
"""

from __future__ import annotations

import argparse
import io
import json
import os

import cv2
import numpy as np

from .core.models import make_K, so3_exp, so3_log
from .core.compose import Projection


# ══ the retinal map ═══════════════════════════════════════════════════════

def fundus_texture(size=1400, seed=7, contrast=1.0):
    """A procedural fundus: vessel arcades, disc, choroidal background.

    Not a substitute for a real wide image -- pass one with --image when you
    have it -- but it has the property that matters for testing a matcher: a
    strongly non-uniform distribution of texture, with sparse regions that yield
    few keypoints, which is what makes real sessions hard.

    The background is a MULTI-SCALE (roughly 1/f) field rather than a single
    blurred noise layer.  That detail matters more than it looks: a single layer
    blurred enough to look like choroid has almost no variance left at the scale
    a patch actually sees -- measured, 0.7 grey levels against a vignette of 16
    -- so every pair correlates on its illumination instead of its anatomy and
    the whole session is unregistrable for reasons that have nothing to do with
    the pipeline.  Real choroid has structure at every scale; so must this.
    """
    rng = np.random.default_rng(seed)
    tex = np.full((size, size), 105.0, np.float32)
    # Octaves chosen so the finest survives the band-limiting blur applied
    # before sampling (see `generate`), and the coarsest spans several patches.
    for sigma, amp in ((size / 175.0, 11.0), (size / 70.0, 12.0),
                       (size / 28.0, 14.0), (size / 11.0, 16.0)):
        layer = cv2.GaussianBlur(
            rng.normal(0.0, 1.0, (size, size)).astype(np.float32), (0, 0), sigma)
        layer /= max(float(layer.std()), 1e-6)
        tex += (amp * contrast) * layer

    # Optic disc: the one high-contrast landmark, off-centre as in a real eye.
    dc = (int(size * 0.62), int(size * 0.48))
    cv2.circle(tex, dc, int(size * 0.035), 205.0, -1, cv2.LINE_AA)
    cv2.circle(tex, dc, int(size * 0.020), 225.0, -1, cv2.LINE_AA)

    # Vessel arcades: recursive branching out of the disc, thinning as they go.
    def branch(p, ang, width, length, depth):
        if depth <= 0 or width < 0.7:
            return
        n = 14
        pts = [p]
        a = ang
        for _ in range(n):
            a += rng.normal(0, 0.13)
            pts.append((pts[-1][0] + length / n * np.cos(a),
                        pts[-1][1] + length / n * np.sin(a)))
        ip = np.array(pts, np.int32)
        cv2.polylines(tex, [ip], False, 45.0, max(1, int(round(width))),
                      cv2.LINE_AA)
        for s in (-1, 1):
            if rng.random() < 0.82:
                branch(pts[-1], a + s * rng.uniform(0.25, 0.75),
                       width * rng.uniform(0.55, 0.75),
                       length * rng.uniform(0.55, 0.8), depth - 1)

    for k in range(8):
        branch(dc, k * np.pi / 4 + rng.normal(0, 0.2), size * 0.011,
               size * 0.20, 5)

    # Choroidal mottling, which is most of what the avascular periphery offers.
    # Drawn semi-transparently so it modulates the multi-scale field rather than
    # punching flat discs through it.
    over = tex.copy()
    for _ in range(1600):
        c = rng.integers(0, size, 2)
        cv2.circle(over, (int(c[0]), int(c[1])), int(rng.integers(5, 22)),
                   float(rng.uniform(78, 132)), -1, cv2.LINE_AA)
    tex = cv2.addWeighted(tex, 0.62, over, 0.38, 0.0)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)

    # Fundus colour: red-dominant, with the structure carried in green.
    g = np.clip(tex, 0, 255).astype(np.uint8)
    r = np.clip(tex * 1.55 + 40, 0, 255).astype(np.uint8)
    b = np.clip(tex * 0.35, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


# ══ masks and degradations ════════════════════════════════════════════════

def blob_mask(h, w, cx, cy, rx, ry, rng, wobble=0.28):
    """An irregular filled blob -- what the red-shift selection actually returns.

    A clean ellipse would be an easier problem than the real one: the true masks
    have ragged, per-capture boundaries, and since ~77% of keypoints sit near
    the rim, boundary shape genuinely affects what gets matched.
    """
    n = 72
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Smooth periodic noise on the radius (low-order Fourier, so it stays a
    # plausible blob rather than a starburst).
    r = np.ones(n)
    for k in range(1, 5):
        r += wobble / k * np.sin(k * th + rng.uniform(0, 2 * np.pi))
    pts = np.stack([cx + rx * r * np.cos(th), cy + ry * r * np.sin(th)], 1)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def apply_vignette(bgr, cx, cy, radius, strength, rng):
    """Radial illumination falloff about a per-capture illumination centre."""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy) / max(radius, 1e-6)
    fall = np.clip(1.0 - strength * d ** 2, 0.05, 1.0)
    return np.clip(bgr.astype(np.float32) * fall[..., None], 0, 255)


def add_speculars(bgr, mask, rng, n=(0, 3), radius=(2, 5)):
    """Corneal/lens reflections, in a different place in every frame.

    Returned separately as well as burned in, so a test can check that the
    pipeline excluded them rather than merely that it survived them.
    """
    h, w = bgr.shape[:2]
    spec = np.zeros((h, w), np.uint8)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return bgr, spec
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        k = int(rng.integers(0, len(xs)))
        r = int(rng.integers(radius[0], radius[1] + 1))
        cv2.circle(spec, (int(xs[k]), int(ys[k])), r, 255, -1, cv2.LINE_AA)
    spec = cv2.GaussianBlur(spec, (0, 0), 1.0)
    out = np.maximum(bgr, spec[..., None].astype(np.float32) * (255.0 / 255.0))
    return out, (spec > 100).astype(np.uint8) * 255


# ══ the generator ═════════════════════════════════════════════════════════

def generate(out_dir, n=20, frame=(640, 480), patch_px=46, focal_ref=1200.0,
             step_deg=0.9, jitter_deg=0.25, roll_deg=3.0, seed=1,
             image=None, map_focal_scale=5.0, noise=2.5, write=True,
             texture_seed=None, centre_deg=(0.0, 0.0)):
    """Generate a synthetic session with ground-truth rotations.

    `patch_px` is the blob's mean diameter in native pixels (measured median is
    45x32).  `step_deg` sets the gaze spacing and therefore the overlap: at the
    default focal a 46 px patch subtends ~2.2 deg, so 0.9 deg leaves roughly
    60% linear overlap between neighbours.

    TWO SITTINGS OF THE SAME EYE are made by sharing `texture_seed` while
    varying `seed`: the retina is then identical and only the gaze positions,
    blobs and exposures differ.  Without that separation two "sessions" get two
    different retinas and can never link, which is not the thing being tested.
    `centre_deg` offsets a sitting's gaze range so it covers a different -- but
    overlapping -- part of the fundus, as a second sitting realistically would.

    Returns a dict with the ground truth, and (if `write`) writes a directory in
    exactly the layout `run_session.load_session` reads.
    """
    rng = np.random.default_rng(seed)
    fh, fw = int(frame[0]), int(frame[1])
    tseed = seed if texture_seed is None else texture_seed
    big = (cv2.imread(image, cv2.IMREAD_COLOR) if image
           else fundus_texture(seed=tseed))
    if big is None:
        raise FileNotFoundError(image)

    f_native = focal_ref * fw / 480.0
    K = make_K(f_native, fw * 0.5, fh * 0.5)
    f_map = f_native * map_focal_scale

    # BAND-LIMIT THE MAP BEFORE SAMPLING.  The map carries `map_focal_scale`
    # times the frame's angular resolution, so remapping into a frame decimates
    # by that factor -- and bilinear interpolation does not prefilter, it just
    # picks the two nearest samples.  Without this blur each patch gets a
    # DIFFERENTLY ALIASED version of the same retina, and two genuinely
    # overlapping patches then fail to correlate: the generator, not the
    # pipeline, becomes the limit on achievable registration.  A real camera is
    # band-limited by its own sampling, so this is also the more faithful model.
    big = cv2.GaussianBlur(big, (0, 0), 0.5 * float(map_focal_scale))
    mh, mw = big.shape[:2]
    proj = Projection(np.array([0.0, 0.0, 1.0]), f_map,
                      "azimuthal_equidistant",
                      origin=np.array([-mw * 0.5, -mh * 0.5]))

    # Gaze positions: a square-ish spiral, so consecutive captures overlap and
    # the graph also has non-consecutive overlaps to find.
    rots, k, ring = [], 0, 0
    while len(rots) < n:
        ring += 1
        for gx in range(-ring, ring + 1):
            for gy in range(-ring, ring + 1):
                if max(abs(gx), abs(gy)) != ring - 1 and ring > 1:
                    continue
                if len(rots) >= n:
                    break
                rx = np.radians(gy * step_deg + centre_deg[1]
                                + rng.normal(0, jitter_deg))
                ry = np.radians(gx * step_deg + centre_deg[0]
                                + rng.normal(0, jitter_deg))
                rz = np.radians(rng.normal(0, roll_deg))
                rots.append(so3_exp([rx, ry, rz]))
                k += 1
    rots = rots[:n]

    gy_, gx_ = np.mgrid[0:fh, 0:fw].astype(np.float64)
    ones = np.ones_like(gx_)
    Ki = np.linalg.inv(K)
    dirs0 = np.stack([gx_.ravel(), gy_.ravel(), ones.ravel()], 1) @ Ki.T
    dirs0 /= np.maximum(np.linalg.norm(dirs0, axis=1, keepdims=True), 1e-12)

    truth = {"n": n, "frame": [fh, fw], "focal_ref_px": focal_ref,
             "focal_native_px": f_native, "map_focal_px_per_rad": f_map,
             "patches": []}
    if write:
        os.makedirs(out_dir, exist_ok=True)

    for i, R in enumerate(rots):
        # Patch pixel -> ray -> rotate into map frame -> project into the map.
        v = dirs0 @ np.asarray(R, np.float64).T
        xy = proj.from_dir(v)
        mx = xy[:, 0].astype(np.float32).reshape(fh, fw)
        my = xy[:, 1].astype(np.float32).reshape(fh, fw)
        full = cv2.remap(big, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        rx = patch_px * 0.5 * rng.uniform(0.85, 1.15)
        ry = rx * rng.uniform(0.62, 0.85)          # measured bboxes are ~45x32
        cx = fw * 0.5 + rng.normal(0, 6)
        cy = fh * 0.5 + rng.normal(0, 6)
        mask = blob_mask(fh, fw, cx, cy, rx, ry, rng)

        img = apply_vignette(full, cx + rng.normal(0, 3), cy + rng.normal(0, 3),
                             rx * rng.uniform(1.1, 1.7),
                             rng.uniform(0.35, 0.75), rng)
        img *= rng.uniform(0.65, 1.35)                          # exposure
        img, spec = add_speculars(img, mask, rng)
        img += rng.normal(0, noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # The real capture path excludes speculars from the mask before writing
        # the extract, so the synthetic session must too or it is not the same
        # problem (see cap.redeye_extract).
        mask[cv2.dilate(spec, np.ones((5, 5), np.uint8)) > 0] = 0

        extract = np.zeros_like(img)
        extract[mask > 0] = img[mask > 0]

        rec = {"index": i, "rotvec": so3_log(R).tolist(),
               "rotation": np.asarray(R).tolist(),
               "mask_area_px": int((mask > 0).sum()),
               "centre": [float(cx), float(cy)]}
        truth["patches"].append(rec)

        if write:
            base = os.path.join(out_dir, f"redeye_{i + 1:02d}_{i:06d}")
            cv2.imwrite(f"{base}_extract.png", extract)
            cv2.imwrite(f"{base}_mask.png", mask)

    if write:
        with io.open(os.path.join(out_dir, "ground_truth.json"), "w",
                     encoding="utf-8") as fh_:
            json.dump(truth, fh_, indent=2)
    return truth


# ══ scoring a run against the truth ═══════════════════════════════════════

def compare(truth, rotations, focal=None, radius=70.0):
    """Error of recovered rotations vs ground truth, gauge-removed.

    Absolute rotations are only defined up to one global rotation (the pipeline
    fixes an arbitrary patch as the reference), so the comparison aligns the two
    sets by their best common rotation first and reports the residual.

    The residual is reported THREE ways, because the total angle on its own is
    actively misleading here.  A rotation's components affect the image by very
    different amounts at this field of view: pointing error moves it by
    f * angle (f ~ 3600 analysis px) while roll moves it only by r * angle
    (r ~ 70 px).  Roll is both the worst-observed parameter and the least
    consequential, so a total-angle figure is dominated by the one component
    nobody should care about.  `displacement_px` is the number that says how
    well the mosaic actually lines up.
    """
    from .core.models import rotation_angle_deg, so3_log
    gt = {p["index"]: np.asarray(p["rotation"], np.float64)
          for p in truth["patches"]}
    f = float(focal if focal else truth.get("focal_native_px", 1200.0) * 3.0)
    common = sorted(set(gt) & set(rotations))
    empty = {"n": len(common), "median_deg": None, "max_deg": None,
             "per_patch": {}}
    if len(common) < 2:
        return empty

    # Best common alignment, estimated ROBUSTLY.
    #
    # The gauge is a nuisance parameter: only relative rotations are observable,
    # so the two sets must be brought into a common frame before anything can be
    # compared.  Doing that with a plain average is a trap -- it is a mean, so a
    # couple of badly placed patches drag the alignment, and then EVERY patch
    # reports the resulting offset as its own error.  Measured, that turned a
    # session whose edges were all accurate to 0.5 native px into an apparent
    # 10 px global error, which is a property of the metric and not of the
    # pipeline.  Two reweighting passes discarding the worst quarter fix it.
    W = {n: 1.0 for n in common}
    A = np.eye(3)
    for _ in range(3):
        Msum = np.zeros((3, 3))
        for n in common:
            Msum += W[n] * (gt[n] @ np.asarray(rotations[n], np.float64).T)
        U, _S, Vt = np.linalg.svd(Msum)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
        A = U @ D @ Vt
        errs = {n: rotation_angle_deg(gt[n].T @ (A @ np.asarray(rotations[n],
                                                                np.float64)))
                for n in common}
        cut = float(np.percentile(list(errs.values()), 75))
        W = {n: (1.0 if errs[n] <= max(cut, 1e-9) else 0.0) for n in common}
        if not any(W.values()):
            W = {n: 1.0 for n in common}
            break

    per, point, roll, disp = {}, [], [], []
    for n in common:
        E = gt[n].T @ (A @ np.asarray(rotations[n], np.float64))
        per[n] = float(rotation_angle_deg(E))
        w = so3_log(E)
        p = float(np.linalg.norm(w[:2]))
        r = float(abs(w[2]))
        point.append(np.degrees(p))
        roll.append(np.degrees(r))
        disp.append(float(np.hypot(f * p, radius * r)))
    vals = np.array(list(per.values()))
    return {"n": len(common),
            "median_deg": float(np.median(vals)),
            "mean_deg": float(vals.mean()),
            "max_deg": float(vals.max()),
            "median_pointing_deg": float(np.median(point)),
            "median_roll_deg": float(np.median(roll)),
            "median_displacement_px": float(np.median(disp)),
            "max_displacement_px": float(np.max(disp)),
            "per_patch": {str(k): round(v, 4) for k, v in per.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image", default=None,
                    help="wide retinal image to sample from (else procedural)")
    ap.add_argument("--step-deg", type=float, default=0.9)
    ap.add_argument("--patch-px", type=int, default=46)
    a = ap.parse_args(argv)
    t = generate(a.out_dir, n=a.n, seed=a.seed, image=a.image,
                 step_deg=a.step_deg, patch_px=a.patch_px)
    areas = [p["mask_area_px"] for p in t["patches"]]
    print(f"wrote {t['n']} patches to {a.out_dir}  "
          f"(mask area min {min(areas)}, median {int(np.median(areas))}, "
          f"max {max(areas)} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
