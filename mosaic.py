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
* Homographies chain onto a reference frame, and each new image is matched
  against the growing mosaic rather than only its predecessor, so a capture that
  overlaps an earlier one still lands correctly.
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
MIN_MATCHES = 8           # fewest good matches to trust a homography
LOWE_RATIO = 0.78         # Lowe ratio test for descriptor matching
RANSAC_REPROJ = 4.0       # px reprojection tolerance for findHomography
MIN_INLIER_FRAC = 0.30    # inliers/matches below this = a bogus alignment
CANVAS_MARGIN = 1.4       # grow the canvas by this multiple of the base patch
MAX_DIM = 2600            # cap the canvas so a runaway homography cannot explode it
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
ANALYSIS_ERODE = 5        # mask erosion IN ANALYSIS SPACE (~1px of original)
MOSAIC_UPSCALE = 3.0      # render the mosaic this much larger than the source, so
                          # the stitched fundus is not limited to ~50px of detail


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


def _match(det, norm, a_img, a_mask, a_S, b_img, b_mask, b_S):
    """Homography mapping image B's ORIGINAL coords onto image A's.

    Matching happens in analysis space; the result is composed back through the
    two crop/scale transforms:  H = inv(S_A) . H_analysis . S_B.

    Returns (H, n_inliers, n_matches)."""
    if a_img is None or b_img is None:
        return None, 0, 0
    ka, da = det.detectAndCompute(a_img, a_mask)
    kb, db = det.detectAndCompute(b_img, b_mask)
    if da is None or db is None or len(ka) < MIN_MATCHES or len(kb) < MIN_MATCHES:
        return None, 0, 0
    matcher = cv2.BFMatcher(norm)
    try:
        knn = matcher.knnMatch(db, da, k=2)
    except cv2.error:
        return None, 0, 0
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0, len(good)
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_an, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    if H_an is None or inl is None:
        return None, 0, len(good)
    n_in = int(inl.sum())
    if n_in < MIN_MATCHES or n_in / len(good) < MIN_INLIER_FRAC:
        return None, n_in, len(good)
    try:
        H = np.linalg.inv(a_S) @ H_an @ b_S
    except np.linalg.LinAlgError:
        return None, n_in, len(good)
    return H, n_in, len(good)


def stitch(images, masks=None, verbose=True):
    """Register and blend red-eye extracts into one mosaic.

    `images` are BGR patches on black (cap.RedeyeResult.extract); `masks` are the
    matching selections, derived from non-black pixels when omitted.  Images that
    cannot be registered are skipped, not forced in.

    Returns (mosaic_bgr, info) where info has: used, skipped, detector, placements.
    """
    info = {"used": 0, "skipped": [], "detector": "", "placements": []}
    imgs = [i for i in images if i is not None]
    if len(imgs) < 2:
        return (imgs[0].copy() if imgs else None), info
    if masks is None:
        masks = [None] * len(imgs)

    det, norm, name = _detector()
    info["detector"] = name
    if verbose:
        print(f"  mosaic: {len(imgs)} image(s), detector = {name}")

    def _mask_of(i):
        if masks[i] is not None:
            return masks[i]
        g = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        return (g > 0).astype(np.uint8) * 255

    # Size the canvas from the CONTENT, not the source frame: the extracts are a
    # small patch adrift in a mostly-black frame, so a frame-sized canvas would be
    # almost entirely empty and the mosaic would be rendered at patch resolution.
    bb0 = content_bbox(_mask_of(0))
    if bb0 is None:
        return imgs[0].copy(), info
    x0, y0, bw0, bh0 = bb0
    u = MOSAIC_UPSCALE
    cw = int(min(MAX_DIM, bw0 * u * (1 + 2 * CANVAS_MARGIN)))
    ch = int(min(MAX_DIM, bh0 * u * (1 + 2 * CANVAS_MARGIN)))
    # Base transform: scale image 0 by `u` and drop its content in the middle.
    T = np.array([[u, 0.0, cw * 0.5 - (x0 + bw0 * 0.5) * u],
                  [0.0, u, ch * 0.5 - (y0 + bh0 * 0.5) * u],
                  [0.0, 0.0, 1.0]], np.float64)

    acc = np.zeros((ch, cw, 3), np.float32)     # running sum for the blend
    cnt = np.zeros((ch, cw), np.float32)        # contributions per pixel

    def _place(img, mask, H):
        warp = cv2.warpPerspective(img, H, (cw, ch))
        wm = cv2.warpPerspective(mask, H, (cw, ch), flags=cv2.INTER_NEAREST)
        sel = wm > 0
        acc[sel] += warp[sel].astype(np.float32)
        cnt[sel] += 1.0
        return int(sel.sum())

    def _mosaic():
        out = np.zeros((ch, cw, 3), np.uint8)
        nz = cnt > 0
        out[nz] = (acc[nz] / cnt[nz][:, None]).astype(np.uint8)
        return out

    n = _place(imgs[0], _mask_of(0), T)
    info["used"] = 1
    info["placements"].append({"index": 0, "px": n, "inliers": None})
    if verbose:
        print(f"    [0] reference, {n}px on a {cw}x{ch} canvas")

    for i in range(1, len(imgs)):
        # Match against the mosaic so far, not just the previous frame, so an
        # image overlapping an EARLIER capture still registers.
        cur = _mosaic()
        cur_mask = (cnt > 0).astype(np.uint8) * 255
        a_img, a_mask, a_S = _analysis(cur, cur_mask)
        b_img, b_mask, b_S = _analysis(imgs[i], _mask_of(i))
        H, n_in, n_match = _match(det, norm, a_img, a_mask, a_S,
                                  b_img, b_mask, b_S)
        if H is None:
            info["skipped"].append(
                {"index": i, "reason": f"no reliable homography "
                                       f"({n_in}/{n_match} inliers)"})
            if verbose:
                print(f"    [{i}] SKIPPED - {n_in}/{n_match} inliers")
            continue
        n = _place(imgs[i], _mask_of(i), H)
        info["used"] += 1
        info["placements"].append({"index": i, "px": n, "inliers": n_in})
        if verbose:
            print(f"    [{i}] placed, {n}px, {n_in}/{n_match} inliers")

    return _mosaic(), info


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
              " [--browse]")
        return 2
    session = argv[1]
    if "--browse" in argv:
        return 0 if browse(session) else 1
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
