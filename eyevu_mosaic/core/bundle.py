"""Session bundle read/write -- the contract with the diagnostic dashboard.

    sessions/<session_id>/
        raw/                  original patches, byte-for-byte unmodified
        meta.json             capture metadata, intrinsics, gaze, config used
        bundle.npz            all array data
        bundle.json           all non-array data
        mosaic.png            full-resolution output
        mosaic_preview.jpg    downscaled, for fast dashboard loading
        coverage.png          per-pixel contribution count
        log.txt

The binding requirement is that the bundle is SUFFICIENT TO RE-RUN matching and
global optimisation offline, without the Pi and without re-reading the raw
images.  That is why the npz carries the analysis images and masks and the
native crops, not just keypoints and descriptors: the acceptance gate here is
photometric (`pairwise.ncc_verify` correlates actual pixels), so a bundle with
descriptors alone could re-fit models but could not re-decide which pairs to
believe -- which is the part worth re-running.

It costs little.  Analysis images are a few hundred px square and native crops
at most ~125x101, so a 40-patch session lands in single-digit MB.
"""

from __future__ import annotations

import io
import json
import os
import shutil

import cv2
import numpy as np

from ..config import MosaicConfig
from . import models as M
from .preprocess import Patch
from .features import Features


BUNDLE_VERSION = 2


# ══ JSON helpers ══════════════════════════════════════════════════════════

def jsonable(o):
    """Convert numpy scalars/arrays and estimates into plain JSON types."""
    if isinstance(o, dict):
        if "level" in o and "dof" in o and "matrix" in o:
            return M.serialisable(o)
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _clean_record(rec):
    """A pair record without its bulky arrays (those live in the npz)."""
    out = {k: v for k, v in rec.items()
           if k not in ("matches", "inlier_mask", "estimate", "estimate_l0")}
    for k in ("estimate", "estimate_l0"):
        if rec.get(k) is not None:
            out[k] = M.serialisable(rec[k])
    return jsonable(out)


# ══ writing ═══════════════════════════════════════════════════════════════

def write(out_dir, *, session_id, cfg, patches, features, records, rotations,
          tracks, components, graph_info, opt_stats, timings, meta=None,
          mosaic=None, coverage=None, compose_info=None, log_text="",
          raw_paths=None, copy_raw=True, outlined=None, error=None):
    """Write a complete session bundle.  Creates `out_dir` if needed.

    Every argument is optional in the sense that a partial run still produces a
    valid bundle -- that is the point.  `run_session` calls this even when the
    pipeline raised, so a failed session still yields whatever was computed plus
    the traceback, rather than nothing.
    """
    os.makedirs(out_dir, exist_ok=True)
    arrays = {}

    # ── per-patch arrays ──
    for p in patches or []:
        i = p.index
        if p.S is not None:
            arrays[f"S_{i}"] = np.asarray(p.S, np.float64)
        for name, val in (("img", p.image), ("mask", p.mask),
                          ("mask_full", p.mask_full),
                          ("native_bgr", p.native_bgr),
                          ("native_mask", p.native_mask)):
            if val is not None:
                arrays[f"{name}_{i}"] = val

    for f in features or []:
        i = f.index
        if f.xy is None or not len(f.xy):
            continue
        arrays[f"kp_xy_{i}"] = f.xy.astype(np.float32)
        arrays[f"kp_size_{i}"] = f.size.astype(np.float32)
        arrays[f"kp_angle_{i}"] = f.angle.astype(np.float32)
        arrays[f"kp_response_{i}"] = f.response.astype(np.float32)
        if f.desc is not None:
            dt = np.float16 if cfg.descriptor_dtype == "float16" else np.float32
            arrays[f"desc_{i}"] = f.desc.astype(dt)

    # ── per-pair arrays ──
    for rec in records or []:
        if rec.get("matches") is None:
            continue
        i, j = rec["pair"]
        arrays[f"pair_matches_{i}_{j}"] = np.asarray(rec["matches"], np.int32)
        arrays[f"pair_inliers_{i}_{j}"] = np.asarray(rec["inlier_mask"], bool)

    # ── global solution ──
    if rotations:
        idx = sorted(rotations)
        arrays["rot_index"] = np.asarray(idx, np.int32)
        arrays["rot_matrices"] = np.stack(
            [np.asarray(rotations[n], np.float64) for n in idx])

    # Tracks are ragged; store flat with offsets rather than an object array,
    # so the npz loads without allow_pickle.
    if tracks:
        flat = np.asarray([obs for t in tracks for obs in t], np.int32)
        offs = np.cumsum([0] + [len(t) for t in tracks]).astype(np.int32)
        arrays["tracks_flat"] = flat
        arrays["tracks_offsets"] = offs

    np.savez_compressed(os.path.join(out_dir, "bundle.npz"), **arrays)

    # ── meta.json ──
    meta_out = {
        "session_id": session_id,
        "bundle_version": BUNDLE_VERSION,
        "config": cfg.to_dict(),
        "intrinsics": {
            "focal_ref_px": cfg.focal_ref_px,
            "ref_frame_width": cfg.ref_frame_width,
            "calibrated": bool(cfg.focal_is_calibrated),
            "note": ("UNCALIBRATED placeholder.  At current patch size the "
                     "projective terms of K R K^-1 are ~3e-4 px, so L0 is "
                     "numerically a Euclidean transform and this value has "
                     "almost no influence.  It matters once patches grow."),
            "per_patch": {str(p.index): {"frame_shape": list(p.frame_shape),
                                         "analysis_scale": p.scale,
                                         "focal_analysis_px": p.focal,
                                         "centre": list(p.centre)}
                          for p in (patches or [])},
        },
        # No calibrated gaze metadata is recorded by the capture path, and an
        # uncalibrated fixation offset is not a reliable prior anyway (the eye
        # drifts and twitches between the target moving and the flash firing).
        # The field is here so a future calibrated capture can populate it
        # without a format change; the pipeline runs with or without it.
        "fixation_targets": (meta or {}).get("fixation_targets"),
        "capture": (meta or {}).get("capture", {}),
        "patches": [{"index": p.index,
                     "path": os.path.basename(p.path or ""),
                     "session": _session_of(p.path),
                     "frame_shape": list(p.frame_shape),
                     "bbox": list(p.bbox) if p.bbox else None}
                    for p in (patches or [])],
        # Which sittings went in.  A run may combine several: the pipeline has
        # no temporal assumptions, so extra captures of the same eye are simply
        # extra chances to find an overlap.
        "sessions": sorted({_session_of(p.path) for p in (patches or [])
                            if p.path}),
    }
    _write_json(os.path.join(out_dir, "meta.json"), meta_out)

    # ── bundle.json ──
    body = {
        "session_id": session_id,
        "bundle_version": BUNDLE_VERSION,
        "patches": [{"index": p.index,
                     "path": os.path.basename(p.path or ""),
                     "session": _session_of(p.path),
                     "accepted": bool(p.accepted),
                     "reason": p.reason,
                     "quality": jsonable(p.quality),
                     "equiv_radius_px": p.radius,
                     "analysis_scale": p.scale,
                     "n_keypoints": next((len(f) for f in (features or [])
                                          if f.index == p.index), 0)}
                    for p in (patches or [])],
        "pairs": [_clean_record(r) for r in (records or [])],
        "graph": jsonable(graph_info or {}),
        "components": jsonable(components or []),
        "optimisation": jsonable(opt_stats or {}),
        "tracks": {"n": len(tracks or []),
                   "lengths": [len(t) for t in (tracks or [])]},
        "compose": jsonable(compose_info or {}),
        "timings": jsonable(timings or {}),
        "error": error,
    }
    _write_json(os.path.join(out_dir, "bundle.json"), body)

    # ── images ──
    if mosaic is not None:
        cv2.imwrite(os.path.join(out_dir, "mosaic.png"), mosaic)
        from .compose import preview as _preview
        cv2.imwrite(os.path.join(out_dir, "mosaic_preview.jpg"),
                    _preview(mosaic, cfg), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if outlined is not None:
        cv2.imwrite(os.path.join(out_dir, "mosaic_outlined.png"), outlined)
    if coverage is not None:
        from .compose import coverage_image
        cv2.imwrite(os.path.join(out_dir, "coverage.png"),
                    coverage_image(coverage))

    with io.open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as fh:
        fh.write(log_text or "")

    # ── raw/ ──
    if copy_raw and raw_paths:
        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        # When a run combines sittings, capture filenames collide across them
        # (every session starts again at redeye_01_...), so raw/ is split by
        # session.  A single-session run keeps the flat layout.
        multi = len({_session_of(p) for p in raw_paths}) > 1
        for src in raw_paths:
            try:
                sub = os.path.join(raw_dir, _session_of(src)) if multi else raw_dir
                os.makedirs(sub, exist_ok=True)
                dst = os.path.join(sub, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
            except OSError:
                pass
    return out_dir


def _session_of(path):
    """The session folder a capture came from -- its immediate parent."""
    if not path:
        return None
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def _write_json(path, obj):
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=2, sort_keys=True)


# ══ reading ═══════════════════════════════════════════════════════════════

def read(bundle_dir):
    """Load a bundle back into the objects the core functions take.

    Returns a dict with keys: cfg, meta, body, patches, features, records,
    rotations, tracks -- enough to re-run `pairwise`, `graph` and `globalopt`
    without the Pi and without the raw images.
    """
    with io.open(os.path.join(bundle_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with io.open(os.path.join(bundle_dir, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    z = np.load(os.path.join(bundle_dir, "bundle.npz"))
    cfg = MosaicConfig.from_dict(meta.get("config", {}))

    per = meta.get("intrinsics", {}).get("per_patch", {})
    patches, features = [], []
    for pj in body.get("patches", []):
        i = int(pj["index"])
        p = Patch(index=i, path=pj.get("path", ""))
        p.accepted = bool(pj.get("accepted", False))
        p.reason = pj.get("reason", "")
        p.quality = pj.get("quality", {})
        p.radius = float(pj.get("equiv_radius_px") or 0.0)
        p.scale = float(pj.get("analysis_scale") or 1.0)
        info = per.get(str(i), {})
        p.frame_shape = tuple(info.get("frame_shape", (0, 0)))
        p.focal = float(info.get("focal_analysis_px") or 0.0)
        p.centre = tuple(info.get("centre", (0.0, 0.0)))
        for name, attr in (("S", "S"), ("img", "image"), ("mask", "mask"),
                           ("mask_full", "mask_full"),
                           ("native_bgr", "native_bgr"),
                           ("native_mask", "native_mask")):
            k = f"{name}_{i}"
            if k in z:
                setattr(p, attr, z[k])
        if p.native_mask is not None:
            dt = cv2.distanceTransform(p.native_mask, cv2.DIST_L2, 3)
            p.native_weight = (dt / max(float(dt.max()), 1e-6)).astype(np.float32)
        if p.mask is not None:
            p.area = int(np.count_nonzero(p.mask))
        patches.append(p)

        f = Features(index=i)
        if f"kp_xy_{i}" in z:
            f.xy = z[f"kp_xy_{i}"]
            f.size = z[f"kp_size_{i}"]
            f.angle = z[f"kp_angle_{i}"]
            f.response = z[f"kp_response_{i}"]
            if f"desc_{i}" in z:
                f.desc = z[f"desc_{i}"].astype(np.float32)
        features.append(f)

    records = []
    for rj in body.get("pairs", []):
        rec = dict(rj)
        i, j = rec["pair"]
        km, ki = f"pair_matches_{i}_{j}", f"pair_inliers_{i}_{j}"
        rec["matches"] = z[km] if km in z else None
        rec["inlier_mask"] = z[ki] if ki in z else None
        for k in ("estimate", "estimate_l0"):
            e = rec.get(k)
            if isinstance(e, dict):
                rec[k] = _revive(e)
        records.append(rec)

    rotations = {}
    if "rot_index" in z:
        for n, R in zip(z["rot_index"].tolist(), z["rot_matrices"]):
            rotations[int(n)] = R

    tracks = []
    if "tracks_flat" in z:
        flat, offs = z["tracks_flat"], z["tracks_offsets"]
        for a, b in zip(offs[:-1], offs[1:]):
            tracks.append([tuple(int(v) for v in obs) for obs in flat[a:b]])

    return {"cfg": cfg, "meta": meta, "body": body, "patches": patches,
            "features": features, "records": records, "rotations": rotations,
            "tracks": tracks}


def _revive(e):
    """A serialised estimate back into the in-memory form."""
    out = dict(e)
    for k in ("matrix", "rotvec", "coeffs", "T"):
        v = e.get(k)
        out[k] = None if v is None else np.asarray(v, np.float64)
    return out
