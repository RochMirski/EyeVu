"""Validate the whole pipeline against a synthetic session's ground truth.

Generates a session with known rotations, runs the real pipeline over it, and
scores what came back.  This is what turns "the mosaic looks plausible" into a
number, and it is the harness the thresholds in `config.py` were measured with.

Reported:
    edge precision / recall   against pairs that genuinely overlap
    edge rotation error       per accepted edge, vs the true relative rotation
    cycle closure errors      what the graph filter actually sees
    global rotation error     after refinement, gauge removed
    connectivity              how many components the session broke into
    stage timings and RSS

    python -m eyevu_mosaic.validate --n 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import cv2
import numpy as np

from .config import MosaicConfig
from .core import graph as GR
from .core import models as M
from .core import pairwise, preprocess
from .core import features as F
from . import synth
from .run_session import load_session, run


def _true_overlap(pa, pb, Rij, cfg):
    """Fraction of the smaller patch that the two share under the true rotation."""
    H = M.H_from_rotation(Rij, preprocess.patch_K(pa), preprocess.patch_K(pb))
    ha, wa = pa.image.shape[:2]
    wm = cv2.warpPerspective(pb.mask, np.linalg.inv(H), (wa, ha),
                             flags=cv2.INTER_NEAREST)
    ov = int(((wm > 0) & (pa.mask > 0)).sum())
    return ov / max(1, min(pa.area, int((wm > 0).sum())))


def validate(n=20, seed=1, cfg=None, out_dir=None, overlap_gate=0.35,
             session_dir=None, verbose=True, report=True):
    """Generate, run, and score.  Returns a dict of metrics.

    `verbose` controls the pipeline's own log; `report` the summary table.
    """
    cfg = cfg or MosaicConfig()
    tmp = None
    if session_dir is None:
        tmp = tempfile.mkdtemp(prefix="eyevu_synth_")
        session_dir = tmp
        truth = synth.generate(session_dir, n=n, seed=seed)
    else:
        with open(os.path.join(session_dir, "ground_truth.json"),
                  encoding="utf-8") as fh:
            truth = json.load(fh)

    out_dir = out_dir or os.path.join(session_dir, "bundle")
    result = run(session_dir, out_dir, cfg, verbose=verbose)

    GT = {p["index"]: np.asarray(p["rotation"], np.float64)
          for p in truth["patches"]}

    # Re-derive patches/features so edges can be scored against true overlap.
    loaded = load_session(session_dir)
    ps = [preprocess.prepare(i, im, m, cfg, path=p)
          for i, (p, im, m) in enumerate(loaded)]
    det = F.SIFTDetector(cfg)
    fs = [F.detect(p, cfg, det) for p in ps]

    with open(os.path.join(out_dir, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    accepted = {tuple(r["pair"]): r for r in body["pairs"] if r["accepted"]}

    tp = fp = fn = 0
    edge_err, missed = [], []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if not (ps[i].accepted and ps[j].accepted):
                continue
            Rij = GT[j].T @ GT[i]
            frac = _true_overlap(ps[i], ps[j], Rij, cfg)
            rec = accepted.get((i, j))
            real = frac >= overlap_gate
            if rec and real:
                tp += 1
                R = M.so3_exp(np.asarray(rec["estimate_l0"]["rotvec"]))
                edge_err.append(M.rotation_angle_deg(Rij.T @ R))
            elif rec and not real:
                fp += 1
            elif real:
                fn += 1
                missed.append((i, j, round(frac, 2)))

    # Global rotations, gauge removed, per component (a component has its own
    # arbitrary frame, so they cannot be scored against one another).
    from .core import bundle as bundle_io
    b = bundle_io.read(out_dir)
    comps = body.get("components", [])
    focal = float(np.median([p.focal for p in ps if p.accepted] or [3600.0]))
    radius = float(np.median([p.radius for p in ps if p.accepted] or [70.0]))
    per_comp = []
    for c in comps:
        rot = {int(k): v for k, v in b["rotations"].items() if int(k) in set(c)}
        if len(rot) >= 2:
            per_comp.append(synth.compare(truth, rot, focal, radius))

    cyc = body.get("graph", {}).get("cycle", {})
    tri = [t["error_deg"] for t in cyc.get("triangles", [])]
    metrics = {
        "n_patches": len(ps),
        "edge_precision": tp / max(1, tp + fp),
        "edge_recall": tp / max(1, tp + fn),
        "edges_tp": tp, "edges_fp": fp, "edges_fn": fn,
        "edge_rotation_error_deg": {
            "median": float(np.median(edge_err)) if edge_err else None,
            "p90": float(np.percentile(edge_err, 90)) if edge_err else None,
            "max": float(np.max(edge_err)) if edge_err else None},
        "triangle_closure_deg": {
            "n": len(tri),
            "median": float(np.median(tri)) if tri else None,
            "max": float(np.max(tri)) if tri else None},
        "edges_dropped_on_cycle": len(cyc.get("dropped", [])),
        "components": [len(c) for c in comps],
        "largest_component": max((len(c) for c in comps), default=0),
        "global_rotation_error_deg": [
            {"n": p["n"], "median": p["median_deg"], "max": p["max_deg"],
             "median_pointing_deg": p.get("median_pointing_deg"),
             "median_roll_deg": p.get("median_roll_deg"),
             "median_displacement_px": p.get("median_displacement_px"),
             "max_displacement_px": p.get("max_displacement_px")}
            for p in per_comp],
        "analysis_px_per_native_px": float(cfg.analysis_upsample),
        "timings": result["timings"],
        "missed_overlaps": missed[:15],
    }

    if report:
        _report(metrics)
    return metrics


def _report(m):
    print("\n" + "=" * 66)
    print("VALIDATION vs GROUND TRUTH")
    print("=" * 66)
    print(f"  patches                {m['n_patches']}")
    print(f"  edge precision         {m['edge_precision']:.2f}   "
          f"({m['edges_tp']} true, {m['edges_fp']} false)")
    print(f"  edge recall            {m['edge_recall']:.2f}   "
          f"({m['edges_fn']} genuine overlaps missed)")
    e = m["edge_rotation_error_deg"]
    if e["median"] is not None:
        print(f"  edge rotation error    median {e['median']:.3f} deg, "
              f"p90 {e['p90']:.3f}, max {e['max']:.3f}")
    t = m["triangle_closure_deg"]
    if t["n"]:
        print(f"  triangle closure       n={t['n']}, median {t['median']:.3f} deg, "
              f"max {t['max']:.3f}")
    print(f"  edges dropped on cycle {m['edges_dropped_on_cycle']}")
    print(f"  components             {m['components']}")
    up = m.get("analysis_px_per_native_px", 1.0)
    for g in m["global_rotation_error_deg"]:
        print(f"  global error ({g['n']:2d} patches)")
        print(f"      total angle        median {g['median']:.3f} deg, "
              f"max {g['max']:.3f}")
        print(f"      pointing / roll    {g['median_pointing_deg']:.3f} deg / "
              f"{g['median_roll_deg']:.3f} deg   (roll barely moves the image)")
        print(f"      ALIGNMENT          median "
              f"{g['median_displacement_px']:.1f} analysis px "
              f"= {g['median_displacement_px'] / up:.2f} native px, "
              f"max {g['max_displacement_px'] / up:.2f} native px")
    tot = m["timings"].get("_total", {})
    print(f"  total                  {tot.get('seconds')}s, "
          f"peak RSS {tot.get('peak_rss_mb')} MB")
    print("=" * 66)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--session", default=None,
                    help="existing synthetic session (must have ground_truth.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overlap-gate", type=float, default=0.35)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    cfg = MosaicConfig()
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            cfg = MosaicConfig.from_json(fh.read())
    validate(n=a.n, seed=a.seed, cfg=cfg, out_dir=a.out,
             session_dir=a.session, overlap_gate=a.overlap_gate,
             verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
