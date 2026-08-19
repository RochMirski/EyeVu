"""Pi entry point: orchestration, timing, resource accounting, error handling.

This is the only module allowed to do IO, and it is deliberately thin -- every
decision lives in `core/`, so the dashboard reproduces a run by calling the same
functions in the same order without going through here.

Two guarantees this module owns:

*   THE RUN NEVER CRASHES THE DEVICE.  Any failure at any stage still writes a
    session bundle containing whatever was computed plus the traceback.  A
    session that half-worked is diagnostic gold; a session that vanished is not.

*   Every stage is instrumented with wall-clock time and peak RSS, and the
    figures go into the bundle.

    python -m eyevu_mosaic.run_session <session_dir> [-o out_dir] [--json cfg]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from glob import glob

import cv2
import numpy as np

from .config import MosaicConfig
from .core import bundle as bundle_io
from .core import compose, features as feat_mod, globalopt, graph
from .core import pairwise, preprocess


# ══ resource accounting ═══════════════════════════════════════════════════

def peak_rss_mb():
    """Peak resident set size in MB, or None where unavailable.

    `resource` is the cheap route and exists on the Pi; psutil is a fallback for
    the dev machine.  Neither is a hard dependency.
    """
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS bytes.
        return round(v / (1024.0 if sys.platform != "darwin" else 1048576.0), 1)
    except (ImportError, AttributeError):
        pass
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1048576.0, 1)
    except Exception:                                       # noqa: BLE001
        return None


class Timer:
    """Per-stage wall clock and peak RSS, collected into the bundle."""

    def __init__(self, log):
        self.stages = {}
        self.log = log
        self._t0 = time.time()

    def stage(self, name):
        return _Stage(self, name)

    def total(self):
        return round(time.time() - self._t0, 3)


class _Stage:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        dt = round(time.time() - self.t, 3)
        rss = peak_rss_mb()
        self.timer.stages[self.name] = {"seconds": dt, "peak_rss_mb": rss}
        self.timer.log(f"  [{self.name}] {dt:.2f}s"
                       + (f", peak RSS {rss} MB" if rss else ""))
        return False


# ══ session loading ═══════════════════════════════════════════════════════

def load_session(session_dir):
    """Load a session's extracts and masks in capture order.

    Reads the layout the capture path already writes:
        redeye_<NN>_<HHMMSS>_extract.png   the isolated fundus pixels on black
        redeye_<NN>_<HHMMSS>_mask.png      the matching selection

    Returns a list of (path, bgr, mask).
    """
    out = []
    for p in sorted(glob(os.path.join(session_dir, "redeye_*_extract.png"))):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        mp = p.replace("_extract.png", "_mask.png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp) else None
        if m is not None and m.shape[:2] != img.shape[:2]:
            m = None
        out.append((p, img, m))
    return out


def load_sessions(session_dirs):
    """Load SEVERAL sessions as one pool of captures.

    Nothing downstream needs to know: a patch is a patch, and the pipeline has
    no notion of capture order or temporal continuity to violate -- it already
    treats every pair independently.  So combining sessions is just a longer
    list, and it is worth doing because a single sitting often fragments into
    disconnected groups that a second sitting of the same eye can bridge.

    The one thing that DOES matter is provenance: which session each patch came
    from is recorded per patch, so a group that turns out to span two sittings
    is visibly that rather than a mystery.

    Sessions are visited in the order given, and captures within each in capture
    order.  Returns a list of (path, bgr, mask).
    """
    out = []
    for d in session_dirs:
        out.extend(load_session(d))
    return out


def session_of(path):
    """Which session folder a capture came from -- its immediate parent."""
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


# ══ the run ═══════════════════════════════════════════════════════════════

def run(session_dir, out_dir=None, cfg=None, verbose=True, session_id=None):
    """Mosaic one session, or several combined.  Always writes a bundle.

    `session_dir` may be a single path or a list of paths.  Combining sittings
    of the same eye is often what turns disconnected groups into one mosaic:
    the pipeline has no temporal assumptions to break, so extra captures are
    simply extra chances to find an overlap.

    With several inputs the bundle lands in the FIRST one's `bundle/` unless
    `out_dir` says otherwise, and every patch records the session it came from.

    Returns a result dict with keys: ok, out_dir, n_patches, n_accepted,
    components, error.
    """
    cfg = cfg or MosaicConfig()
    dirs = ([session_dir] if isinstance(session_dir, (str, bytes, os.PathLike))
            else list(session_dir))
    dirs = [os.path.normpath(str(d)) for d in dirs]
    if not dirs:
        raise ValueError("no session directory given")
    names = [os.path.basename(d) for d in dirs]
    session_id = session_id or (names[0] if len(dirs) == 1
                                else f"combined_{len(dirs)}_sessions")
    out_dir = out_dir or os.path.join(dirs[0], "bundle")
    lines = []

    def log(msg):
        lines.append(str(msg))
        if verbose:
            print(msg)

    cv2.setNumThreads(int(cfg.cv_num_threads))
    timer = Timer(log)
    state = {"patches": [], "features": [], "records": [], "rotations": {},
             "tracks": [], "components": [], "graph_info": {}, "opt": {},
             "compose": {}, "mosaic": None, "coverage": None, "outlined": None,
             "raw": []}
    error = None

    try:
        log(f"session {session_id}"
            + (f"  ({len(dirs)} sessions combined: {', '.join(names)})"
               if len(dirs) > 1 else ""))
        with timer.stage("load"):
            loaded = load_sessions(dirs)
            # Both halves of a capture are original data: the extract and the
            # mask that says which of its pixels are real.  raw/ keeps both, so
            # the bundle can be re-run from source and not only from the npz.
            state["raw"] = []
            for p, _img, _m in loaded:
                state["raw"].append(p)
                mp = p.replace("_extract.png", "_mask.png")
                if os.path.isfile(mp):
                    state["raw"].append(mp)
            if len(dirs) > 1:
                per = {}
                for p, _i, _m in loaded:
                    per[session_of(p)] = per.get(session_of(p), 0) + 1
                log(f"  {len(loaded)} capture(s): "
                    + ", ".join(f"{k} x{v}" for k, v in per.items()))
            else:
                log(f"  {len(loaded)} capture(s)")
        if not loaded:
            raise RuntimeError(f"no captures found in {', '.join(dirs)}")

        # ── preprocess + quality gate ──
        with timer.stage("preprocess"):
            patches = [preprocess.prepare(i, img, m, cfg, path=p)
                       for i, (p, img, m) in enumerate(loaded)]
            state["patches"] = patches
            good = [p for p in patches if p.accepted]
            for p in patches:
                if not p.accepted:
                    log(f"  [{p.index}] REJECTED - {p.reason}")
            log(f"  {len(good)}/{len(patches)} passed the quality gate")

        # ── features ──
        with timer.stage("features"):
            det = feat_mod.SIFTDetector(cfg)
            fs = [feat_mod.detect(p, cfg, det) for p in good]
            state["features"] = fs
            for p, f in zip(good, fs):
                if len(f) < cfg.min_keypoints:
                    p.accepted = False
                    p.reason = f"only {len(f)} keypoints"
                    log(f"  [{p.index}] REJECTED - {p.reason}")
            keep = [(p, f) for p, f in zip(good, fs) if p.accepted]
            good = [p for p, _ in keep]
            fs = [f for _, f in keep]
            counts = [len(f) for f in fs]
            if counts:
                log(f"  keypoints: total {sum(counts)}, min {min(counts)}, "
                    f"median {int(np.median(counts))}, max {max(counts)}")

        if len(good) < 2:
            raise RuntimeError(f"need >= 2 usable captures, have {len(good)}")

        # ── shortlist + pairwise ──
        with timer.stage("shortlist"):
            pairs, skipped = pairwise.shortlist(good, cfg)
            log(f"  {len(pairs)} candidate pair(s), {len(skipped)} skipped")

        with timer.stage("match"):
            by_i = {p.index: p for p in good}
            by_f = {f.index: f for f in fs}
            rng = np.random.default_rng(int(cfg.shortlist_seed))
            records = []
            for a, b in pairs:
                rec = pairwise.match_pair(by_i[a], by_i[b], by_f[a], by_f[b],
                                          cfg, rng)
                records.append(rec)
            state["records"] = records
            acc = [r for r in records if r["accepted"]]
            log(f"  {len(acc)}/{len(records)} pair(s) accepted")
            for r in sorted(acc, key=lambda r: -r["n_inliers"])[:10]:
                log(f"    {r['pair'][0]}-{r['pair'][1]}: "
                    f"{r['n_inliers']}/{r['n_putative']} inliers, "
                    f"L{r['level']}, ncc {r['ncc']:.2f}")

        # ── graph ──
        with timer.stage("graph"):
            nodes, edges = graph.build(records)
            # Cycle errors are scored in image pixels, so the filter needs the
            # session's actual focal and patch size -- see graph.cycle_filter.
            edges, cyc = graph.cycle_filter(
                nodes, edges, cfg,
                focal=float(np.median([p.focal for p in good])),
                radius=float(np.median([p.radius for p in good])))
            comps = graph.components(nodes, edges)
            state["components"] = comps
            log(f"  {len(cyc['triangles'])} triangle(s) tested, "
                f"{len(cyc['dropped'])} edge(s) dropped on cycle error")
            log(f"  {len(comps)} connected component(s): "
                f"{[len(c) for c in comps]}")

        if not comps:
            raise RuntimeError("no pair survived verification; nothing to place")

        # ── initialise + refine, per component ──
        trees, all_rot, opt = [], {}, {}
        med_f = float(np.median([p.focal for p in good]))
        med_r = float(np.median([p.radius for p in good]))
        with timer.stage("global_opt"):
            for ci, comp in enumerate(comps):
                if len(comp) < 2:
                    trees.append([])
                    all_rot[comp[0]] = np.eye(3)
                    continue
                sub = {k: v for k, v in edges.items()
                       if k[0] in set(comp) and k[1] in set(comp)}
                tree = graph.max_spanning_tree(comp, sub)
                trees.append(tree)
                G = graph.initial_rotations(comp, sub, tree)
                G, st = globalopt.refine(good, fs, records, comp, sub, G, cfg,
                                         focal=med_f, radius=med_r)
                opt[str(ci)] = st
                all_rot.update(G)
                log(f"  component {ci} ({len(comp)} patches): {st['method']}")
                for d in st.get("outlier_edges_dropped", []):
                    log(f"    edge {d['edge']} residual {d['residual_px']:.1f}px"
                        + (f" KEPT ({d['kept']})" if d.get("kept") else " dropped"))
                ba = st.get("bundle_adjustment", {})
                if ba.get("ran"):
                    log(f"    BA {ba['n_parameters']} params, "
                        f"RMS {ba['rms_before_px']:.2f} -> "
                        f"{ba['rms_after_px']:.2f} px")
            state["rotations"] = all_rot
            state["opt"] = opt
            state["graph_info"] = dict(
                graph.summarise(nodes, edges, comps, trees), cycle=cyc,
                skipped_pairs=skipped)

        tracks, _ = globalopt.build_tracks(records, cfg)
        state["tracks"] = tracks

        # ── composite, one mosaic per component ──
        with timer.stage("compose"):
            mosaics = []
            for ci, comp in enumerate(comps):
                G = {n: all_rot[n] for n in comp if n in all_rot}
                if len(G) < 1:
                    continue
                img, cov, proj, cinfo = compose.composite(good, G, cfg)
                cinfo["component"] = ci
                cinfo["patches"] = list(comp)
                # Where each patch's own boundary landed.  Kept as geometry in
                # the bundle as well as rendered, so the dashboard can hit-test
                # or recolour rather than being stuck with a flattened picture.
                outl = compose.patch_outlines(good, G, proj, img.shape)
                cinfo["outlines"] = {str(k): v.tolist() for k, v in outl.items()}
                over = (compose.draw_outlines(img, outl, dim=cfg.outline_dim,
                                              thickness=cfg.outline_thickness)
                        if cfg.write_outlined_mosaic else None)
                mosaics.append((ci, img, cov, cinfo, over))
                log(f"  component {ci}: {cinfo['width']}x{cinfo['height']}, "
                    f"max stack {cinfo['max_stack_depth']}, "
                    f"{cinfo['covered_px']} covered px")
            state["compose"] = {"components": [m[3] for m in mosaics]}
            if mosaics:
                state["mosaic"], state["coverage"] = mosaics[0][1], mosaics[0][2]
                state["outlined"] = mosaics[0][4]

        # Multiple components mean the session did not connect.  Each is emitted
        # separately and the failure is reported.  No attempt is made to guess a
        # relative placement: there is no calibrated gaze prior to do it with,
        # and a fabricated placement would look like a result.
        if len(comps) > 1:
            log(f"  WARNING: {len(comps)} disconnected group(s). "
                f"Emitting one mosaic each; they cannot be placed relative to "
                f"one another without a calibrated gaze prior.")
            os.makedirs(out_dir, exist_ok=True)
            for ci, img, cov, _info, over in mosaics[1:]:
                cv2.imwrite(os.path.join(out_dir, f"mosaic_component_{ci}.png"),
                            img)
                cv2.imwrite(os.path.join(out_dir, f"coverage_component_{ci}.png"),
                            compose.coverage_image(cov))
                if over is not None:
                    cv2.imwrite(os.path.join(
                        out_dir, f"mosaic_component_{ci}_outlined.png"), over)

    except Exception:                                       # noqa: BLE001
        error = traceback.format_exc()
        log("PIPELINE ERROR:\n" + error)
        if not cfg.fail_soft:
            raise
    finally:
        gc.collect()
        timings = dict(timer.stages)
        timings["_total"] = {"seconds": timer.total(),
                             "peak_rss_mb": peak_rss_mb()}
        try:
            bundle_io.write(
                out_dir, session_id=session_id, cfg=cfg,
                patches=state["patches"], features=state["features"],
                records=state["records"], rotations=state["rotations"],
                tracks=state["tracks"], components=state["components"],
                graph_info=state["graph_info"], opt_stats=state["opt"],
                timings=timings, mosaic=state["mosaic"],
                coverage=state["coverage"], compose_info=state["compose"],
                log_text="\n".join(lines), raw_paths=state["raw"],
                outlined=state["outlined"], error=error)
            log(f"  bundle written: {out_dir}")
        except Exception:                                   # noqa: BLE001
            # Last resort: if even the bundle write fails, say so and keep the
            # traceback on stdout rather than dying silently.
            print("BUNDLE WRITE FAILED:\n" + traceback.format_exc())

    return {"ok": error is None and state["mosaic"] is not None,
            "out_dir": out_dir,
            "n_patches": len(state["patches"]),
            "n_accepted": sum(1 for p in state["patches"] if p.accepted),
            "components": [list(c) for c in state["components"]],
            "n_pairs_accepted": sum(1 for r in state["records"]
                                    if r.get("accepted")),
            "timings": timings,
            "error": error}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session_dir", nargs="+",
                    help="one session folder, or several to combine into one "
                         "mosaic (different sittings of the same eye)")
    ap.add_argument("-o", "--out", default=None, help="bundle output directory")
    ap.add_argument("--name", default=None, help="session id for the bundle")
    ap.add_argument("--config", default=None, help="JSON file of config overrides")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    cfg = MosaicConfig()
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            cfg = MosaicConfig.from_json(fh.read())
    r = run(a.session_dir, a.out, cfg, verbose=not a.quiet, session_id=a.name)
    if not a.quiet:
        t = r["timings"]["_total"]
        print(f"\n{'OK' if r['ok'] else 'INCOMPLETE'} - "
              f"{r['n_accepted']}/{r['n_patches']} patches, "
              f"{r['n_pairs_accepted']} pairs, "
              f"components {[len(c) for c in r['components']]}, "
              f"{t['seconds']:.1f}s, peak RSS {t['peak_rss_mb']} MB")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
