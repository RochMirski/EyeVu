#!/usr/bin/env python3
"""Run the mosaicing algorithm on a captured session, and show all its workings.

Pick a session folder in a file dialog (opening at Sessions/), and this runs the
pipeline end to end on it and puts every intermediate on screen:

    Captures        the raw red-eye extracts that went in
    Keypoints       each patch as the DETECTOR sees it, with its SIFT points
    Mosaic          the stitched result — press `b` to outline each contributing
                    patch in its own colour, labelled with its index
    Coverage        how many patches contributed to each output pixel (new only)

...plus, on the console, the full pair graph, the model level and photometric
score of every accepted pair, why each rejected pair was rejected, the connected
components, and the per-stage timings.

Runs `eyevu_mosaic` by DEFAULT.  `--old` runs the previous mosaic.py instead, so
a tuning change can still be judged against the old behaviour on the same
captures.  The two write different things into the session: the new pipeline
writes a full bundle/ directory (see eyevu_mosaic/README.md), the old one writes
mosaic.png and keypoints.png beside the captures.  --no-save turns both off.

Several sessions can be selected at once (ctrl/shift-click) and are mosaicked
TOGETHER.  That is worth doing rather than a curiosity: a single sitting often
fragments into disconnected groups, and a second sitting of the same eye
frequently bridges them.  Measured on two real sessions that placed 5 and 6
captures on their own: combined, 20 of 22, with 17 of the 37 accepted pairs
joining the two sittings.

    python try_mosaic.py                       # pick in the dialog
    python try_mosaic.py <dir> [<dir> ...]     # skip the dialog; several = combine
    python try_mosaic.py --old                 # run the previous pipeline instead
    python try_mosaic.py --no-save             # write nothing into the session
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

import mosaic

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "Sessions")


def list_sessions(root_dir=None):
    """Session folders under Sessions/, newest first, with their capture counts.

    Returns a list of (path, n_captures, patient).
    """
    root_dir = root_dir or (SESSIONS if os.path.isdir(SESSIONS) else HERE)
    out = []
    for name in sorted(os.listdir(root_dir), reverse=True):
        d = os.path.join(root_dir, name)
        if not os.path.isdir(d):
            continue
        n = len(glob.glob(os.path.join(d, "redeye_*_extract.png")))
        if n:
            out.append((d, n, session_patient(d)))
    return out


def session_patient(session_dir):
    """Patient this session belongs to.

    Read from the session's own `session.json` where the capture path wrote one.
    Sessions recorded before patient numbering existed have no such file, and
    they are all the same patient — hence the default.  See cap.PATIENT_ID.
    """
    p = os.path.join(session_dir, "session.json")
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return str(json.load(fh).get("patient", "0"))
        except (OSError, ValueError):
            pass
    return "0"


def pick_sessions(start=None):
    """Multi-select list of capture sessions.  [] if cancelled.

    Several sessions can be selected and are mosaicked TOGETHER.  That is worth
    doing: a single sitting often fragments into disconnected groups, and a
    second sitting of the same eye frequently bridges them — measured on two
    real sessions that placed 5 and 6 captures separately but 20 of 22 combined.
    Ctrl-click / shift-click to multi-select.
    """
    try:
        import tkinter as tk
    except ImportError:                              # headless box, no tk
        print("tkinter unavailable - pass session directories as arguments.")
        return []
    rows = list_sessions(start)
    if not rows:
        print(f"No sessions with captures under {start or SESSIONS}")
        return []

    chosen = []
    root = tk.Tk()
    root.title("Pick capture session(s) to mosaic")
    root.attributes("-topmost", True)                # or it hides behind the IDE
    tk.Label(root, text=("Select one session, or several to combine them "
                         "(ctrl/shift-click).\nCombining different sittings of "
                         "the same eye often joins groups a single sitting "
                         "cannot."),
             justify="left", anchor="w").pack(fill="x", padx=10, pady=(10, 4))
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10)
    bar = tk.Scrollbar(frame)
    bar.pack(side="right", fill="y")
    box = tk.Listbox(frame, selectmode="extended", width=64,
                     height=min(20, max(6, len(rows))), yscrollcommand=bar.set)
    for d, n, pid in rows:
        box.insert("end", f"{os.path.basename(d):32s}  {n:3d} captures   "
                          f"patient {pid}")
    box.pack(side="left", fill="both", expand=True)
    bar.config(command=box.yview)
    box.selection_set(0)

    def ok():
        chosen.extend(rows[i][0] for i in box.curselection())
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=10, pady=10)
    tk.Button(btns, text="Mosaic", width=12, command=ok).pack(side="right")
    tk.Button(btns, text="Cancel", width=12,
              command=root.destroy).pack(side="right", padx=6)
    box.bind("<Double-Button-1>", lambda _e: ok())
    root.bind("<Return>", lambda _e: ok())
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return chosen


def _show(name, img, cap_h=900):
    if img is None:
        return
    vis = img
    if vis.shape[0] > cap_h:
        s = cap_h / float(vis.shape[0])
        vis = cv2.resize(vis, (int(vis.shape[1] * s), cap_h))
    cv2.imshow(name, vis)


def _content_box(img, pad=8):
    """(x0, y0, x1, y1) of the non-black content, padded.

    Computed from the PLAIN mosaic and reused for the outlined one: the two must
    be cropped identically or the image jumps as you toggle.
    """
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ys, xs = np.nonzero(g)
    if not len(xs):
        return None
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(img.shape[1], int(xs.max()) + pad + 1),
            min(img.shape[0], int(ys.max()) + pad + 1))


def _crop(img, box):
    """Crop to a shared box, so toggled variants do not jump about."""
    if img is None:
        return None
    if box is None:
        return img
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


# Window name -> list of variants; index 0 is plain, 1 (if present) has the
# per-patch boundaries drawn.  `b` cycles.
_VIEWS = {}
_VARIANT = 0


def show_views(views, variant=0):
    """Display each window at the given variant, falling back when it has none."""
    global _VIEWS, _VARIANT
    _VIEWS, _VARIANT = views, variant
    for name, variants in views.items():
        _show(name, variants[min(variant, len(variants) - 1)])


def toggle_outlines():
    """Flip every mosaic window between plain and outlined.  True if anything did."""
    if not any(len(v) > 1 for v in _VIEWS.values()):
        return False
    show_views(_VIEWS, 1 - _VARIANT)
    return True


def report_new(session, save=True):
    """Run `eyevu_mosaic` on one session and display every stage.

    The interesting difference from the old report below is what gets printed
    for a pair that did NOT register: the new pipeline records a reason for
    every rejection, which is usually what you want when tuning.
    """
    from eyevu_mosaic.config import MosaicConfig
    from eyevu_mosaic.run_session import load_sessions, run, session_of

    dirs = [session] if isinstance(session, str) else list(session)
    name = (os.path.basename(dirs[0].rstrip("/\\")) if len(dirs) == 1
            else f"{len(dirs)} sessions combined")
    loaded = load_sessions(dirs)
    print(f"\n=== {name}: {len(loaded)} capture(s)  [eyevu_mosaic] ===")
    if len(dirs) > 1:
        for d in dirs:
            print(f"    {os.path.basename(d)}  patient {session_patient(d)}")
    if not loaded:
        print("  no redeye_*_extract.png files here - is this a session folder?")
        return False

    imgs = [im for _p, im, _m in loaded]
    masks = [m for _p, _im, m in loaded]
    paths = [p for p, _im, _m in loaded]
    # With several sittings, label each capture with its session so the contact
    # sheet says where a patch came from.
    labels = [(os.path.basename(p).replace("_extract.png", "")[7:]
               if len(dirs) == 1
               else f"{session_of(p)[-6:]}/{os.path.basename(p)[7:9]}")
              for p in paths]

    # Keypoints are still drawn by the old module: it is the only thing that
    # renders them, it is unaffected by the rewrite, and the count per patch is
    # still the number that decides whether a session can register at all.
    sheet, _ = mosaic.keypoint_sheet(imgs, masks, labels)
    _show("Keypoints", sheet)
    _show("Captures", mosaic.contact_sheet(imgs, cols=4, labels=labels,
                                           masks=masks))

    out_dir = os.path.join(dirs[0], "bundle")
    res = run(dirs, out_dir, MosaicConfig(), verbose=True)

    import json
    with open(os.path.join(out_dir, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)

    acc = [p for p in body["pairs"] if p["accepted"]]
    rej = [p for p in body["pairs"] if not p["accepted"]]
    print(f"\n  pair graph: {len(acc)} accepted of {len(body['pairs'])} attempted")
    for p in sorted(acc, key=lambda p: -p["ncc"]):
        print(f"    {p['pair'][0]:2d}-{p['pair'][1]:2d}  L{p['level']}  "
              f"{p['n_inliers']:3d}/{p['n_putative']:3d} inliers  "
              f"ncc {p['ncc']:+.2f}  overlap {p['overlap_px']:6d}px  "
              f"[{p.get('source', 'features')}]")
    if rej:
        # Strip the embedded numbers, or every "rotation 110.1 deg implausible"
        # is its own category and the tally says nothing.
        import collections
        import re
        why = collections.Counter(
            re.sub(r"[-+]?\d*\.?\d+", "N", r["reason"].split(";")[0]).strip()
            for r in rej)
        print(f"  rejected ({len(rej)}), by reason:")
        for reason, n in why.most_common(8):
            print(f"    {n:4d}  {reason}")

    cyc = body.get("graph", {}).get("cycle", {})
    if cyc.get("dropped"):
        print(f"  cycle consistency dropped {len(cyc['dropped'])} edge(s):")
        for d in cyc["dropped"]:
            print(f"    {d['edge']}  closure {d['error_px']:.1f}px")
    for ci, st in sorted(body.get("optimisation", {}).items()):
        drops = [d for d in st.get("outlier_edges_dropped", []) if not d.get("kept")]
        print(f"  component {ci}: {st['method']}"
              + (f", {len(drops)} outlier edge(s) dropped" if drops else ""))

    comps = body.get("components", [])
    print(f"  components: {[len(c) for c in comps]}  ->  "
          f"{max((len(c) for c in comps), default=0)}/{len(loaded)} placed")
    if len(dirs) > 1 and comps:
        # The point of combining: did the sittings actually join up?
        sess = {p["index"]: p["session"] for p in body["patches"]}
        cross = sum(1 for p in body["pairs"] if p["accepted"]
                    and sess[p["pair"][0]] != sess[p["pair"][1]])
        import collections
        spread = collections.Counter(sess[i] for i in comps[0])
        print(f"  {cross} accepted pair(s) join different sittings; "
              f"largest group spans {dict(spread)}")
    if len(comps) > 1:
        print("  NOTE: disconnected. Each group is a separate mosaic; without a "
              "calibrated gaze prior they cannot be placed relative to each other.")
    t = res["timings"]["_total"]
    print(f"  {t['seconds']:.1f}s, peak RSS {t['peak_rss_mb']} MB")

    # Each mosaic window holds two variants -- plain, and with every
    # contributing patch's own boundary drawn in its own colour and labelled
    # with its index.  `b` toggles.  This is the quickest way to see WHICH
    # capture went where when a group looks wrong: a patch placed on top of a
    # neighbour, or one hanging off the edge on a single bad edge, is obvious
    # in the outlines and invisible in the blended result.
    views = {}

    def _pair(win, plain_path, outlined_path):
        plain = cv2.imread(plain_path, cv2.IMREAD_COLOR)
        if plain is None:
            return
        over = cv2.imread(outlined_path, cv2.IMREAD_COLOR)
        # Crop both to the SAME box, or the two variants jump as you toggle.
        box = _content_box(plain)
        variants = [_crop(plain, box)]
        if over is not None:
            variants.append(_crop(over, box))
        views[win] = variants

    _pair("Mosaic", os.path.join(out_dir, "mosaic.png"),
          os.path.join(out_dir, "mosaic_outlined.png"))
    for ci in range(1, len(comps)):
        _pair(f"Mosaic (group {ci})",
              os.path.join(out_dir, f"mosaic_component_{ci}.png"),
              os.path.join(out_dir, f"mosaic_component_{ci}_outlined.png"))
    cov = cv2.imread(os.path.join(out_dir, "coverage.png"), cv2.IMREAD_COLOR)
    if cov is not None:
        views["Coverage"] = [mosaic.crop_to_content(cov)]
    show_views(views, 0)

    if not save:
        # The run has to write the bundle to produce anything at all, so
        # --no-save removes it afterwards rather than suppressing it.
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        print("  --no-save: bundle removed")
    else:
        print(f"  bundle: {out_dir}")
    return True


def report(session, save=True):
    """Run the OLD mosaic.py on one session and display every stage."""
    imgs, masks, paths = mosaic.load_session(session)
    name = os.path.basename(session.rstrip("/\\"))
    print(f"\n=== {name}: {len(imgs)} capture(s)  [old mosaic.py] ===")
    if not imgs:
        print("  no redeye_*_extract.png files here - is this a session folder?")
        return False

    sizes = []
    for m, im in zip(masks, imgs):
        bb = mosaic.content_bbox(mosaic.mask_for(im, m))
        sizes.append(f"{bb[2]}x{bb[3]}" if bb else "empty")
    print(f"  patch sizes: {', '.join(sizes)}")

    labels = [os.path.basename(p).replace("_extract.png", "")[7:] for p in paths]

    # ── keypoints: the number that decides whether anything can match at all ──
    sheet, _ = mosaic.keypoint_sheet(imgs, masks, labels)
    _show("Keypoints", sheet)
    if save and sheet is not None:
        cv2.imwrite(os.path.join(session, "keypoints.png"), sheet)

    _show("Captures", mosaic.contact_sheet(imgs, cols=4, labels=labels,
                                           masks=masks))

    # ── the stitch itself, with its pass-by-pass working printed ──
    mos, info = mosaic.stitch(imgs, masks, verbose=True)
    mos = mosaic.crop_to_content(mos)

    print(f"\n  passes: {info['passes']}   detector: {info['detector']}")
    if info["pairs"]:
        print("  pair graph (first pass):")
        for p in info["pairs"]:
            print(f"    {p['a']:2d}-{p['b']:2d}  {p['inliers']:2d} inliers  "
                  f"ncc {p['ncc']:+.2f}  {p['model']}")
    else:
        print("  pair graph: EMPTY - no two captures registered")
    print(f"  pieces: {info['pieces']}")
    print(f"  used {info['used']}/{len(imgs)}"
          + (f", mosaic {mos.shape[1]}x{mos.shape[0]}" if mos is not None else ""))
    if info["skipped"]:
        print(f"  not in the mosaic: {[s['index'] for s in info['skipped']]}")

    _show("Mosaic", mos)
    if save and mos is not None:
        out = os.path.join(session, "mosaic.png")
        cv2.imwrite(out, mos)
        print(f"  written: {out}")
    return True


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    save = "--no-save" not in argv
    use_old = "--old" in argv
    run_one = report if use_old else report_new
    sessions = args or None
    print("pipeline: " + ("old mosaic.py" if use_old
                          else "eyevu_mosaic  (--old for the previous one)"))

    while True:
        if sessions is None:
            sessions = pick_sessions()
            if not sessions:
                print("Nothing picked.")
                return 0
        bad = [s for s in sessions if not os.path.isdir(s)]
        if bad:
            print(f"Not a directory: {bad[0]}")
            return 2
        if len(sessions) > 1 and use_old:
            # The old pipeline has no notion of combining sittings.
            print(f"--old takes one session at a time; using {sessions[0]}")
            sessions = sessions[:1]

        cv2.destroyAllWindows()
        try:
            run_one(sessions if not use_old else sessions[0], save=save)
        except Exception as e:                       # noqa: BLE001 - a tuning run
            import traceback                         # should show the fault, not die
            print(f"\n[ERROR] {e!r}")
            traceback.print_exc()

        print("\n  any window focused: b = toggle patch boundaries, "
              "SPACE/o = pick another session, r = re-run this one, ESC/q = quit")
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k != ord("b"):
                break
            # Stay in the loop: toggling should not cost you the session.
            if not toggle_outlines():
                print("  (no boundary overlay for this view)")
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            continue                                 # same session(s), re-run
        sessions = None                              # anything else: pick again
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
