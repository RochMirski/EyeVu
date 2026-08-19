#!/usr/bin/env python3
"""Run the mosaicing algorithm on a captured session, and show all its workings.

Pick a session folder in a file dialog (opening at Sessions/), and this runs the
current mosaic.py end to end on it and puts every intermediate on screen:

    Captures        the raw red-eye extracts that went in
    Keypoints       each patch as the DETECTOR sees it, with its SIFT points —
                    green kept, dim red discarded for touching the black surround
    Mosaic          the stitched result

...plus, on the console, the full pair graph with inlier counts and NCC scores,
the pass-by-pass agglomeration, and which captures ended up where.

The point is to be re-run after every change to the algorithm on data already
collected, so a tuning change can be judged against real captures rather than
against the next patient.  Nothing here writes to the session except the output
images, and --no-save turns even that off.

    python try_mosaic.py                 # pick a folder in the dialog
    python try_mosaic.py <session_dir>   # skip the dialog
    python try_mosaic.py --no-save       # do not write anything into the session
"""

from __future__ import annotations

import os
import sys

import cv2

import mosaic

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "Sessions")


def pick_session(start=None):
    """File-explorer folder picker, opening at Sessions/.  None if cancelled."""
    start = start or (SESSIONS if os.path.isdir(SESSIONS) else HERE)
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:                              # headless box, no tk
        print("tkinter unavailable - pass the session directory as an argument.")
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)                # or it hides behind the IDE
    path = filedialog.askdirectory(initialdir=start,
                                   title="Pick a capture session to mosaic")
    root.destroy()
    return path or None


def _show(name, img, cap_h=900):
    if img is None:
        return
    vis = img
    if vis.shape[0] > cap_h:
        s = cap_h / float(vis.shape[0])
        vis = cv2.resize(vis, (int(vis.shape[1] * s), cap_h))
    cv2.imshow(name, vis)


def report(session, save=True):
    """Run the whole pipeline on one session and display every stage."""
    imgs, masks, paths = mosaic.load_session(session)
    name = os.path.basename(session.rstrip("/\\"))
    print(f"\n=== {name}: {len(imgs)} capture(s) ===")
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

    _show("Captures", mosaic.contact_sheet(imgs, cols=4, labels=labels))

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
    session = args[0] if args else None

    while True:
        if session is None:
            session = pick_session()
            if session is None:
                print("Nothing picked.")
                return 0
        if not os.path.isdir(session):
            print(f"Not a directory: {session}")
            return 2

        cv2.destroyAllWindows()
        try:
            report(session, save=save)
        except Exception as e:                       # noqa: BLE001 - a tuning run
            import traceback                         # should show the fault, not die
            print(f"\n[ERROR] {e!r}")
            traceback.print_exc()

        print("\n  any window focused: SPACE/o = pick another session, "
              "r = re-run this one, ESC/q = quit")
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            continue                                 # same session, re-run
        session = None                               # anything else: pick again
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
