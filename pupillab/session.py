#!/usr/bin/env python3
"""Offline alignment-guidance replay over an ordered capture session.

Replays a sequence of captures the way the device would see them live: builds the
context, runs the same `coarse_locate` cascade (ridge + red-eye + RITnet) seeded by
the previous frame's centre, computes the operator instruction with a stateful
`GuidanceTracker`, and writes a `guidance.jpg` per folder plus a
`session_guidance.jpg` contact sheet.  This is the tuning surface for the new
ordered capture set (the existing Transfers/ are mixed sessions).

Usage:
    python pupillab/session.py                       # all Transfers/capture_* in order
    python pupillab/session.py path/to/session_dir   # capture_* under a session dir
    python pupillab/session.py capture_a capture_b    # specific folders, in order
    python pupillab/session.py --no-cover --target cover_top_mid
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pupillab import context                      # noqa: E402
import guidance                                   # noqa: E402

cap = context.cap
cv2 = cap.cv2


def _resolve_folders(args_folders):
    """Turn CLI args into an ordered list of capture folders."""
    if not args_folders:
        return context.find_capture_folders()
    folders = []
    for a in args_folders:
        path = a if os.path.isdir(a) else os.path.join(context.TRANSFERS_DIR, a)
        if not os.path.isdir(path):
            print(f"  not found: {a}")
            continue
        if os.path.isfile(os.path.join(path, "ambient.jpg")):
            folders.append(path)                  # a capture folder itself
        else:                                     # a session dir of capture_* folders
            for e in sorted(os.listdir(path)):
                sub = os.path.join(path, e)
                if e.startswith("capture_") and os.path.isdir(sub):
                    folders.append(sub)
    return folders


def _contact_sheet(cells, cell_w=360, cols=4):
    """Grid of labelled (name, BGR) cells -> one BGR image."""
    if not cells:
        return None
    tiles = []
    for name, img in cells:
        h, w = img.shape[:2]
        scaled = cv2.resize(img, (cell_w, max(1, int(h * cell_w / w))))
        bar = np.zeros((20, cell_w, 3), np.uint8)
        cv2.putText(bar, name, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, scaled]))
    row_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, row_h - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def replay(folders, target_mode="centre", use_cover=True, allow_ml=True):
    tracker = guidance.GuidanceTracker()
    prior = None
    cells = []
    for folder in folders:
        name = os.path.basename(os.path.normpath(folder))
        loaded = context.load_capture(folder)
        if loaded is None:
            print(f"  SKIP {name} (missing ambient/flash)")
            continue
        amb, fla, both, meta = loaded
        ctx = context.build_context(amb, fla, both, meta, use_cover=use_cover)

        res = cap.coarse_locate(ctx.ambient, ctx.flash, ctx.cover_mask, prior,
                                allow_ml=allow_ml)
        center = res[0] if res else None
        conf = res[2] if res else 0.0
        source = res[3] if res else ""

        target = guidance.target_point(ctx.flash.shape, ctx.cover_mask, target_mode)
        g = tracker.update(center, ctx.flash.shape, target, conf, source)
        prior = center if center is not None else prior

        vis = guidance.annotate(cv2.convertScaleAbs(ctx.flash, alpha=3.0), g, target,
                                center)
        cv2.imwrite(os.path.join(folder, "guidance.jpg"), vis)
        cells.append((name, vis))
        off = f"  off={g.distance:.0f}px {source} conf={conf:.2f}" if center else ""
        print(f"  {name}: {g.instruction}{off}")

    sheet = _contact_sheet(cells)
    if sheet is not None:
        out = os.path.join(context.TRANSFERS_DIR, "session_guidance.jpg")
        cv2.imwrite(out, sheet)
        print(f"\nContact sheet -> {os.path.relpath(out, context.TRANSFERS_DIR)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*", help="session dir, or capture folders in order")
    ap.add_argument("--no-cover", action="store_true", help="disable cover masking")
    ap.add_argument("--no-ml", action="store_true", help="skip RITnet (ridge+redeye only)")
    ap.add_argument("--target", default="centre", choices=["centre", "cover_top_mid"])
    args = ap.parse_args()

    if not cap.CV2_AVAILABLE:
        print("ERROR: cv2 not available.")
        sys.exit(1)
    folders = _resolve_folders(args.folders)
    if not folders:
        print("No capture folders found.")
        return
    print(f"Replaying {len(folders)} capture(s) as a session "
          f"(cover={'off' if args.no_cover else 'on'}, target={args.target}):")
    replay(folders, target_mode=args.target, use_cover=not args.no_cover,
           allow_ml=not args.no_ml)


if __name__ == "__main__":
    main()
