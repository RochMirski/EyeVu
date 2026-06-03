#!/usr/bin/env python3
"""Headless batch runner — run every registered detector over capture folders.

Same registry as the Streamlit dashboard, no browser needed.  For each capture it
builds the shared context once, runs all detectors with their default parameters,
writes a grouped ``debug_montage.jpg`` (one labelled block per detector + a
comparison panel), and writes ``annotated.jpg`` from the best result.

Usage:
    python pupillab/run_batch.py                     # all Transfers/capture_* folders
    python pupillab/run_batch.py capture_20260530_145755   # one folder (name or path)
    python pupillab/run_batch.py --only ridge_baseline,ritnet   # subset of detectors
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python pupillab/run_batch.py` to import the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pupillab import context, montage, registry   # noqa: E402

cap = context.cap


def run_folder(folder: str, detectors, roi_opts) -> int:
    """Run `detectors` over one capture folder; write montage + annotated.jpg."""
    loaded = context.load_capture(folder)
    name = os.path.basename(os.path.normpath(folder))
    if loaded is None:
        print(f"  SKIP {name} — missing ambient.jpg/flash.jpg")
        return -1
    ambient_rgb, flash_rgb, both_rgb, meta = loaded
    ctx = context.build_context(ambient_rgb, flash_rgb, both_rgb, meta, **roi_opts)

    results = [det.detect(ctx, det.default_params()) for det in detectors]
    montage.save_montage(folder, ctx, results)

    # annotated.jpg from the best (highest-confidence) pupil.
    best = montage.best_result(results)
    overlays = [best.overlay()] if best and best.overlay() else []
    img = cap.process_image(flash_rgb, overlays)
    img.save(os.path.join(folder, "annotated.jpg"))

    summary = ", ".join(
        f"{r.detector}:{'pupil' if r.pupil is not None else ('n/a' if not r.ok else 'none')}"
        for r in results)
    pick = best.detector if best else "none"
    print(f"  {name}: [{summary}]  -> best={pick}")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", help="single capture folder (name or path)")
    ap.add_argument("--only", help="comma-separated detector names to run (default: all)")
    ap.add_argument("--no-roi", action="store_true",
                    help="search the full frame instead of the box above the LED cover")
    ap.add_argument("--roi-side", type=float, default=context.ROI_SIDE_FRAC,
                    help="ROI box side as a fraction of the smaller image dimension")
    ap.add_argument("--roi-offset", type=float, default=context.ROI_OFFSET_FRAC,
                    help="ROI centre above the cover top, as a fraction of the box side")
    args = ap.parse_args()
    roi_opts = dict(use_roi=not args.no_roi, roi_side_frac=args.roi_side,
                    roi_offset_frac=args.roi_offset)

    if not cap.CV2_AVAILABLE:
        print("ERROR: cv2 not available — pip install -r requirements.txt")
        sys.exit(1)

    all_dets = registry.get_all()
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        detectors = [d for d in all_dets if d.name in wanted]
        if not detectors:
            print(f"ERROR: no detectors matched {sorted(wanted)}; "
                  f"available: {[d.name for d in all_dets]}")
            sys.exit(1)
    else:
        detectors = all_dets

    print(f"Detectors: {[d.name for d in detectors]}\n")

    if args.folder:
        folder = (args.folder if os.path.isdir(args.folder)
                  else os.path.join(context.TRANSFERS_DIR, args.folder))
        if not os.path.isdir(folder):
            print(f"ERROR: capture folder not found: {args.folder}")
            sys.exit(1)
        folders = [folder]
    else:
        folders = context.find_capture_folders()

    if not folders:
        print(f"No capture folders found in {context.TRANSFERS_DIR}")
        return

    print(f"Processing {len(folders)} folder(s):")
    done = sum(1 for f in folders if run_folder(f, detectors, roi_opts) >= 0)
    print(f"\nDone — {done}/{len(folders)} processed. "
          f"Open each debug_montage.jpg / annotated.jpg to inspect.")


if __name__ == "__main__":
    main()
