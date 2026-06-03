#!/usr/bin/env python3
"""Streamlit dashboard for the pupil-detection lab.

Run from the repo root:
    streamlit run pupillab/app.py

Pick a capture, tick the detector modules to run, tune their parameters with the
auto-generated sliders, and inspect each detector's debug stages and overlay
side-by-side plus a comparison panel.  Everything is driven by the registry, so a
newly added module shows up here automatically (checkbox + sliders + stages).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

# Allow `streamlit run pupillab/app.py` to import the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pupillab import context, montage, registry   # noqa: E402
import guidance                                     # noqa: E402  (repo root on path)

cap = context.cap
cv2 = cap.cv2


def _bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def _build_ctx_cached(folder: str, _sig, use_roi, side_frac, offset_frac, use_cover):
    """Build (and cache) the context for a folder; _sig busts the cache on edits."""
    loaded = context.load_capture(folder)
    if loaded is None:
        return None
    ambient_rgb, flash_rgb, both_rgb, meta = loaded
    ctx = context.build_context(ambient_rgb, flash_rgb, both_rgb, meta,
                                use_roi=use_roi, roi_side_frac=side_frac,
                                roi_offset_frac=offset_frac, use_cover=use_cover)
    return ctx, flash_rgb


def _folder_signature(folder: str):
    """A cheap change-signature so cache invalidates when the files change."""
    parts = []
    for fn in ("ambient.jpg", "flash.jpg", "both.jpg", "meta.json"):
        path = os.path.join(folder, fn)
        parts.append(os.path.getmtime(path) if os.path.isfile(path) else 0)
    return tuple(parts)


def main():
    st.set_page_config(page_title="EyeVu Pupil Lab", layout="wide")
    st.title("EyeVu — Pupil Detection Lab")

    detectors = registry.get_all()
    folders = context.find_capture_folders()

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header("Capture")
    if not folders:
        st.warning(f"No capture folders in {context.TRANSFERS_DIR}")
        st.stop()
    names = [os.path.basename(f) for f in folders]
    sel = st.sidebar.selectbox("Folder", names, index=len(names) - 1)
    folder = folders[names.index(sel)]

    st.sidebar.header("Search ROI")
    st.sidebar.caption("Box above the LED cover that the pupil is assumed to lie in.")
    use_cover = st.sidebar.checkbox("Use LED-cover mask", value=True,
                                    help="Off = ignore the calibrated cover entirely.")
    use_roi = st.sidebar.checkbox("Restrict to ROI", value=context.ROI_ENABLE)
    side_frac = st.sidebar.slider("Box side (× smaller image dim)", 0.3, 1.0,
                                  context.ROI_SIDE_FRAC, 0.05, disabled=not use_roi)
    offset_frac = st.sidebar.slider("Centre above cover top (× side)", -0.2, 0.5,
                                    context.ROI_OFFSET_FRAC, 0.01, disabled=not use_roi)

    st.sidebar.header("Detectors")
    enabled, all_params = [], {}
    for det in detectors:
        on = st.sidebar.checkbox(det.name, value=True, key=f"on_{det.name}")
        if not on:
            continue
        enabled.append(det)
        pv = {}
        if det.params:
            with st.sidebar.expander(f"{det.name} parameters"):
                for spec in det.params:
                    if spec.kind == "int":
                        pv[spec.name] = st.slider(
                            spec.name, int(spec.min), int(spec.max),
                            int(spec.default), int(max(1, spec.step)),
                            help=spec.help, key=f"{det.name}_{spec.name}")
                    else:
                        pv[spec.name] = st.slider(
                            spec.name, float(spec.min), float(spec.max),
                            float(spec.default), float(spec.step),
                            help=spec.help, key=f"{det.name}_{spec.name}")
        all_params[det.name] = pv

    built = _build_ctx_cached(folder, _folder_signature(folder),
                              use_roi, side_frac, offset_frac, use_cover)
    if built is None:
        st.error("Missing ambient.jpg or flash.jpg in this folder.")
        st.stop()
    ctx, flash_rgb = built
    if use_roi and ctx.roi is None:
        st.info("No LED-cover calibration for this capture — ROI disabled, "
                "searching the full frame.")

    # ── Run enabled detectors ────────────────────────────────────────────
    results = []
    for det in enabled:
        with st.spinner(f"running {det.name}…"):
            results.append(det.detect(ctx, all_params.get(det.name, {})))

    # ── Comparison panel ─────────────────────────────────────────────────
    st.subheader("Comparison — pupil overlay on flash")
    cols = st.columns(max(1, len(results)))
    for col, res in zip(cols, results):
        vis = cv2.convertScaleAbs(ctx.flash, alpha=3.0)
        ov = res.overlay()
        if ov is not None:
            cap._draw_overlays(vis, [ov])
        if res.pupil is not None:
            (_, _), (MA, _), _ = res.pupil
            cap_txt = f"r={MA / 2:.0f}px  conf={res.confidence:.2f}  {res.elapsed_ms:.0f}ms"
        elif not res.ok:
            cap_txt = f"unavailable — {res.notes}"
        else:
            cap_txt = f"no pupil — {res.notes}"
        col.image(_bgr2rgb(vis), caption=f"{res.detector}: {cap_txt}",
                  width="stretch")

    # ── Alignment guidance (coarse cascade -> move instruction) ───────────
    st.subheader("Alignment guidance")
    target_mode = st.radio("Target", ["centre", "cover_top_mid"], horizontal=True,
                           key="guide_target")
    loc = cap.coarse_locate(ctx.ambient, ctx.flash, ctx.cover_mask, None,
                            allow_ml=True)
    center = loc[0] if loc else None
    conf = loc[2] if loc else 0.0
    source = loc[3] if loc else ""
    target = guidance.target_point(ctx.flash.shape, ctx.cover_mask, target_mode)
    g = guidance.GuidanceTracker().update(center, ctx.flash.shape, target, conf, source)
    gvis = guidance.annotate(cv2.convertScaleAbs(ctx.flash, alpha=3.0), g, target, center)
    gcol, tcol = st.columns([2, 3])
    gcol.image(_bgr2rgb(gvis), caption=f"{g.state}: {g.instruction}", width="stretch")
    tcol.markdown(f"### {g.instruction}")
    tcol.write(f"state: **{g.state}**")
    if center is not None:
        tcol.write(f"offset: {g.distance:.0f}px  ·  source: {source}  ·  conf: {conf:.2f}")
        tcol.write(f"pupil: ({center[0]:.0f}, {center[1]:.0f})  ·  target: {target}")

    # ── Per-detector debug stages ────────────────────────────────────────
    for res in results:
        st.subheader(f"{res.detector}")
        if res.notes:
            st.caption(res.notes)
        if not res.stages:
            st.caption("(no debug stages)")
            continue
        scols = st.columns(min(4, len(res.stages)))
        for i, (name, img) in enumerate(res.stages):
            scols[i % len(scols)].image(_bgr2rgb(img), caption=name,
                                        width="stretch")

    # ── Export ───────────────────────────────────────────────────────────
    st.sidebar.header("Export")
    if st.sidebar.button("Save debug_montage.jpg to folder"):
        path = montage.save_montage(folder, ctx, results)
        st.sidebar.success(f"Wrote {os.path.relpath(path, context.TRANSFERS_DIR)}")
    if st.sidebar.button("Save annotated.jpg (best result)"):
        best = montage.best_result(results)
        overlays = [best.overlay()] if best and best.overlay() else []
        img = cap.process_image(flash_rgb, overlays)
        img.save(os.path.join(folder, "annotated.jpg"))
        st.sidebar.success(f"Wrote annotated.jpg (best={best.detector if best else 'none'})")


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the module top-level, not under __main__.
    main()
