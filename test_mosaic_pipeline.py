#!/usr/bin/env python3
"""End-to-end tests for eyevu_mosaic: preprocessing, bundle contract, run.

These use the synthetic session generator, so they need no hardware and no real
captures -- but they exercise the same code path `run_session` runs on the Pi,
including the bundle write/read round-trip that the dashboard depends on.

Run:  pytest test_mosaic_pipeline.py     or     python test_mosaic_pipeline.py
"""

import json
import os

import numpy as np
import pytest

from eyevu_mosaic.config import MosaicConfig
from eyevu_mosaic.core import bundle as bundle_io
from eyevu_mosaic.core import compose, models as M, preprocess
from eyevu_mosaic import synth
from eyevu_mosaic.run_session import load_session, run


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    truth = synth.generate(str(d), n=8, seed=3)
    return str(d), truth


@pytest.fixture(scope="module")
def done(session, tmp_path_factory):
    sdir, truth = session
    out = str(tmp_path_factory.mktemp("bundle"))
    res = run(sdir, out, MosaicConfig(), verbose=False)
    return sdir, truth, out, res


# ── preprocessing ─────────────────────────────────────────────────────────

def test_analysis_scale_depends_only_on_resolution():
    """The bug this replaced: per-patch bbox scaling invents relative scale.

    Two patches of the same eye at the same capture resolution must land at the
    same angular sampling however different their own blobs are, or a 3-DOF
    rotation model -- which has no scale freedom -- simply cannot fit them.
    """
    cfg = MosaicConfig()
    assert (preprocess.analysis_scale(480, cfg)
            == preprocess.analysis_scale(480, cfg))
    # A 960-wide frame is already 2x oversampled, so it is scaled half as much.
    assert np.isclose(preprocess.analysis_scale(960, cfg),
                      preprocess.analysis_scale(480, cfg) / 2.0)


def test_mixed_resolution_patches_share_one_angular_scale(session):
    """Real sessions mix 480x640 and 960x1280 captures."""
    import cv2
    sdir, _ = session
    cfg = MosaicConfig()
    path, img, mask = load_session(sdir)[0]
    big = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)
    bigm = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2),
                      interpolation=cv2.INTER_NEAREST)
    a = preprocess.prepare(0, img, mask, cfg)
    b = preprocess.prepare(1, big, bigm, cfg)
    # Same retina, twice the pixels: the analysis images must come out the same
    # size to within rounding, and the focal lengths must match.
    assert abs(a.image.shape[0] - b.image.shape[0]) <= 3
    assert abs(a.image.shape[1] - b.image.shape[1]) <= 3
    assert np.isclose(a.focal, b.focal, rtol=1e-6)


def test_quality_gate_records_reasons_and_never_silently_drops():
    cfg = MosaicConfig()
    blank = np.zeros((640, 480, 3), np.uint8)
    mask = np.zeros((640, 480), np.uint8)
    p = preprocess.prepare(0, blank, mask, cfg)
    assert p.accepted is False
    assert p.reason and "mask" in p.reason
    assert "mask_area_px" in p.quality


def test_tiny_patch_is_rejected_with_a_reason():
    """The real sessions contain a 7x9 px, 49 px-area capture."""
    import cv2
    cfg = MosaicConfig()
    img = np.zeros((640, 480, 3), np.uint8)
    mask = np.zeros((640, 480), np.uint8)
    cv2.circle(mask, (240, 320), 4, 255, -1)
    img[mask > 0] = 90
    p = preprocess.prepare(0, img, mask, cfg)
    assert not p.accepted and "too small" in p.reason


def test_equivalent_radius_matches_area():
    m = np.zeros((100, 100), np.uint8)
    import cv2
    cv2.circle(m, (50, 50), 20, 255, -1)
    assert abs(preprocess.equivalent_radius(m) - 20.0) < 0.6


def test_polynomial_flattening_removes_a_vignette():
    """The reason 'poly' is the default and Gaussian high-pass is not.

    A quadratic illumination falloff must come out, and the texture under it
    must survive -- at this patch size a high-pass cannot do both.
    """
    import cv2
    rng = np.random.default_rng(0)
    h = w = 160
    tex = cv2.GaussianBlur(rng.normal(120, 20, (h, w)).astype(np.float32), (0, 0), 3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2
    vign = np.clip(1.0 - 0.7 * d2, 0.15, 1.0)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 60, 255, -1)

    cfg = MosaicConfig()
    cfg.clahe_clip = 0.0
    flat = preprocess.flatten_illumination(
        np.clip(tex * vign, 0, 255).astype(np.uint8), mask, 60.0, cfg)
    sel = mask > 0
    # Correlation with the true texture must beat correlation with the vignette.
    def corr(a, b):
        a = a[sel].astype(np.float64) - a[sel].mean()
        b = b[sel].astype(np.float64) - b[sel].mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    assert corr(flat, tex) > 0.9
    assert abs(corr(flat, vign)) < 0.5


# ── projection ────────────────────────────────────────────────────────────

def test_projection_roundtrip():
    proj = compose.Projection(np.array([0.0, 0.1, 0.99]), 3600.0)
    rng = np.random.default_rng(1)
    xy = rng.uniform(-300, 300, (50, 2))
    assert np.allclose(proj.from_dir(proj.to_dir(xy)), xy, atol=1e-6)


def test_azimuthal_equidistant_is_equidistant():
    """Radius on the canvas must be proportional to arc length on the sphere."""
    proj = compose.Projection(np.array([0.0, 0.0, 1.0]), 1000.0)
    for deg in (1.0, 5.0, 20.0, 45.0):
        th = np.radians(deg)
        d = np.array([[np.sin(th), 0.0, np.cos(th)]])
        r = np.linalg.norm(proj.from_dir(d)[0])
        assert np.isclose(r, 1000.0 * th, rtol=1e-6)


def test_gnomonic_differs_and_diverges():
    az = compose.Projection(np.array([0.0, 0.0, 1.0]), 1000.0)
    gn = compose.Projection(np.array([0.0, 0.0, 1.0]), 1000.0, "gnomonic")
    th = np.radians(45.0)
    d = np.array([[np.sin(th), 0.0, np.cos(th)]])
    assert np.linalg.norm(gn.from_dir(d)[0]) > np.linalg.norm(az.from_dir(d)[0])


def test_mean_direction_of_identity_is_the_axis():
    assert np.allclose(compose.mean_direction({0: np.eye(3)}),
                       [0.0, 0.0, 1.0], atol=1e-12)


def test_patch_outlines_land_where_the_patches_land(done):
    """An outline must enclose the pixels that patch actually contributed.

    This is the check that makes the overlay trustworthy as a diagnostic: if the
    polygons were merely plausible rather than correct, they would mislead
    exactly when you most need them — when a group looks wrong.
    """
    import cv2
    _s, _t, out, _r = done
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    comp0 = body["compose"]["components"][0]
    outlines = comp0["outlines"]
    assert sorted(int(k) for k in outlines) == sorted(comp0["patches"])

    cov = cv2.imread(os.path.join(out, "coverage.png"), 0)
    filled = np.zeros_like(cov)
    for poly in outlines.values():
        cv2.fillPoly(filled, [np.array(poly, np.int32)], 255)
    covered = cov > 0
    inside = (covered & (filled > 0)).sum() / max(1, covered.sum())
    assert inside > 0.9, f"only {inside:.0%} of covered pixels lie in an outline"


def test_outlines_are_non_degenerate(done):
    import cv2
    _s, _t, out, _r = done
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    for comp in body["compose"]["components"]:
        for k, poly in comp["outlines"].items():
            p = np.array(poly, np.int32)
            assert len(p) >= 3, k
            assert cv2.contourArea(p) > 0, k


def test_draw_outlines_changes_the_image_without_resizing_it():
    """Toggling must not move the picture, or comparison is useless."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    poly = np.array([[10, 10], [60, 12], [58, 45], [12, 44]], np.int32)
    out = compose.draw_outlines(img, {3: poly}, dim=0.3)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert out is not img and not np.array_equal(img, out)


def test_outlined_mosaic_is_written(done):
    _s, _t, out, _r = done
    assert os.path.isfile(os.path.join(out, "mosaic_outlined.png"))


def test_patch_palette_is_distinct():
    cols = compose.patch_palette(8)
    assert len(cols) == 8 and len(set(cols)) == 8


def test_weighted_median_ignores_an_outlier_contribution():
    """Why median compositing: one bad patch must not reach the output."""
    vals = np.zeros((5, 4, 4, 3), np.float32)
    vals[:4] = 100.0
    vals[4] = 255.0                       # a blown-out specular contribution
    wts = np.ones((5, 4, 4), np.float32)
    out = compose._weighted_median(vals, wts)
    assert np.allclose(out, 100.0)


# ── the run ───────────────────────────────────────────────────────────────

def test_run_produces_a_bundle_with_every_required_file(done):
    _sdir, _truth, out, _res = done
    for name in ("meta.json", "bundle.json", "bundle.npz", "log.txt",
                 "mosaic.png", "mosaic_preview.jpg", "coverage.png"):
        assert os.path.isfile(os.path.join(out, name)), name
    assert os.path.isdir(os.path.join(out, "raw"))


def test_run_records_every_patch_accepted_or_not(done):
    _sdir, truth, out, _res = done
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    assert len(body["patches"]) == truth["n"]
    for p in body["patches"]:
        assert "accepted" in p and "quality" in p
        if not p["accepted"]:
            assert p["reason"], "a rejected patch must always carry a reason"


def test_run_records_rejected_pairs_with_reasons(done):
    _s, _t, out, _r = done
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    assert body["pairs"], "pair records must be kept, accepted or not"
    for p in body["pairs"]:
        if not p["accepted"]:
            assert p["reason"]


def test_config_is_serialised_into_the_bundle(done):
    _s, _t, out, _r = done
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    cfg = MosaicConfig.from_dict(meta["config"])
    assert cfg.to_dict() == MosaicConfig().to_dict()


def test_timings_and_rss_are_recorded(done):
    _s, _t, out, res = done
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    for stage in ("load", "preprocess", "features", "match", "graph"):
        assert stage in body["timings"]
        assert body["timings"][stage]["seconds"] >= 0
    assert "_total" in res["timings"]


def test_run_places_most_patches(done):
    _s, truth, _o, res = done
    assert res["n_accepted"] == truth["n"]
    assert max(len(c) for c in res["components"]) >= 4


# ── the bundle contract ───────────────────────────────────────────────────

def test_bundle_roundtrip_restores_everything_matching_needs(done):
    """The bundle must support re-running matching WITHOUT the raw images."""
    _s, _t, out, _r = done
    b = bundle_io.read(out)
    assert b["patches"] and b["features"]
    for p in b["patches"]:
        if p.accepted:
            # Pixels are needed because acceptance here is photometric.
            assert p.image is not None and p.mask is not None
            assert p.native_bgr is not None and p.native_weight is not None
            assert p.focal > 0
    got = [f for f in b["features"] if len(f)]
    assert got and all(f.desc is not None for f in got)
    assert all(f.desc.shape[1] == 128 for f in got)


def test_bundle_roundtrip_restores_estimates_and_rotations(done):
    _s, _t, out, _r = done
    b = bundle_io.read(out)
    assert b["rotations"]
    for R in b["rotations"].values():
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    acc = [r for r in b["records"] if r["accepted"]]
    assert acc
    for r in acc:
        est = r["estimate_l0"]
        assert est is not None and est["rotvec"] is not None
        assert M.so3_exp(est["rotvec"]).shape == (3, 3)


def test_matching_can_be_rerun_offline_from_the_bundle(done):
    """The headline requirement: reproduce a pair decision with no raw images."""
    from eyevu_mosaic.core import pairwise
    _s, _t, out, _r = done
    b = bundle_io.read(out)
    cfg = b["cfg"]
    pats = {p.index: p for p in b["patches"]}
    feats = {f.index: f for f in b["features"]}
    acc = [r for r in b["records"] if r["accepted"] and r.get("source") == "features"]
    if not acc:
        pytest.skip("no feature-derived edge in this session")
    r = acc[0]
    i, j = r["pair"]
    again = pairwise.match_pair(pats[i], pats[j], feats[i], feats[j], cfg)
    assert again["accepted"]
    assert abs(again["ncc"] - r["ncc"]) < 0.05


def test_descriptors_are_stored_compactly(done):
    _s, _t, out, _r = done
    z = np.load(os.path.join(out, "bundle.npz"))
    keys = [k for k in z.files if k.startswith("desc_")]
    assert keys
    assert z[keys[0]].dtype == np.float16


def test_bundle_json_is_pure_json(done):
    """No numpy types may leak in -- the dashboard parses this directly."""
    _s, _t, out, _r = done
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        json.load(fh)
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as fh:
        json.load(fh)


# ── failure handling ──────────────────────────────────────────────────────

# ── combining sittings ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def two_sessions(tmp_path_factory):
    """Two sittings of the SAME eye, aimed at overlapping regions.

    Same `texture_seed` so the retina is identical; different `seed` so the
    gaze positions, blob shapes and exposures differ as they would between
    sittings; `centre_deg` shifts the second sitting's coverage so it overlaps
    the first rather than repeating it.
    """
    a = tmp_path_factory.mktemp("sitting_a")
    b = tmp_path_factory.mktemp("sitting_b")
    ta = synth.generate(str(a), n=6, seed=11, texture_seed=99)
    tb = synth.generate(str(b), n=6, seed=12, texture_seed=99,
                        centre_deg=(0.9, 0.0))
    return str(a), str(b), ta, tb


def test_load_sessions_pools_captures_in_order(two_sessions):
    from eyevu_mosaic.run_session import load_session, load_sessions, session_of
    a, b, _ta, _tb = two_sessions
    one, two = load_session(a), load_session(b)
    both = load_sessions([a, b])
    assert len(both) == len(one) + len(two)
    # Order is: all of the first session, then all of the second.
    assert [p for p, _, _ in both] == ([p for p, _, _ in one]
                                       + [p for p, _, _ in two])
    assert session_of(both[0][0]) == os.path.basename(a)
    assert session_of(both[-1][0]) == os.path.basename(b)


def test_combined_run_records_provenance_per_patch(two_sessions, tmp_path):
    a, b, _ta, _tb = two_sessions
    out = str(tmp_path / "combined")
    res = run([a, b], out, MosaicConfig(), verbose=False)
    assert res["n_patches"] == 12
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    names = {os.path.basename(a), os.path.basename(b)}
    assert {p["session"] for p in body["patches"]} == names
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as fh:
        assert set(json.load(fh)["sessions"]) == names


def test_combined_run_splits_raw_by_session(two_sessions, tmp_path):
    """Capture filenames repeat across sittings, so raw/ must not flatten."""
    a, b, _ta, _tb = two_sessions
    out = str(tmp_path / "combined_raw")
    run([a, b], out, MosaicConfig(), verbose=False)
    raw = os.path.join(out, "raw")
    subs = sorted(d for d in os.listdir(raw)
                  if os.path.isdir(os.path.join(raw, d)))
    assert subs == sorted([os.path.basename(a), os.path.basename(b)])
    # Nothing was lost to a name collision.
    total = sum(len(os.listdir(os.path.join(raw, s))) for s in subs)
    assert total == 12 * 2                       # extract + mask per capture


def test_single_session_run_is_unchanged_by_the_list_support(session, tmp_path):
    """Passing a bare string must behave exactly as before."""
    sdir, _truth = session
    one = run(sdir, str(tmp_path / "as_str"), MosaicConfig(), verbose=False)
    lst = run([sdir], str(tmp_path / "as_list"), MosaicConfig(), verbose=False)
    assert one["n_patches"] == lst["n_patches"]
    assert one["components"] == lst["components"]
    # ...and raw/ stays flat for a single session.
    assert not [d for d in os.listdir(os.path.join(str(tmp_path / "as_str"), "raw"))
                if os.path.isdir(os.path.join(
                    str(tmp_path / "as_str"), "raw", d))]


def test_combining_finds_cross_session_overlaps(two_sessions, tmp_path):
    """The reason to combine at all: sittings must be able to link.

    If no pair ever spans two sessions, combining is pointless -- it would just
    be two independent runs sharing a bundle.
    """
    a, b, _ta, _tb = two_sessions
    out = str(tmp_path / "cross")
    run([a, b], out, MosaicConfig(), verbose=False)
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        body = json.load(fh)
    sess = {p["index"]: p["session"] for p in body["patches"]}
    cross = [p for p in body["pairs"] if p["accepted"]
             and sess[p["pair"][0]] != sess[p["pair"][1]]]
    assert cross, "no pair linked the two sittings"


def test_bundle_adjust_degrades_without_scipy(done, monkeypatch):
    """scipy may genuinely be absent on the Pi -- the installer's pip block for
    it is commented out -- so its absence must cost one stage, not the run."""
    from eyevu_mosaic.core import globalopt
    monkeypatch.setattr(globalopt, "SCIPY_AVAILABLE", False)
    _s, _t, out, _r = done
    b = bundle_io.read(out)
    G, focal, stats = globalopt.bundle_adjust(
        b["patches"], b["features"], b["tracks"],
        {n: R for n, R in list(b["rotations"].items())[:3]}, b["cfg"])
    assert stats["ran"] is False
    assert "scipy" in stats["reason"]
    assert focal is None and G is not None


def test_rotation_averaging_needs_no_scipy():
    """The fallback path must be pure numpy, or the fallback is not one."""
    import ast
    import inspect
    from eyevu_mosaic.core import globalopt
    src = inspect.getsource(globalopt.rotation_average)
    assert "scipy" not in src and "optimize" not in src
    # And the module must import even when scipy does not.
    tree = ast.parse(inspect.getsource(globalopt))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in top:
        names = ([a.name for a in node.names]
                 + ([node.module] if isinstance(node, ast.ImportFrom) else []))
        assert not any((n or "").startswith("scipy") for n in names), \
            "scipy must not be imported unguarded at module level"


def test_empty_session_still_writes_a_bundle(tmp_path):
    """The pipeline must never crash the device; it must leave evidence."""
    out = str(tmp_path / "b")
    res = run(str(tmp_path), out, MosaicConfig(), verbose=False)
    assert res["ok"] is False
    assert res["error"] and "no captures" in res["error"]
    assert os.path.isfile(os.path.join(out, "bundle.json"))
    with open(os.path.join(out, "bundle.json"), encoding="utf-8") as fh:
        assert json.load(fh)["error"]


def test_single_capture_session_fails_cleanly(tmp_path):
    import cv2
    d = tmp_path / "s"
    d.mkdir()
    img = np.zeros((640, 480, 3), np.uint8)
    mask = np.zeros((640, 480), np.uint8)
    cv2.circle(mask, (240, 320), 20, 255, -1)
    img[mask > 0] = 100
    cv2.imwrite(str(d / "redeye_01_000000_extract.png"), img)
    cv2.imwrite(str(d / "redeye_01_000000_mask.png"), mask)
    res = run(str(d), str(tmp_path / "b"), MosaicConfig(), verbose=False)
    assert res["ok"] is False and "need >= 2" in res["error"]
    assert os.path.isfile(os.path.join(str(tmp_path / "b"), "log.txt"))


def test_synthetic_ground_truth_is_self_consistent(session):
    """The generator must place patches where it says it does.

    If this fails, every accuracy number measured against it is meaningless.
    """
    import cv2
    sdir, truth = session
    cfg = MosaicConfig()
    loaded = load_session(sdir)
    GT = {p["index"]: np.asarray(p["rotation"]) for p in truth["patches"]}
    ps = [preprocess.prepare(i, im, m, cfg) for i, (_p, im, m) in enumerate(loaded)]
    best = 0.0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            H = M.H_from_rotation(GT[j].T @ GT[i],
                                  preprocess.patch_K(ps[i]),
                                  preprocess.patch_K(ps[j]))
            ha, wa = ps[i].image.shape[:2]
            wm = cv2.warpPerspective(ps[j].mask, np.linalg.inv(H), (wa, ha),
                                     flags=cv2.INTER_NEAREST)
            if int(((wm > 0) & (ps[i].mask > 0)).sum()) < 3000:
                continue
            from eyevu_mosaic.core.pairwise import ncc_verify
            _ok, ncc, _ov = ncc_verify(ps[i], ps[j], H, cfg)
            best = max(best, ncc)
    assert best > 0.75, f"ground-truth geometry does not align patches ({best:.2f})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
