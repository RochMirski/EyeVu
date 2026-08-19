#!/usr/bin/env python3
"""Unit tests for the match graph: cycle consistency, components, MST, init.

Built entirely from synthetic rotations -- no images, no hardware -- because the
properties under test are structural.  Cycle consistency in particular is the
highest-value filter in the pipeline (it uses evidence from OUTSIDE a pair,
which is the only thing that reliably separates true from false edges on this
material), so it is worth testing hard.

Run:  pytest test_mosaic_graph.py     or     python test_mosaic_graph.py
"""

import numpy as np
import pytest

from eyevu_mosaic.config import MosaicConfig
from eyevu_mosaic.core import graph as GR
from eyevu_mosaic.core import globalopt as GO
from eyevu_mosaic.core import models as M


def _truth(n=6, scale=2.0, seed=0):
    """Ground-truth global rotations G_i (patch frame -> reference frame)."""
    rng = np.random.default_rng(seed)
    return {i: M.so3_exp(np.radians(rng.normal(0, scale, 3))) for i in range(n)}


def _rec(i, j, G, *, inliers=6, ncc=0.8, residual=1.0, perturb=None):
    """A synthetic accepted pair record, consistent with G unless perturbed."""
    R = G[j].T @ G[i]                       # maps frame i -> frame j
    if perturb is not None:
        R = M.so3_exp(np.radians(perturb)) @ R
    return {"pair": [i, j], "accepted": True, "n_inliers": inliers,
            "ncc": ncc, "median_residual": residual, "level": 0,
            "estimate_l0": M.make_estimate(0, R=R, matrix=np.eye(3)),
            "estimate": M.make_estimate(0, R=R, matrix=np.eye(3)),
            "matches": None, "inlier_mask": None}


def _cfg(**kw):
    c = MosaicConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ── construction ──────────────────────────────────────────────────────────

def test_build_creates_both_directions():
    G = _truth(3)
    nodes, edges = GR.build([_rec(0, 1, G), _rec(1, 2, G)])
    assert nodes == {0, 1, 2}
    assert (0, 1) in edges and (1, 0) in edges
    assert np.allclose(edges[(0, 1)]["R"].T, edges[(1, 0)]["R"], atol=1e-12)


def test_build_ignores_rejected_records():
    G = _truth(3)
    r = _rec(0, 1, G)
    r["accepted"] = False
    nodes, edges = GR.build([r, _rec(1, 2, G)])
    assert (0, 1) not in edges
    assert nodes == {1, 2}


def test_edge_weight_prefers_agreement_over_raw_count():
    """Inlier count is the weak signal here; NCC must dominate the weight."""
    many_bad = {"n_inliers": 20, "ncc": 0.10, "median_residual": 5.0}
    few_good = {"n_inliers": 4, "ncc": 0.90, "median_residual": 0.5}
    assert GR.edge_weight(few_good) > GR.edge_weight(many_bad)


# ── cycle consistency ─────────────────────────────────────────────────────

def test_consistent_triangle_survives():
    G = _truth(3)
    nodes, edges = GR.build([_rec(0, 1, G), _rec(1, 2, G), _rec(0, 2, G)])
    kept, rep = GR.cycle_filter(nodes, edges, _cfg())
    assert rep["dropped"] == []
    assert len(rep["triangles"]) == 1
    assert rep["triangles"][0]["error_deg"] < 1e-6
    assert len({tuple(sorted(k)) for k in kept}) == 3


def test_inconsistent_edge_is_detected_and_dropped():
    """A plausible-looking bad edge that only a loop can expose.

    The bad edge passes every per-pair gate -- 5 inliers and NCC 0.55 are both
    above threshold -- so nothing inside `pairwise` could have rejected it.
    Only the triangle does.
    """
    G = _truth(3)
    # A POINTING error: 0.5 deg of rx displaces the image by f*angle ~ 31 px.
    bad = _rec(0, 2, G, inliers=5, ncc=0.55, perturb=[0.5, 0.0, 0.0])
    nodes, edges = GR.build([_rec(0, 1, G, ncc=0.9), _rec(1, 2, G, ncc=0.9), bad])
    kept, rep = GR.cycle_filter(nodes, edges, _cfg())
    assert rep["dropped"], "a 31 px loop error must be caught"
    assert rep["dropped"][0]["edge"] == [0, 2]
    assert (0, 2) not in kept and (2, 0) not in kept
    assert rep["dropped"][0]["error_px"] > 18.0


def test_pixel_metric_weights_pointing_far_above_roll():
    """The whole reason the filter scores pixels rather than degrees.

    The same angular error costs f*angle when it is pointing and only r*angle
    when it is roll -- a ratio of ~50 at this field of view.  Roll is also the
    worst-observed parameter (measured median 2.1 deg on edges independently
    confirmed correct), so a degree-based gate spends its whole budget on the
    one DOF that barely moves the image, and discards good edges.  This test
    pins the trade-off down so it cannot be silently reverted.
    """
    G = _truth(3)

    def closure_px(perturb):
        nodes, edges = GR.build([_rec(0, 1, G), _rec(1, 2, G),
                                 _rec(0, 2, G, perturb=perturb)])
        _k, rep = GR.cycle_filter(nodes, edges, _cfg(cycle_max_error_px=1e9))
        return rep["triangles"][0]["error_px"]

    roll = closure_px([0.0, 0.0, 2.0])
    point = closure_px([2.0, 0.0, 0.0])
    assert point > 10 * roll, (point, roll)
    # ...and that difference is what the default threshold acts on.
    for perturb, expect_drop in (([0.0, 0.0, 2.0], False), ([2.0, 0.0, 0.0], True)):
        nodes, edges = GR.build([_rec(0, 1, G), _rec(1, 2, G),
                                 _rec(0, 2, G, perturb=perturb)])
        _k, rep = GR.cycle_filter(nodes, edges, _cfg())
        assert bool(rep["dropped"]) is expect_drop


def test_the_weakest_edge_of_a_bad_triangle_is_the_one_dropped():
    G = _truth(3)
    # Corrupt 1-2, but give 0-2 the lowest confidence; the filter must still
    # drop by confidence, which is all it can honestly do from one triangle.
    recs = [_rec(0, 1, G, ncc=0.95, inliers=12),
            _rec(1, 2, G, ncc=0.90, inliers=10, perturb=[6.0, 0.0, 0.0]),
            _rec(0, 2, G, ncc=0.30, inliers=3)]
    nodes, edges = GR.build(recs)
    _kept, rep = GR.cycle_filter(nodes, edges, _cfg())
    assert rep["dropped"][0]["edge"] == [0, 2]


def test_small_error_below_threshold_is_kept():
    G = _truth(3)
    recs = [_rec(0, 1, G), _rec(1, 2, G),
            _rec(0, 2, G, perturb=[0.0, 0.0, 1.0])]      # 1 deg < 2 deg gate
    nodes, edges = GR.build(recs)
    _kept, rep = GR.cycle_filter(nodes, edges, _cfg(cycle_max_error_deg=2.0))
    assert rep["dropped"] == []


def test_threshold_is_respected():
    G = _truth(3)
    # 0.3 deg of pointing error ~ 19 px at the default focal.
    recs = [_rec(0, 1, G), _rec(1, 2, G),
            _rec(0, 2, G, perturb=[0.3, 0.0, 0.0])]
    for thr, expect_drop in ((10.0, True), (40.0, False)):
        nodes, edges = GR.build(recs)
        _k, rep = GR.cycle_filter(nodes, edges, _cfg(cycle_max_error_px=thr))
        assert bool(rep["dropped"]) is expect_drop


def test_edges_in_no_triangle_are_flagged_not_dropped():
    """An untested edge is unproven, not disproven -- it must survive."""
    G = _truth(3)
    nodes, edges = GR.build([_rec(0, 1, G), _rec(1, 2, G)])
    kept, rep = GR.cycle_filter(nodes, edges, _cfg())
    assert rep["dropped"] == []
    assert sorted(rep["untested_edges"]) == [[0, 1], [1, 2]]
    assert len({tuple(sorted(k)) for k in kept}) == 2


def test_cycle_check_can_be_disabled():
    G = _truth(3)
    nodes, edges = GR.build([_rec(0, 1, G), _rec(1, 2, G),
                             _rec(0, 2, G, perturb=[0, 0, 20.0])])
    kept, rep = GR.cycle_filter(nodes, edges, _cfg(cycle_check=False))
    assert rep["dropped"] == [] and len({tuple(sorted(k)) for k in kept}) == 3


def test_multiple_bad_edges_are_removed_iteratively():
    G = _truth(5)
    good = [_rec(i, j, G, ncc=0.9, inliers=12)
            for i in range(5) for j in range(i + 1, 5)]
    bad = [_rec(0, 3, G, ncc=0.2, inliers=3, perturb=[10.0, 0, 0]),
           _rec(1, 4, G, ncc=0.2, inliers=3, perturb=[0, 12.0, 0])]
    recs = [r for r in good if r["pair"] not in ([0, 3], [1, 4])] + bad
    nodes, edges = GR.build(recs)
    kept, rep = GR.cycle_filter(nodes, edges, _cfg())
    dropped = {tuple(d["edge"]) for d in rep["dropped"]}
    assert (0, 3) in dropped and (1, 4) in dropped
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        assert (i, j) in kept


# ── components and spanning tree ──────────────────────────────────────────

def test_components_are_found_and_ordered_largest_first():
    G = _truth(6)
    recs = [_rec(0, 1, G), _rec(1, 2, G), _rec(0, 2, G), _rec(3, 4, G)]
    nodes, edges = GR.build(recs)
    comps = GR.components(nodes, edges)
    assert comps == [[0, 1, 2], [3, 4]]
    assert 5 not in nodes                    # a patch with no edge is not a node


def test_mst_spans_and_is_a_tree():
    G = _truth(5)
    recs = [_rec(i, j, G, inliers=3 + i + j)
            for i in range(5) for j in range(i + 1, 5)]
    nodes, edges = GR.build(recs)
    comp = GR.components(nodes, edges)[0]
    tree = GR.max_spanning_tree(comp, edges)
    assert len(tree) == len(comp) - 1
    seen = set()
    for i, j in tree:
        seen.update((i, j))
    assert seen == set(comp)


def test_mst_prefers_confident_edges():
    G = _truth(3)
    recs = [_rec(0, 1, G, ncc=0.95, inliers=20),
            _rec(1, 2, G, ncc=0.95, inliers=20),
            _rec(0, 2, G, ncc=0.20, inliers=3)]
    nodes, edges = GR.build(recs)
    tree = GR.max_spanning_tree(GR.components(nodes, edges)[0], edges)
    assert (0, 2) not in tree and (2, 0) not in tree


# ── initialisation and refinement ─────────────────────────────────────────

def test_initial_rotations_recover_truth_up_to_gauge():
    G = _truth(6, seed=4)
    recs = [_rec(i, j, G) for i in range(6) for j in range(i + 1, 6)]
    nodes, edges = GR.build(recs)
    comp = GR.components(nodes, edges)[0]
    tree = GR.max_spanning_tree(comp, edges)
    est = GR.initial_rotations(comp, edges, tree)
    assert set(est) == set(comp)
    # Only relative rotations are observable; align on one patch and compare.
    root = comp[0]
    A = G[root] @ est[root].T
    for n in comp:
        assert M.rotation_angle_deg(G[n].T @ (A @ est[n])) < 1e-8


def test_initial_rotations_chain_through_a_path_graph():
    """No loops at all: the tree IS the graph, and error must not accumulate."""
    G = _truth(8, seed=6)
    recs = [_rec(i, i + 1, G) for i in range(7)]
    nodes, edges = GR.build(recs)
    comp = GR.components(nodes, edges)[0]
    est = GR.initial_rotations(comp, edges, GR.max_spanning_tree(comp, edges))
    A = G[comp[0]] @ est[comp[0]].T
    for n in comp:
        assert M.rotation_angle_deg(G[n].T @ (A @ est[n])) < 1e-7


def test_rotation_averaging_improves_a_noisy_graph():
    """Averaging must use the redundancy the spanning tree throws away."""
    rng = np.random.default_rng(9)
    G = _truth(7, seed=8)
    recs = []
    for i in range(7):
        for j in range(i + 1, 7):
            recs.append(_rec(i, j, G, perturb=rng.normal(0, 0.6, 3)))
    nodes, edges = GR.build(recs)
    comp = GR.components(nodes, edges)[0]
    tree = GR.max_spanning_tree(comp, edges)
    init = GR.initial_rotations(comp, edges, tree)
    avg, stats = GO.rotation_average(comp, edges, dict(init), iters=40)

    def err(sol):
        A = G[comp[0]] @ sol[comp[0]].T
        return float(np.median([M.rotation_angle_deg(G[n].T @ (A @ sol[n]))
                                for n in comp]))

    assert err(avg) < err(init)
    assert stats["iterations"] >= 1


def test_rotation_averaging_holds_the_gauge_fixed():
    G = _truth(4, seed=10)
    recs = [_rec(i, j, G) for i in range(4) for j in range(i + 1, 4)]
    nodes, edges = GR.build(recs)
    comp = GR.components(nodes, edges)[0]
    init = GR.initial_rotations(comp, edges, GR.max_spanning_tree(comp, edges))
    root = min(init, key=lambda n: (np.linalg.norm(M.so3_log(init[n])), n))
    out, _ = GO.rotation_average(comp, edges, dict(init), iters=20)
    assert np.allclose(out[root], init[root], atol=1e-12)


# ── tracks ────────────────────────────────────────────────────────────────

def test_tracks_merge_across_pairs():
    """Keypoint 0 of patch 0 == kp 1 of patch 1 == kp 2 of patch 2."""
    cfg = _cfg()
    recs = [
        {"pair": [0, 1], "accepted": True,
         "matches": np.array([[0, 1]]), "inlier_mask": np.array([True])},
        {"pair": [1, 2], "accepted": True,
         "matches": np.array([[1, 2]]), "inlier_mask": np.array([True])},
    ]
    tracks, stats = GO.build_tracks(recs, cfg)
    assert stats["n_tracks"] == 1
    assert tracks[0] == [(0, 0), (1, 1), (2, 2)]


def test_track_with_two_observations_in_one_patch_is_discarded():
    """One point cannot be in two places in one image: the track is unusable."""
    cfg = _cfg()
    recs = [
        {"pair": [0, 1], "accepted": True,
         "matches": np.array([[0, 1]]), "inlier_mask": np.array([True])},
        {"pair": [0, 1], "accepted": True,
         "matches": np.array([[5, 1]]), "inlier_mask": np.array([True])},
    ]
    tracks, stats = GO.build_tracks(recs, cfg)
    assert stats["n_tracks"] == 0
    assert stats["n_conflicting_tracks"] == 1


def test_outlier_correspondences_do_not_enter_tracks():
    cfg = _cfg()
    recs = [{"pair": [0, 1], "accepted": True,
             "matches": np.array([[0, 1], [2, 3]]),
             "inlier_mask": np.array([True, False])}]
    tracks, stats = GO.build_tracks(recs, cfg)
    assert stats["n_tracks"] == 1 and tracks[0] == [(0, 0), (1, 1)]


def test_landmark_priors_are_excluded_from_tracks():
    """Landmark rows carry index -1 and are not real keypoints."""
    cfg = _cfg()
    recs = [{"pair": [0, 1], "accepted": True,
             "matches": np.array([[-1, -1], [0, 1]]),
             "inlier_mask": np.array([True, True])}]
    tracks, stats = GO.build_tracks(recs, cfg)
    assert stats["n_tracks"] == 1 and tracks[0] == [(0, 0), (1, 1)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
