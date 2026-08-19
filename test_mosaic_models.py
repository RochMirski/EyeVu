#!/usr/bin/env python3
"""Unit tests for eyevu_mosaic transform models (no hardware, no images).

Everything here runs on synthetic rotations, so it is fast and portable: the
properties being checked -- round-trip, composition, gauge invariance -- are
algebraic, and if they hold the geometry plumbing in the rest of the pipeline
is trustworthy whatever the imagery does.

Run:  pytest test_mosaic_models.py     or     python test_mosaic_models.py
"""

import numpy as np
import pytest

from eyevu_mosaic.core import models as M


def _K(f=6000.0, cx=120.0, cy=100.0):
    return M.make_K(f, cx, cy)


def _rots(n=12, scale=3.0, seed=0):
    """A spread of small rotations, of the magnitude an eye actually makes."""
    rng = np.random.default_rng(seed)
    return [M.so3_exp(np.radians(rng.normal(0, scale, 3))) for _ in range(n)]


def _pts(n=30, seed=1, span=200.0, org=(20.0, 20.0)):
    rng = np.random.default_rng(seed)
    return np.stack([org[0] + rng.uniform(0, span, n),
                     org[1] + rng.uniform(0, span, n)], axis=1)


# ── SO(3) round-trip ──────────────────────────────────────────────────────

@pytest.mark.parametrize("deg", [0.0, 1e-7, 0.5, 5.0, 45.0, 90.0, 179.0])
def test_so3_exp_log_roundtrip(deg):
    axis = np.array([0.3, -0.7, 0.65])
    axis /= np.linalg.norm(axis)
    w = axis * np.radians(deg)
    R = M.so3_exp(w)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert np.allclose(M.so3_log(R), w, atol=1e-7)


def test_so3_log_exp_roundtrip_random():
    for R in _rots(30, scale=20.0, seed=3):
        assert np.allclose(M.so3_exp(M.so3_log(R)), R, atol=1e-9)


def test_so3_exp_is_smooth_through_zero():
    """The optimiser evaluates near the identity; exp must not blow up there."""
    for s in (1e-12, 1e-9, 1e-6, 1e-3):
        R = M.so3_exp(np.array([s, -s, s]))
        assert np.all(np.isfinite(R))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_rotation_angle_matches_construction():
    for deg in (0.5, 3.0, 17.0, 88.0):
        R = M.so3_exp(np.array([0.0, 0.0, np.radians(deg)]))
        assert np.isclose(M.rotation_angle_deg(R), deg, atol=1e-6)


# ── the conjugate-rotation model ──────────────────────────────────────────

def test_rotation_recovered_exactly_from_two_correspondences():
    """The whole design rests on this: 3 DOF, 2 correspondences, no iteration."""
    K = _K()
    for R in _rots(10, seed=5):
        src = _pts(2, seed=7)
        dst = M.apply_H(M.H_from_rotation(R, K, K), src)
        got = M.fit_rotation(src, dst, K, K)
        assert got is not None
        assert M.rotation_angle_deg(R.T @ got) < 1e-6


def test_rotation_overconstrained_is_consistent():
    K = _K()
    R = M.so3_exp(np.radians([1.2, -0.8, 2.5]))
    src = _pts(25, seed=11)
    dst = M.apply_H(M.H_from_rotation(R, K, K), src)
    got = M.fit_rotation(src, dst, K, K)
    assert M.rotation_angle_deg(R.T @ got) < 1e-6
    assert M.transfer_error(M.H_from_rotation(got, K, K), src, dst).max() < 1e-6


def test_rotation_fit_degrades_gracefully_with_noise():
    K = _K()
    R = M.so3_exp(np.radians([1.0, 0.5, -1.5]))
    src = _pts(40, seed=13)
    dst = M.apply_H(M.H_from_rotation(R, K, K), src)
    rng = np.random.default_rng(2)
    got = M.fit_rotation(src, dst + rng.normal(0, 1.0, dst.shape), K, K)
    # 1 px of noise on a 6000 px focal is ~0.01 deg of rotation.
    assert M.rotation_angle_deg(R.T @ got) < 0.2


def test_rotation_fit_rejects_degenerate_input():
    K = _K()
    assert M.fit_rotation(np.zeros((1, 2)), np.zeros((1, 2)), K, K) is None
    assert M.fit_rotation(np.zeros((0, 2)), np.zeros((0, 2)), K, K) is None


def test_different_intrinsics_per_patch():
    """Sessions really do mix 480x640 and 960x1280 captures."""
    Ka, Kb = M.make_K(6000.0, 120.0, 100.0), M.make_K(6000.0, 90.0, 140.0)
    R = M.so3_exp(np.radians([0.9, -1.1, 0.4]))
    src = _pts(6, seed=17)
    dst = M.apply_H(M.H_from_rotation(R, Ka, Kb), src)
    got = M.fit_rotation(src, dst, Ka, Kb)
    assert M.rotation_angle_deg(R.T @ got) < 1e-6


# ── composition ───────────────────────────────────────────────────────────

def test_conjugation_is_a_homomorphism():
    """H(R1 R2) == H(R1) H(R2): what makes chaining through the graph valid."""
    K = _K()
    R1 = M.so3_exp(np.radians([1.0, 2.0, -0.5]))
    R2 = M.so3_exp(np.radians([-0.7, 0.3, 1.9]))
    lhs = M.H_from_rotation(R1 @ R2, K, K)
    rhs = M.H_from_rotation(R1, K, K) @ M.H_from_rotation(R2, K, K)
    assert np.allclose(lhs / lhs[2, 2], rhs / rhs[2, 2], atol=1e-9)


def test_composition_through_three_patches_closes():
    Ka, Kb, Kc = _K(cx=100), _K(cx=130), _K(cx=115, cy=140)
    Ra = M.so3_exp(np.radians([0.5, 1.0, 0.2]))
    Rb = M.so3_exp(np.radians([-1.0, 0.4, 0.9]))
    Hab = M.H_from_rotation(Rb.T @ Ra, Ka, Kb)      # a -> b
    Hbc = M.H_from_rotation(Rb, Kb, Kc)             # b -> c
    Hac = M.H_from_rotation(Rb @ (Rb.T @ Ra), Ka, Kc)
    got = Hbc @ Hab
    assert np.allclose(got / got[2, 2], Hac / Hac[2, 2], atol=1e-9)


def test_inverse_transform_is_the_reverse_pair():
    K = _K()
    R = M.so3_exp(np.radians([2.0, -1.0, 0.5]))
    H = M.H_from_rotation(R, K, K)
    Hi = M.H_from_rotation(R.T, K, K)
    p = H @ Hi
    assert np.allclose(p / p[2, 2], np.eye(3), atol=1e-9)


# ── gauge invariance ──────────────────────────────────────────────────────

def test_relative_transforms_are_gauge_invariant():
    """A global rotation of every patch must not change any RELATIVE transform.

    This is why bundle adjustment can hold one patch fixed without loss: the
    absolute frame is unobservable, only differences carry information.
    """
    K = _K()
    G = {i: R for i, R in enumerate(_rots(6, seed=19))}
    A = M.so3_exp(np.radians([12.0, -25.0, 40.0]))          # arbitrary gauge
    Gp = {i: A @ R for i, R in G.items()}
    for i in G:
        for j in G:
            if i == j:
                continue
            r0 = G[j].T @ G[i]
            r1 = Gp[j].T @ Gp[i]
            assert M.rotation_angle_deg(r0.T @ r1) < 1e-9


def test_gauge_change_leaves_reprojection_unchanged():
    K = _K()
    G = {i: R for i, R in enumerate(_rots(4, seed=23))}
    A = M.so3_exp(np.radians([5.0, 9.0, -3.0]))
    src = _pts(12, seed=29)
    for i in G:
        for j in G:
            if i == j:
                continue
            h0 = M.apply_H(M.H_from_rotation(G[j].T @ G[i], K, K), src)
            h1 = M.apply_H(M.H_from_rotation((A @ G[j]).T @ (A @ G[i]), K, K), src)
            assert np.allclose(h0, h1, atol=1e-7)


# ── higher rungs of the ladder ────────────────────────────────────────────

def test_affine_roundtrip():
    A = np.array([[1.02, 0.03, 12.0], [-0.04, 0.98, -7.0], [0, 0, 1.0]])
    src = _pts(20, seed=31)
    dst = M.apply_H(A, src)
    got = M.fit_affine(src, dst)
    assert np.allclose(got, A, atol=1e-8)


def test_affine_needs_three_points():
    assert M.fit_affine(_pts(2, seed=1), _pts(2, seed=2)) is None


def test_homography_roundtrip():
    H = np.array([[1.01, 0.02, 9.0], [-0.03, 0.99, -4.0], [1e-5, -2e-5, 1.0]])
    src = _pts(20, seed=37)
    dst = M.apply_H(H, src)
    got = M.fit_homography(src, dst)
    assert got is not None
    assert M.transfer_error(got, src, dst).max() < 1e-6


def test_homography_needs_four_points():
    assert M.fit_homography(_pts(3, seed=1), _pts(3, seed=2)) is None


def test_quadratic_roundtrip():
    """Can et al.'s 12-parameter model must reproduce a quadratic exactly."""
    src = _pts(40, seed=41)
    T, ns = M._normalise(src)
    coeffs = np.array([[3.0, 210.0, 4.0, 1.5, -0.8, 0.6],
                       [-2.0, 5.0, 195.0, -0.4, 1.1, 0.9]])
    dst = M._quad_basis(ns) @ coeffs.T
    got = M.fit_quadratic(src, dst)
    assert got is not None
    c2, T2 = got
    pred = M.apply_quadratic(c2, T2, src)
    assert np.abs(pred - dst).max() < 1e-6


def test_quadratic_needs_six_points():
    assert M.fit_quadratic(_pts(5, seed=1), _pts(5, seed=2)) is None


def test_ladder_metadata_is_self_consistent():
    assert len(M.LEVEL_NAMES) == len(M.LEVEL_DOF) == len(M.LEVEL_MIN_CORR) == 5
    # A model can never be determined by fewer than dof/2 point correspondences.
    for dof, mc in zip(M.LEVEL_DOF, M.LEVEL_MIN_CORR):
        assert mc >= dof / 2.0


# ── the estimate record ───────────────────────────────────────────────────

def test_estimate_apply_and_residuals_agree():
    K = _K()
    R = M.so3_exp(np.radians([1.0, -2.0, 0.5]))
    est = M.make_estimate(0, R=R, focal=6000.0, matrix=M.H_from_rotation(R, K, K))
    src = _pts(15, seed=43)
    dst = M.estimate_apply(est, src)
    assert M.estimate_residuals(est, src, dst).max() < 1e-9
    assert M.rotation_angle_deg(R.T @ M.estimate_rotation(est)) < 1e-9


def test_estimate_serialisation_roundtrip():
    K = _K()
    R = M.so3_exp(np.radians([0.4, 0.9, -1.2]))
    est = M.make_estimate(0, R=R, focal=6000.0, matrix=M.H_from_rotation(R, K, K))
    s = M.serialisable(est)
    import json
    s2 = json.loads(json.dumps(s))               # must survive a JSON round-trip
    assert s2["level"] == 0 and s2["name"] == "rotation"
    assert np.allclose(np.array(s2["rotvec"]), M.so3_log(R), atol=1e-12)


def test_quadratic_estimate_has_no_rotation():
    """Above L1 there is no rotation to report, and callers must see that."""
    src = _pts(20, seed=47)
    got = M.fit_quadratic(src, M.apply_H(np.eye(3), src))
    est = M.make_estimate(4, coeffs=got[0], T=got[1], matrix=np.eye(3))
    assert M.estimate_rotation(est) is None


def test_rays_are_unit_and_invertible():
    K = _K()
    pts = _pts(20, seed=53)
    v = M.rays(pts, K)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)
    back = M.apply_H(K, v[:, :2] / v[:, 2:3])
    assert np.allclose(back, pts, atol=1e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
