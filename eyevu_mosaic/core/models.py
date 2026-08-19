"""Transform models and the DOF escalation ladder.

The central design decision of this pipeline lives here: patch-to-patch motion
is modelled as a CONJUGATE ROTATION rather than a homography.

Under rotation of the eye about its centre, with the camera entrance pupil near
the eye's nodal point, the mapping between two views is

        H = K R K^-1                       (3 DOF, given K)

An 8-DOF homography needs 4 correspondences and realistically 12-20 inliers
before RANSAC is trustworthy.  This needs TWO, and is overconstrained by three.
On patches that yield 3-5 inliers, that is the difference between "not enough
matches" and "enough matches" without touching the detector.

A NOTE ON WHAT THIS MODEL ACTUALLY IS AT CURRENT PATCH SIZE
-----------------------------------------------------------
The physical justification above is weaker than it looks: the scope is at a
working distance, not at the nodal point, so as the eye rotates the pupil also
translates in the image.  The DOF argument is what carries the design, not the
optics.  And numerically, for a ~225px analysis patch with f ~ 6000 analysis px,
the projective terms of K R K^-1 go as (x/f)^2 ~ 3e-4 px -- utterly negligible.

So at current scale L0 IS a Euclidean transform: in-plane rotation plus
translation, 3 DOF, 2 correspondences.  That is exactly the right model for
this data, and it is strictly better than the 4-DOF similarity it replaces,
which spends its extra DOF absorbing scale error that should not exist.

The K R K^-1 parameterisation is kept rather than fitting a bare Euclidean
transform for two reasons: it composes correctly through the match graph on
SO(3) (which is what makes cycle consistency and rotation averaging meaningful),
and it stays correct as patches grow, where the projective terms stop being
negligible and L1's focal refinement starts to earn its keep.

References
    Stewart, Tsai, Roysam, IEEE TMI 2003     -- model escalation (dual-bootstrap)
    Can, Stewart, Roysam, Tanenbaum, PAMI 2002 -- the 12-parameter quadratic
    Brown & Lowe, IJCV 2007                  -- rotation-model bundle adjustment
    Hartley, Trumpf, Dai, Li, IJCV 2013      -- rotation averaging
"""

from __future__ import annotations

import numpy as np

# Level names, indexed by level number.
LEVEL_NAMES = ("rotation", "rotation+focal", "affine", "homography", "quadratic")
LEVEL_DOF = (3, 4, 6, 8, 12)
LEVEL_MIN_CORR = (2, 3, 3, 4, 6)


# ══ SO(3) ═════════════════════════════════════════════════════════════════

def so3_exp(w):
    """Rodrigues: rotation vector (3,) -> rotation matrix (3, 3)."""
    w = np.asarray(w, np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        # Second-order expansion keeps this smooth and accurate through zero,
        # which matters because the optimiser evaluates near the identity.
        W = _skew(w)
        return np.eye(3) + W + 0.5 * (W @ W)
    k = w / th
    K = _skew(k)
    return (np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K))


def so3_log(R):
    """Rotation matrix (3, 3) -> rotation vector (3,)."""
    R = np.asarray(R, np.float64).reshape(3, 3)
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]], np.float64) * 0.5
    if abs(np.pi - th) < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from the
        # symmetric part instead.
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        if axis[i] > 1e-9:
            axis = A[:, i] / axis[i]
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        return axis * th
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]], np.float64) * (th / (2.0 * np.sin(th)))


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], np.float64)


def rotation_angle_deg(R):
    """Magnitude of a rotation, in degrees.  Used for cycle-consistency errors.

    Computed as atan2(sin, cos) rather than arccos(cos).  It matters: arccos
    loses precision catastrophically near the identity, where its derivative
    diverges, so a rotation that is zero to machine precision reads as ~1e-6
    degrees of error.  Since almost every angle this pipeline measures IS near
    zero -- cycle closure errors, per-edge residuals, convergence checks -- the
    stable form is the one to use.
    """
    R = np.asarray(R, np.float64)
    s = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2],
                              R[0, 2] - R[2, 0],
                              R[1, 0] - R[0, 1]])
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.degrees(np.arctan2(s, c)))


# ══ intrinsics ════════════════════════════════════════════════════════════

def make_K(focal, cx, cy):
    """Pinhole intrinsic matrix from a focal length and principal point."""
    return np.array([[float(focal), 0.0, float(cx)],
                     [0.0, float(focal), float(cy)],
                     [0.0, 0.0, 1.0]], np.float64)


def analysis_focal(cfg, frame_width, analysis_scale):
    """Focal length in ANALYSIS pixels for a patch.

    The rig's optics are fixed, so focal length in native pixels scales with the
    frame's resolution, and analysis space rescales again by `analysis_scale`:

        f_native  = focal_ref_px * frame_width / ref_frame_width
        f_analysis = f_native * analysis_scale

    Getting this per-patch consistency right is what lets patches captured at
    480x640 and 960x1280 -- which really do occur in the same session -- share
    one rotation graph.
    """
    f_native = cfg.focal_ref_px * float(frame_width) / float(cfg.ref_frame_width)
    return f_native * float(analysis_scale)


def H_from_rotation(R, K_src, K_dst):
    """The conjugate rotation taking SRC image points to DST image points."""
    return np.asarray(K_dst, np.float64) @ np.asarray(R, np.float64) @ \
        np.linalg.inv(np.asarray(K_src, np.float64))


def rays(pts, K):
    """Image points (N, 2) -> unit viewing rays (N, 3) through K."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    Ki = np.linalg.inv(np.asarray(K, np.float64))
    h = np.hstack([pts, np.ones((len(pts), 1))])
    v = h @ Ki.T
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# ══ fitting ═══════════════════════════════════════════════════════════════

def fit_rotation(src, dst, K_src, K_dst, weights=None):
    """Closed-form rotation from >= 2 correspondences (Kabsch on viewing rays).

    Both point sets are lifted to unit rays and the rotation that best carries
    one onto the other is recovered by SVD.  Exact, non-iterative, and defined
    from two correspondences -- which is the whole point of the model.

    Returns a (3, 3) rotation, or None if degenerate.
    """
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 2 or len(src) != len(dst):
        return None
    u = rays(src, K_src)
    v = rays(dst, K_dst)
    w = np.ones(len(u)) if weights is None else np.asarray(weights, np.float64)
    M = (v * w[:, None]).T @ u
    if not np.all(np.isfinite(M)):
        return None
    try:
        U, _S, Vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return None
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    return R if np.all(np.isfinite(R)) else None


def fit_rotation_focal(src, dst, K_src, K_dst, cx_s, cy_s, cx_d, cy_d,
                       f0, iters=12):
    """L1: rotation plus a SHARED focal length, from >= 3 correspondences.

    Alternates between the closed-form rotation at the current focal and a
    1-D Gauss-Newton step on the focal at the current rotation.  Cheap, stable,
    and it degrades to L0 if the focal is unobservable -- which at current patch
    size it very nearly is, so the step is bounded hard.

    Returns (R, f) or None.
    """
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 3:
        return None
    f = float(f0)
    R = fit_rotation(src, dst, K_src, K_dst)
    if R is None:
        return None
    for _ in range(iters):
        Ks, Kd = make_K(f, cx_s, cy_s), make_K(f, cx_d, cy_d)
        R = fit_rotation(src, dst, Ks, Kd)
        if R is None:
            return None
        # Numerical derivative of the residual wrt f; the analytic one buys
        # nothing at this problem size.
        h = max(1.0, 1e-3 * f)
        e0 = _rot_residual(R, src, dst, f, cx_s, cy_s, cx_d, cy_d)
        e1 = _rot_residual(R, src, dst, f + h, cx_s, cy_s, cx_d, cy_d)
        g = (e1 - e0) / h
        if abs(g) < 1e-12:
            break
        step = float(np.clip(-e0 / g, -0.2 * f, 0.2 * f))
        f_new = float(np.clip(f + step, 0.2 * f0, 5.0 * f0))
        if abs(f_new - f) < 1e-6 * f:
            f = f_new
            break
        f = f_new
    Ks, Kd = make_K(f, cx_s, cy_s), make_K(f, cx_d, cy_d)
    R = fit_rotation(src, dst, Ks, Kd)
    return None if R is None else (R, f)


def _rot_residual(R, src, dst, f, cx_s, cy_s, cx_d, cy_d):
    H = H_from_rotation(R, make_K(f, cx_s, cy_s), make_K(f, cx_d, cy_d))
    return float(np.mean(transfer_error(H, src, dst)))


def fit_affine(src, dst):
    """L2: 6-DOF affine by least squares, from >= 3 correspondences."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 3:
        return None
    A = np.zeros((2 * len(src), 6))
    b = np.zeros(2 * len(src))
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = src[:, 0], src[:, 1], 1.0
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = src[:, 0], src[:, 1], 1.0
    b[0::2], b[1::2] = dst[:, 0], dst[:, 1]
    try:
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return np.array([[p[0], p[1], p[2]],
                     [p[3], p[4], p[5]],
                     [0.0, 0.0, 1.0]], np.float64)


def fit_homography(src, dst):
    """L3: 8-DOF homography by normalised DLT, from >= 4 correspondences."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 4:
        return None
    Ts, ns = _normalise(src)
    Td, nd = _normalise(dst)
    A = []
    for (x, y), (u, v) in zip(ns, nd):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    try:
        _U, _S, Vt = np.linalg.svd(np.asarray(A, np.float64))
    except np.linalg.LinAlgError:
        return None
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


def fit_quadratic(src, dst):
    """L4: the 12-parameter quadratic of Can et al. (PAMI 2002).

        x' = a0 + a1 x + a2 y + a3 x^2 + a4 xy + a5 y^2
        y' = b0 + b1 x + b2 y + b3 x^2 + b4 xy + b5 y^2

    Designed for wide-baseline retinal pairs, where the curvature of the retinal
    surface makes a planar homography systematically wrong.  Needs >= 6
    correspondences, and coordinates are normalised first or the quadratic terms
    wreck the conditioning.

    Returns (coeffs (2, 6), normalisation T) or None.
    """
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 6:
        return None
    T, ns = _normalise(src)
    M = _quad_basis(ns)
    try:
        p, *_ = np.linalg.lstsq(M, dst, rcond=None)      # (6, 2)
    except np.linalg.LinAlgError:
        return None
    return (p.T.copy(), T) if np.all(np.isfinite(p)) else None


def _quad_basis(p):
    x, y = p[:, 0], p[:, 1]
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _normalise(pts):
    """Hartley normalisation: centroid to the origin, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = (np.sqrt(2.0) / d) if d > 1e-12 else 1.0
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]], np.float64)
    return T, (pts - c) * s


# ══ applying and scoring ══════════════════════════════════════════════════

def apply_H(H, pts):
    """Perspective transform of (N, 2) points by a 3x3."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, np.float64).T
    w = h[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.sign(w) * 1e-12 + 1e-12, w)
    return h[:, :2] / w


def apply_quadratic(coeffs, T, pts):
    """Apply a fitted 12-parameter quadratic to (N, 2) points."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ns = apply_H(T, pts)
    return _quad_basis(ns) @ np.asarray(coeffs, np.float64).T


def transfer_error(H, src, dst):
    """Per-correspondence transfer distance in pixels, for a 3x3 model."""
    return np.linalg.norm(apply_H(H, src) - np.asarray(dst, np.float64).reshape(-1, 2),
                          axis=1)


# ══ the estimate record ═══════════════════════════════════════════════════

def make_estimate(level, *, R=None, focal=None, matrix=None, coeffs=None, T=None):
    """One fitted pair transform, in the form the whole pipeline passes around.

    `matrix` is always populated for levels 0-3 so downstream code (warping,
    graph chaining, bounding boxes) has a uniform 3x3 to use.  Level 4 has no
    matrix form and carries `coeffs`/`T` instead; callers that need a linear
    approximation should fall back to the level-3 fit.
    """
    return {
        "level": int(level),
        "name": LEVEL_NAMES[int(level)],
        "dof": LEVEL_DOF[int(level)],
        "matrix": None if matrix is None else np.asarray(matrix, np.float64),
        "rotvec": None if R is None else so3_log(R),
        "focal": None if focal is None else float(focal),
        "coeffs": None if coeffs is None else np.asarray(coeffs, np.float64),
        "T": None if T is None else np.asarray(T, np.float64),
    }


def estimate_apply(est, pts):
    """Map points through an estimate, whatever level it settled at."""
    if est.get("level") == 4 and est.get("coeffs") is not None:
        return apply_quadratic(est["coeffs"], est["T"], pts)
    return apply_H(est["matrix"], pts)


def estimate_residuals(est, src, dst):
    """Per-correspondence residual distances for an estimate."""
    return np.linalg.norm(
        estimate_apply(est, src) - np.asarray(dst, np.float64).reshape(-1, 2),
        axis=1)


def estimate_rotation(est):
    """The (3, 3) rotation of a level-0/1 estimate, or None above the ladder."""
    rv = est.get("rotvec")
    return None if rv is None else so3_exp(rv)


def serialisable(est):
    """An estimate as plain JSON types, for bundle.json."""
    out = {"level": est["level"], "name": est["name"], "dof": est["dof"],
           "focal": est.get("focal")}
    for k in ("matrix", "rotvec", "coeffs", "T"):
        v = est.get(k)
        out[k] = None if v is None else np.asarray(v).tolist()
    return out
