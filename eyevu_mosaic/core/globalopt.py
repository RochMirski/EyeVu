"""Feature tracks, rotation averaging, and robust bundle adjustment.

Chaining rotations outward along a spanning tree gives every patch a position,
but the error accumulates along each chain and nothing ever closes a loop.  This
module spends the redundancy in the graph: every correspondence that survived
pairwise verification constrains the global solution, not just the ones that
happened to lie on the tree.

Bundle adjustment is over per-patch ROTATIONS ONLY (3 parameters each, one patch
held fixed for gauge), plus optionally a shared focal length.  That is the whole
model -- there are no translations, because the model says there are none.  With
40 patches this is a few hundred parameters, so scipy.optimize.least_squares
with an explicit `jac_sparsity` is entirely adequate; Ceres or GTSAM would be a
painful build on this hardware for no benefit at this problem size.
"""

from __future__ import annotations

import numpy as np

from . import models as M
from .preprocess import patch_K

# scipy is used ONLY for bundle adjustment.  It is imported lazily and its
# absence is survivable, because it may genuinely be absent: the Pi installer
# has the numpy/scipy/scikit-image pip block commented out, so a device can be
# running with neither.  Without scipy the pipeline still registers, still
# reconciles the graph by rotation averaging (pure numpy), and still composites
# -- it just skips the final refinement and says so in the bundle.  Failing to
# import at all would be far worse than losing one stage.
try:
    from scipy import optimize
    from scipy.sparse import lil_matrix
    SCIPY_AVAILABLE = True
except ImportError:                                         # pragma: no cover
    optimize = None
    lil_matrix = None
    SCIPY_AVAILABLE = False


# ══ tracks ════════════════════════════════════════════════════════════════

def _so3_exp_batch(W):
    """Rodrigues for a stack of rotation vectors (N, 3) -> (N, 3, 3).

    The optimiser evaluates this once per patch per residual call, hundreds of
    times per solve, so it is worth not doing in a Python loop.
    """
    W = np.asarray(W, np.float64).reshape(-1, 3)
    th = np.linalg.norm(W, axis=1)
    small = th < 1e-9
    safe = np.where(small, 1.0, th)
    k = W / safe[:, None]
    K = np.zeros((len(W), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], (1.0 - np.cos(th))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    if small.any():
        # Second-order expansion stays accurate and smooth through zero.
        Ws = np.zeros((int(small.sum()), 3, 3))
        w = W[small]
        Ws[:, 0, 1], Ws[:, 0, 2] = -w[:, 2], w[:, 1]
        Ws[:, 1, 0], Ws[:, 1, 2] = w[:, 2], -w[:, 0]
        Ws[:, 2, 0], Ws[:, 2, 1] = -w[:, 1], w[:, 0]
        R[small] = np.eye(3)[None] + Ws + 0.5 * (Ws @ Ws)
    return R


def _so3_log_batch(R):
    """Rotation matrices (N, 3, 3) -> rotation vectors (N, 3).

    atan2 form, stable near the identity -- which is where every one of these
    residuals lives once the solution is close.
    """
    R = np.asarray(R, np.float64).reshape(-1, 3, 3)
    a = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=1) * 0.5
    s = np.linalg.norm(a, axis=1)
    c = (np.trace(R, axis1=1, axis2=2) - 1.0) * 0.5
    th = np.arctan2(s, np.clip(c, -1.0, 1.0))
    k = np.where(s < 1e-12, 1.0, th / np.maximum(s, 1e-12))
    return a * k[:, None]


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_tracks(records, cfg, keep_nodes=None):
    """Union-find over inlier correspondences -> feature tracks.

    A track is a set of (patch, keypoint) observations believed to be the same
    physical point.  Tracks with two observations in the SAME patch are
    discarded outright: one point cannot be in two places in one image, so such
    a track proves at least one of its edges is wrong and there is no principled
    way to decide which.

    Returns (tracks, stats) with each track a sorted list of (patch, kp).
    """
    uf = _UF()
    for rec in records:
        if not rec.get("accepted") or rec.get("matches") is None:
            continue
        i, j = rec["pair"]
        if keep_nodes is not None and (i not in keep_nodes or j not in keep_nodes):
            continue
        mt, inl = rec["matches"], rec["inlier_mask"]
        for (a, b), ok in zip(mt, inl):
            if ok and a >= 0 and b >= 0:      # -1 marks a landmark prior
                uf.union((i, int(a)), (j, int(b)))

    groups = {}
    for obs in uf.p:
        groups.setdefault(uf.find(obs), []).append(obs)

    tracks, n_conflict = [], 0
    for g in groups.values():
        pats = [o[0] for o in g]
        if len(set(pats)) != len(pats):
            n_conflict += 1
            continue
        if len(g) < max(2, int(cfg.ba_min_track_len)):
            continue
        tracks.append(sorted(g))
    tracks.sort(key=lambda t: (-len(t), t[0]))
    stats = {"n_tracks": len(tracks), "n_conflicting_tracks": n_conflict,
             "n_observations": int(sum(len(t) for t in tracks)),
             "max_track_len": max((len(t) for t in tracks), default=0)}
    return tracks, stats


# ══ rotation averaging ════════════════════════════════════════════════════

def rotation_average(comp, edges, G, iters=30, tol=1e-9):
    """Iteratively reconcile every edge, not just the spanning tree.

    For each patch, each neighbour proposes G_i = G_j R_ij; the proposals are
    averaged in the tangent space at the current estimate and the patch is
    updated.  This is the simple single-rotation-averaging iteration of Hartley
    et al. (IJCV 2013), which is plenty at this graph size and, unlike bundle
    adjustment, needs no feature tracks at all -- so it is also the fallback
    when the graph is not a pure rotation graph.

    The root (the patch already at identity) is held fixed for gauge.
    """
    G = {k: np.array(v, np.float64) for k, v in G.items()}
    if len(G) < 2:
        return G, {"iterations": 0, "final_delta_deg": 0.0}
    root = min(G, key=lambda n: (np.linalg.norm(M.so3_log(G[n])), n))

    last = 0.0
    for it in range(int(iters)):
        delta = 0.0
        for i in sorted(G):
            if i == root:
                continue
            props = [G[j] @ edges[(i, j)]["R"]
                     for j in comp if j != i and (i, j) in edges and j in G]
            if len(props) < 2:
                continue
            w = np.array([edges[(i, j)]["w"]
                          for j in comp if j != i and (i, j) in edges and j in G])
            w = w / max(w.sum(), 1e-9)
            # Average in the tangent space at the current estimate.
            tang = np.array([M.so3_log(G[i].T @ P) for P in props])
            step = (tang * w[:, None]).sum(axis=0)
            G[i] = G[i] @ M.so3_exp(step)
            delta = max(delta, float(np.degrees(np.linalg.norm(step))))
        last = delta
        if delta < tol:
            break
    return G, {"iterations": it + 1, "final_delta_deg": last}


# ══ bundle adjustment ═════════════════════════════════════════════════════

def bundle_adjust(patches, features, tracks, G, cfg, edge_priors=None,
                  focal=3600.0, radius=70.0):
    """Robust BA over per-patch rotations (and optionally a shared focal).

    Parameters
        rotvec per patch, EXCEPT the gauge-fixed root  ->  3 (n - 1)
        two tangent offsets per track direction        ->  2 T
        shared focal, if enabled                       ->  1

    Track directions are explicit parameters, which makes each residual depend
    on exactly one patch and one track and therefore makes `jac_sparsity` an
    honest, very sparse pattern.

    Each direction is parameterised by TWO offsets in the tangent plane of its
    initial estimate, not by a free 3-vector.  A 3-vector has a redundant radial
    DOF, and pinning it with a unit-norm penalty makes the problem stiff: the
    optimiser then spends its whole budget fighting a constraint instead of
    fitting the data.  Measured, that version hit the 600-evaluation ceiling
    without converging on every component of a 12-patch session (31 s of solver
    time for three components of 5, 4 and 2 patches).  A minimal parameterisation
    has no null space and needs no penalty.

    Huber loss, because a track that survived pairwise verification can still
    contain one bad observation and a squared loss would let it dominate.

    Returns (G_refined, focal, stats).
    """
    nodes = sorted(G)
    stats = {"ran": False, "reason": "", "n_tracks": len(tracks)}
    by_index = {p.index: p for p in patches}
    feat = {f.index: f for f in features}

    # Bound the problem size.  Each track adds three parameters and two
    # residuals per observation, and cost grows with both -- but the marginal
    # value of the thousandth track is nil once the rotations are already
    # determined by the first few hundred.  Longest tracks first: a track seen
    # in five patches constrains far more than one seen in two.
    if len(tracks) > cfg.ba_max_tracks:
        tracks = sorted(tracks, key=lambda t: -len(t))[:int(cfg.ba_max_tracks)]
        stats["n_tracks_used"] = len(tracks)

    if not SCIPY_AVAILABLE:
        stats["reason"] = "scipy unavailable; rotation averaging only"
        return G, None, stats

    obs = [(ti, p, k) for ti, t in enumerate(tracks) for (p, k) in t if p in G]
    # Drop tracks that lost all but one observation to the component filter.
    seen = {}
    for ti, p, k in obs:
        seen[ti] = seen.get(ti, 0) + 1
    obs = [o for o in obs if seen[o[0]] >= 2]
    tids = sorted({o[0] for o in obs})
    # Priors alone are enough to bundle: a graph carried entirely by direct
    # alignment has no tracks at all, and refusing to refine it would leave the
    # spanning-tree chain unimproved.
    if len(nodes) < 2 or (not obs and not edge_priors):
        stats["reason"] = "not enough tracks or edge priors to bundle"
        return G, None, stats

    nmap = {n: i for i, n in enumerate(nodes)}
    tmap = {t: i for i, t in enumerate(tids)}
    root = nodes[0]
    free = [n for n in nodes if n != root]
    fmap = {n: i for i, n in enumerate(free)}
    nT = len(tids)
    n_rot = 3 * len(free)
    refine_f = bool(cfg.ba_refine_focal)
    f0 = float(np.median([by_index[n].focal for n in nodes]))

    # ── initial values ──
    x0 = np.zeros(n_rot + 2 * nT + (1 if refine_f else 0))
    for n in free:
        x0[3 * fmap[n]:3 * fmap[n] + 3] = M.so3_log(G[n])
    dirs = np.zeros((nT, 3))
    for ti, p, k in obs:
        v = M.rays(feat[p].xy[k][None, :], patch_K(by_index[p]))[0]
        dirs[tmap[ti]] += G[p] @ v            # patch frame -> reference frame
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    # Orthonormal tangent basis at each initial direction; the two parameters
    # are offsets along these, so the direction stays on the sphere by
    # construction and there is no redundant DOF to constrain.
    seed = np.tile(np.array([1.0, 0.0, 0.0]), (max(nT, 1), 1))
    seed[np.abs(dirs[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    e1 = seed - dirs * np.sum(seed * dirs, axis=1, keepdims=True)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(dirs, e1)
    if refine_f:
        x0[-1] = f0

    # dtype is explicit because a component carried entirely by direct
    # alignment has no observations at all, and np.array([]) is float -- which
    # then fails as an index rather than harmlessly selecting nothing.
    obs_t = np.array([tmap[o[0]] for o in obs], dtype=int)
    obs_p = np.array([nmap[o[1]] for o in obs], dtype=int)
    obs_xy = np.array([feat[o[1]].xy[o[2]] for o in obs],
                      np.float64).reshape(-1, 2)
    centres = np.array([by_index[n].centre for n in nodes], np.float64)
    focals = np.array([by_index[n].focal for n in nodes], np.float64)

    # Relative-rotation priors.  An edge found by DIRECT alignment contributes
    # no correspondences, so it is invisible to a purely reprojection-based BA
    # -- which then happily drifts away from it.  Measured on a 30-patch session
    # where most edges came from the direct path: BA left those edges violated
    # by 4-7 px and the global pointing error at 0.54 deg, against 0.03 deg on a
    # session where the feature path carried the graph.  Adding the edges as
    # explicit constraints is the fix, and it makes this a hybrid of
    # reprojection and relative-pose BA rather than one or the other.
    priors = [(nmap[i], nmap[j], R) for (i, j, R) in (edge_priors or [])
              if i in nmap and j in nmap]
    nP = len(priors)
    pri_i = np.array([p[0] for p in priors], int)
    pri_j = np.array([p[1] for p in priors], int)
    pri_R = np.array([p[2] for p in priors]) if nP else np.zeros((0, 3, 3))
    # Weights convert an angular deviation into the pixels it would cost, so the
    # priors are commensurate with the reprojection residuals and the Huber
    # scale means the same thing for both.
    pri_w = np.array([focal, focal, radius], np.float64)

    free_slot = np.array([nmap[n] for n in free], int)

    def unpack(x):
        Rs = np.repeat(np.eye(3)[None], len(nodes), axis=0)
        if len(free):
            Rs[free_slot] = _so3_exp_batch(x[:n_rot].reshape(-1, 3))
        ab = x[n_rot:n_rot + 2 * nT].reshape(nT, 2)
        d = dirs + ab[:, 0:1] * e1 + ab[:, 1:2] * e2
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        f = x[-1] if refine_f else None
        return Rs, d, f

    def residuals(x):
        Rs, d, f = unpack(x)
        # Reference-frame direction -> this patch's frame: v_i = G_i^T v_ref.
        v = np.einsum("nij,nj->ni", np.transpose(Rs[obs_p], (0, 2, 1)), d[obs_t])
        z = np.where(np.abs(v[:, 2]) < 1e-9, 1e-9, v[:, 2])
        fp = (np.full(len(obs), f) if f is not None else focals[obs_p])
        px = centres[obs_p, 0] + fp * v[:, 0] / z
        py = centres[obs_p, 1] + fp * v[:, 1] / z
        r = np.empty(2 * len(obs) + 3 * nP)
        r[0:2 * len(obs):2] = px - obs_xy[:, 0]
        r[1:2 * len(obs):2] = py - obs_xy[:, 1]
        # Behind-camera observations are meaningless; penalise hard rather than
        # letting a sign flip look like a good fit.
        bad = v[:, 2] <= 0
        if bad.any():
            r[0:2 * len(obs):2][bad] = 1e4
            r[1:2 * len(obs):2][bad] = 1e4
        if nP:
            # Deviation of the solved relative rotation from the measured one.
            pred = np.matmul(np.transpose(Rs[pri_j], (0, 2, 1)), Rs[pri_i])
            E = np.matmul(np.transpose(pred, (0, 2, 1)), pri_R)
            r[2 * len(obs):] = (_so3_log_batch(E) * pri_w[None, :]).ravel()
        return r

    # ── sparsity: residual (t, p) touches only patch p and track t ──
    S = lil_matrix((2 * len(obs) + 3 * nP, len(x0)), dtype=int)
    for r, (ti, pi) in enumerate(zip(obs_t, obs_p)):
        n = nodes[pi]
        if n != root:
            S[2 * r, 3 * fmap[n]:3 * fmap[n] + 3] = 1
            S[2 * r + 1, 3 * fmap[n]:3 * fmap[n] + 3] = 1
        S[2 * r, n_rot + 2 * ti:n_rot + 2 * ti + 2] = 1
        S[2 * r + 1, n_rot + 2 * ti:n_rot + 2 * ti + 2] = 1
        if refine_f:
            S[2 * r, -1] = 1
            S[2 * r + 1, -1] = 1
    base = 2 * len(obs)
    for k, (pi, pj, _R) in enumerate(priors):
        for slot in (pi, pj):
            n = nodes[slot]
            if n == root:
                continue
            S[base + 3 * k:base + 3 * k + 3,
              3 * fmap[n]:3 * fmap[n] + 3] = 1

    r0 = residuals(x0)
    try:
        sol = optimize.least_squares(
            residuals, x0, jac_sparsity=S, loss="huber",
            f_scale=float(cfg.ba_huber_delta), method="trf",
            # The parameter vector mixes units -- rotations in radians (~1e-2)
            # with dimensionless tangent offsets (~1e-3) -- so a single trust
            # region radius fits neither and the solver grinds.  Scaling from
            # the Jacobian fixes that.  The tolerances are set for a problem
            # measured in PIXELS; scipy's 1e-8 defaults chase precision far
            # below the noise floor and were burning the whole evaluation
            # budget without improving the fit (600 nfev, status 0, on
            # components of 4 and 5 patches).
            x_scale="jac",
            ftol=float(cfg.ba_ftol), xtol=float(cfg.ba_xtol),
            gtol=float(cfg.ba_gtol),
            max_nfev=int(cfg.ba_max_iters) * 10, verbose=0)
    except (ValueError, np.linalg.LinAlgError) as e:
        stats["reason"] = f"least_squares failed: {e}"
        return G, None, stats

    Rs, _d, f = unpack(sol.x)
    out = {n: Rs[nmap[n]] for n in nodes}
    npx = 2 * len(obs)
    stats.update({
        "ran": True,
        "n_parameters": int(len(x0)),
        "n_residuals": int(len(r0)),
        "n_observations": int(len(obs)),
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "nfev": int(sol.nfev),
        "rms_before_px": float(np.sqrt(np.mean(r0[:npx] ** 2))),
        "rms_after_px": float(np.sqrt(np.mean(sol.fun[:npx] ** 2))),
        "median_before_px": float(np.median(np.abs(r0[:npx]))),
        "median_after_px": float(np.median(np.abs(sol.fun[:npx]))),
        "focal_before": f0,
        "focal_after": (float(f) if f is not None else None),
    })
    return out, (float(f) if f is not None else None), stats


def refine(patches, features, records, comp, edges, G, cfg,
           focal=None, radius=None):
    """Global refinement for one component: BA when valid, averaging otherwise.

    If most edges settled ABOVE L1 the graph is not a pure rotation graph, and
    bundling it as one would impose a model the edges have already been shown
    not to obey.  In that case fall back to rotation averaging over the L0
    rotations and record the per-edge residual left over, rather than producing
    a confident-looking answer from the wrong model.

    Before either, bad edges are removed by consensus -- see `_reject_outliers`.
    """
    comp_set = set(comp)
    used = [e["rec"] for k, e in edges.items() if k[0] < k[1]
            and k[0] in comp_set and k[1] in comp_set]
    levels = [r.get("level", 0) for r in used]
    high = sum(1 for l in levels if l is not None and l > 1)
    stats = {"n_edges": len(used), "n_edges_above_l1": high,
             "levels": levels}

    f = float(focal if focal else 3600.0)
    r = float(radius if radius else 70.0)
    dropped = []

    # Solve, then look for edges the solution cannot reconcile, then re-solve
    # without them.  The order matters: rotation averaging SPREADS a bad edge's
    # error over the whole component, which hides it, whereas bundle adjustment
    # is driven by feature tracks and pulls the solution onto the good
    # structure, leaving a bad edge standing out by two orders of magnitude.
    # Measured on a 30-patch session: two false edges sat at 117 px against a
    # median of 0.2 px after BA -- but were unremarkable before it.
    for round_i in range(int(cfg.global_outlier_rounds)):
        G, avg = rotation_average(comp, edges, G, iters=30)
        stats["rotation_averaging"] = avg
        # Only edges that contributed NO correspondences become explicit
        # relative-rotation constraints.  A feature edge is already represented
        # in the problem by its inlier tracks, so adding it again would both
        # double-count it and -- because a prior residual couples two patches
        # where a reprojection residual touches only one -- wreck the Jacobian
        # sparsity that makes this solve cheap.  Measured, priors on every edge
        # took bundle adjustment on a 40-patch session from 12 s to 79 s.
        priors = [(i, j, e["R"]) for (i, j), e in edges.items()
                  if i < j and i in comp_set and j in comp_set
                  and e["rec"].get("matches") is None]
        G, stats = _solve(patches, features, records, comp_set, edges, G, cfg,
                          stats, used, high, priors, f, r)
        edges, more = _reject_outliers(comp, edges, G, cfg, f, r)
        dropped += more
        if not [d for d in more if not d.get("kept")]:
            break

    stats["outlier_edges_dropped"] = dropped
    stats["outlier_rounds"] = round_i + 1
    stats["edge_residuals"] = _edge_residuals(comp, edges, G, f, r)
    return G, stats


def _solve(patches, features, records, comp_set, edges, G, cfg, stats,
           used, high, priors=None, focal=3600.0, radius=70.0):
    """One pass of the chosen global solver over the current edge set."""
    if not cfg.bundle_adjust:
        stats["method"] = "rotation_averaging (bundle_adjust disabled)"
    elif used and high > len(used) / 2:
        stats["method"] = ("rotation_averaging (mixed-model graph: "
                           f"{high}/{len(used)} edges above L1)")
    else:
        tracks, tstats = build_tracks(records, cfg, keep_nodes=comp_set)
        stats["tracks"] = tstats
        G2, focal_out, bstats = bundle_adjust(
            patches, features, tracks, G, cfg,
            edge_priors=(priors if cfg.ba_use_edge_priors else None),
            focal=focal, radius=radius)
        stats["bundle_adjustment"] = bstats
        if bstats.get("ran"):
            stats["method"] = "bundle_adjustment"
            G = G2
            if focal_out is not None:
                stats["focal_refined"] = focal_out
        else:
            stats["method"] = f"rotation_averaging ({bstats.get('reason', '')})"
    return G, stats


def _reject_outliers(comp, edges, G, cfg, focal, radius):
    """Drop edges the global solution cannot reconcile, worst first.

    Cycle consistency only tests edges that sit in a triangle.  On these
    sessions the graph is barely denser than a tree -- 32 edges over 30 patches
    gave just 9 triangles -- so most edges are never cross-checked, and one bad
    edge in a tree-like graph corrupts every patch downstream of it.  Measured:
    two false edges took a 30-patch session from ~1 px of median misalignment to
    11 px.

    This is the same idea applied to the whole graph at once: solve, ask each
    edge how far it is from what the consensus implies, drop the worst offender,
    re-solve.  An edge whose removal would disconnect the component is kept
    however bad it looks -- losing a patch entirely is worse than placing it
    imperfectly, and the residual is reported either way.
    """
    dropped = []
    if not cfg.global_outlier_reject:
        return edges, dropped
    edges = dict(edges)
    skip = set()
    for _ in range(int(cfg.global_outlier_max_drops)):
        res = [r for r in _edge_residuals(comp, edges, G, focal, radius)
               if tuple(r["edge"]) not in skip]
        if not res:
            break
        worst = max(res, key=lambda r: r["residual_px"])
        if worst["residual_px"] <= cfg.global_outlier_max_px:
            break
        i, j = worst["edge"]
        if _is_bridge(comp, edges, (i, j)):
            # Keeping a patch badly placed beats losing it entirely.  Note it,
            # then carry on looking -- a bridge must not stop the search, or a
            # second bad edge behind it is never reached.
            worst["kept"] = "would disconnect the component"
            dropped.append(worst)
            skip.add((i, j))
            continue
        edges.pop((i, j), None)
        edges.pop((j, i), None)
        dropped.append(worst)
        G, _ = rotation_average(comp, edges, G, iters=20)
    return edges, dropped


def _is_bridge(comp, edges, edge):
    """Would removing this edge split the component?"""
    i, j = edge
    adj = {n: set() for n in comp}
    for (a, b) in edges:
        if (a, b) in ((i, j), (j, i)):
            continue
        if a in adj and b in adj:
            adj[a].add(b)
    seen, stack = set(), [comp[0]]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj[n] - seen)
    return len(seen) != len(comp)


def _edge_residuals(comp, edges, G, focal=3600.0, radius=70.0):
    """How far each edge is from being explained by the global solution.

    This is the 'per-edge residual correction' bookkeeping: after averaging or
    bundling, an edge that still disagrees is telling you it was wrong, and the
    dashboard should show it.

    Reported in image pixels as well as degrees, for the same reason cycle
    errors are (see graph.cycle_filter): pointing error costs f * angle and roll
    only r * angle, a ratio of ~50 here, so a figure in degrees is dominated by
    the one component that barely moves the image.
    """
    out = []
    for (i, j), e in sorted(edges.items()):
        if i >= j or i not in G or j not in G:
            continue
        pred = G[i].T @ G[j]                # what the global solution implies
        E = pred @ e["R"]
        w = M.so3_log(E)
        px = float(np.hypot(focal * np.linalg.norm(w[:2]), radius * abs(w[2])))
        out.append({"edge": [i, j], "residual_px": round(px, 3),
                    "residual_deg": round(float(M.rotation_angle_deg(E)), 4),
                    "source": e["rec"].get("source", "features")})
    return out
