"""The match graph: edge weighting, cycle consistency, components, MST init.

Nodes are patches; edges are accepted pairwise estimates.  Two things happen
here that the pairwise stage cannot do, because they need more than one pair at
a time.

CYCLE CONSISTENCY is the important one.  For any triangle (i, j, k) the
composed rotation R_ki R_jk R_ij must be the identity, because it is a loop.
It costs three matrix multiplies per triangle and it catches edges that look
entirely plausible on their own -- right inlier count, decent NCC -- but cannot
be reconciled with the rest of the graph.  Given how weakly inlier count
discriminates on this material (true pairs 3-5, false pairs 5-8, no separation),
a test that uses evidence from OUTSIDE the pair is worth more than any amount of
further tuning inside it.

An edge that participates in no triangle cannot be tested this way and is kept
on its photometric evidence alone, flagged so the dashboard can show that it was
never cross-checked.
"""

from __future__ import annotations

import numpy as np

from . import models as M


def edge_weight(rec):
    """Confidence in one edge.

    Inlier count alone is a poor discriminator here, so the photometric score
    carries most of the weight and the residual damps edges that fit loosely.
    """
    n = float(rec.get("n_inliers", 0))
    ncc = float(rec.get("ncc", 0.0))
    med = float(rec.get("median_residual") or 0.0)
    return n * max(ncc, 0.0) / (1.0 + med)


def build(records):
    """Accepted records -> {(i, j): edge}, both directions, plus node set.

    Each edge carries the best-level estimate (used for warping) and the L0
    rotation (used for every graph and rotation-averaging operation), so a
    mixed-level graph still has a consistent rotation structure to reason over.
    """
    edges, nodes = {}, set()
    for rec in records:
        if not rec.get("accepted"):
            continue
        i, j = rec["pair"]
        R = _rec_rotation(rec)
        if R is None:
            continue
        w = edge_weight(rec)
        nodes.update((i, j))
        edges[(i, j)] = {"R": R, "w": w, "rec": rec, "reversed": False}
        edges[(j, i)] = {"R": R.T, "w": w, "rec": rec, "reversed": True}
    return nodes, edges


def _rec_rotation(rec):
    """The L0 rotation for an edge, whatever level the pair finally settled at."""
    est = rec.get("estimate_l0") or rec.get("estimate")
    if not est or est.get("rotvec") is None:
        return None
    return M.so3_exp(est["rotvec"])


# ══ cycle consistency ═════════════════════════════════════════════════════

def cycle_filter(nodes, edges, cfg, focal=None, radius=None):
    """Reject the weakest edge of any triangle that does not close.

    THE ERROR IS MEASURED IN PIXELS, NOT DEGREES, and that is a deliberate
    departure from the obvious formulation.  A rotation has three degrees of
    freedom which affect the image completely differently at this field of view:

        rx, ry (pointing)   move the image by  f * angle   -- f ~ 3600 px, so
                            0.03 deg is already 1.9 px
        rz (roll)           moves it by        r * angle   -- r ~ 70 px, so
                            2 deg is only 2.4 px

    Roll is therefore both the WORST-observed parameter (it is fitted from the
    angular arrangement of a handful of points over a short lever arm) and the
    LEAST consequential one.  A closure error in degrees is dominated by it.

    Measured on a synthetic session where every accepted edge was independently
    confirmed correct: median edge error 2.12 deg, of which pointing was 0.030
    deg and roll 2.12 deg -- and the resulting median triangle closure was 4.6
    deg.  A 2-degree gate, the natural choice, discarded three of the four
    triangles in the session, every one of them good.  In pixels the same edges
    are accurate to a median of 3.4 analysis px, and gross errors stand out by
    orders of magnitude, which is the separation a filter needs.

    Both numbers are reported; only the pixel one decides.

    Returns (edges, report).
    """
    report = {"triangles": [], "dropped": [], "untested_edges": [],
              "metric": "image displacement in analysis px",
              "threshold_px": float(cfg.cycle_max_error_px)}
    if not cfg.cycle_check:
        return edges, report
    f = float(focal if focal else 3600.0)
    r = float(radius if radius else 70.0)

    undirected = sorted({tuple(sorted(k)) for k in edges})
    adj = {n: set() for n in nodes}
    for i, j in undirected:
        adj[i].add(j)
        adj[j].add(i)

    reported = set()
    # Each pass re-evaluates EVERY triangle still present, condemns the single
    # worst, and starts over.  Re-evaluating from scratch is what makes this
    # correct with more than one bad edge: a triangle judged acceptable while a
    # bad edge was still in the graph may be the one that exposes the next bad
    # edge once it is gone, and conversely a triangle can look bad only because
    # it inherited an error that has since been removed.  Dropping one edge per
    # pass, rather than every offending edge at once, avoids removing three good
    # edges because one bad edge spoiled three triangles.
    while True:
        worst, worst_err, worst_tri = None, float(cfg.cycle_max_error_px), None
        for i, j in sorted({tuple(sorted(k)) for k in edges}):
            for k in sorted(adj[i] & adj[j]):
                if k in (i, j):
                    continue
                tri = tuple(sorted((i, j, k)))
                res = _closure_error(tri, edges, f, r)
                if res is None:
                    continue
                e, deg = res
                if tri not in reported:
                    reported.add(tri)
                    report["triangles"].append({"triangle": list(tri),
                                                "error_px": round(e, 3),
                                                "error_deg": round(deg, 4)})
                if e > worst_err:
                    a, b, c = tri
                    cands = [p for p in ((a, b), (b, c), (a, c)) if p in edges]
                    if not cands:
                        continue
                    # With a single triangle there is no way to tell WHICH of
                    # the three edges is wrong, so drop the least confident --
                    # the best available heuristic, not a proof.
                    weak = min(cands, key=lambda p: edges[p]["w"])
                    worst, worst_err, worst_tri = weak, e, tri
        if worst is None:
            break
        report["dropped"].append({"edge": list(worst),
                                  "triangle": list(worst_tri),
                                  "error_px": round(worst_err, 3),
                                  "reason": "cycle inconsistency"})
        edges.pop(worst, None)
        edges.pop((worst[1], worst[0]), None)
        i, j = worst
        adj[i].discard(j)
        adj[j].discard(i)

    for i, j in sorted({tuple(sorted(k)) for k in edges}):
        if not (adj[i] & adj[j]):
            report["untested_edges"].append([i, j])
    return edges, report


def _closure_error(tri, edges, focal, radius):
    """How far the loop R_ki R_jk R_ij fails to close.

    Returns (displacement_px, angle_deg) or None.  The displacement weights each
    rotation component by how much it actually moves the image: pointing error
    by the focal length, roll by the patch radius.  See `cycle_filter`.
    """
    i, j, k = tri
    try:
        Rij = edges[(i, j)]["R"]
        Rjk = edges[(j, k)]["R"]
        Rki = edges[(k, i)]["R"]
    except KeyError:
        return None
    E = Rki @ Rjk @ Rij
    w = M.so3_log(E)
    px = float(np.hypot(focal * np.linalg.norm(w[:2]), radius * abs(w[2])))
    return px, M.rotation_angle_deg(E)


# ══ components and spanning tree ══════════════════════════════════════════

def components(nodes, edges):
    """Connected components of the graph, largest first, each a sorted list."""
    adj = {n: set() for n in nodes}
    for i, j in edges:
        adj[i].add(j)
    seen, out = set(), []
    for start in sorted(nodes):
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(adj[n] - comp)
        seen |= comp
        out.append(sorted(comp))
    return sorted(out, key=lambda c: (-len(c), c[0]))


def max_spanning_tree(comp, edges):
    """Maximum spanning tree over one component, by edge confidence (Kruskal)."""
    parent = {n: n for n in comp}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cand = sorted({tuple(sorted(k)) for k in edges if k[0] in parent},
                  key=lambda p: -edges[p]["w"])
    tree = []
    for i, j in cand:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
    return tree


def initial_rotations(comp, edges, tree):
    """Chain rotations outward from the best-connected node.

    Global rotation G_i takes a ray in patch i's frame into the reference frame,
    so for an edge whose R takes frame i to frame j:  G_j = G_i R_ij^T.

    The root is the highest-degree node rather than the first capture: the seed
    defines the frame every other patch's error accumulates relative to, and
    starting from a weakly connected patch is what caps a mosaic at two frames.
    """
    if not comp:
        return {}
    deg = {n: 0.0 for n in comp}
    for i, j in tree:
        deg[i] += edges[(i, j)]["w"]
        deg[j] += edges[(i, j)]["w"]
    root = max(comp, key=lambda n: (deg[n], -n))

    adj = {n: [] for n in comp}
    for i, j in tree:
        adj[i].append(j)
        adj[j].append(i)

    G = {root: np.eye(3)}
    stack = [root]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j in G:
                continue
            G[j] = G[i] @ edges[(i, j)]["R"].T
            stack.append(j)
    return G


def summarise(nodes, edges, comps, tree_by_comp):
    """Compact graph description for bundle.json."""
    return {
        "n_nodes": len(nodes),
        "n_edges": len({tuple(sorted(k)) for k in edges}),
        "components": [list(c) for c in comps],
        "mst_edges": {str(ci): [list(e) for e in t]
                      for ci, t in enumerate(tree_by_comp)},
    }
