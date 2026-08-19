#!/usr/bin/env python3
"""Unit tests for mosaic.py (no hardware, no torch needed).

Synthetic "captures": elliptical patches cut from one shared texture at known
offsets, dropped into full frames the way a red-eye extract is.  Because the
ground-truth geometry is known, registration can be checked for ACCURACY rather
than just for having produced something.

Run: python test_mosaic.py
"""

import cv2
import numpy as np

import mosaic

FRAME_H, FRAME_W = 640, 480
PATCH = 70                      # side of the square cut from the texture
PATCH_R = 34                    # radius of the visible disc inside it
ORIGIN = (200, 250)             # where the patch sits in the frame (x, y)
CENTRE = (ORIGIN[0] + PATCH // 2, ORIGIN[1] + PATCH // 2)


def _texture(size=420, seed=7):
    """A low-contrast blotchy field with vessel-like strokes — fundus-ish."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.normal(90, 12, (size, size)).astype(np.float32),
                           (0, 0), 3)
    for _ in range(60):
        p = rng.integers(20, size - 20, 4)
        cv2.line(tex, (p[0], p[1]), (p[2], p[3]), float(rng.uniform(30, 70)),
                 int(rng.integers(1, 3)), cv2.LINE_AA)
    for _ in range(120):
        c = rng.integers(10, size - 10, 2)
        cv2.circle(tex, (int(c[0]), int(c[1])), int(rng.integers(2, 7)),
                   float(rng.uniform(40, 150)), -1, cv2.LINE_AA)
    tex = np.clip(cv2.GaussianBlur(tex, (0, 0), 1.0), 0, 255).astype(np.uint8)
    return cv2.merge([tex // 4, tex // 4, tex])          # reddish, like an extract


_BIG = _texture()


def _capture(cx, cy, ang=0.0, scale=1.0):
    """One capture centred on texture point (cx, cy).  Returns (frame, mask)."""
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, scale)
    M[0, 2] += PATCH / 2 - cx
    M[1, 2] += PATCH / 2 - cy
    crop = cv2.warpAffine(_BIG, M, (PATCH, PATCH))
    disc = np.zeros((PATCH, PATCH), np.uint8)
    cv2.circle(disc, (PATCH // 2, PATCH // 2), PATCH_R, 255, -1)
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
    ox, oy = ORIGIN
    frame[oy:oy + PATCH, ox:ox + PATCH][disc > 0] = crop[disc > 0]
    mask[oy:oy + PATCH, ox:ox + PATCH] = disc
    return frame, mask


# A chain of overlapping views, most central capture in the middle.  Only
# neighbours overlap, so nothing can be stitched by matching everything to the
# first capture — the graph has to walk the chain.
CHAIN = [(243, 208, 0, 1.0), (222, 212, 4, 1.0), (204, 216, -3, 1.03),
         (186, 220, 6, 0.98), (166, 224, 2, 1.0)]


def _session(extra=()):
    imgs, masks = [], []
    for cx, cy, a, s in list(CHAIN) + list(extra):
        f, m = _capture(cx, cy, a, s)
        imgs.append(f)
        masks.append(m)
    return imgs, masks


def _noise_capture():
    """A patch of pure noise: matches nothing, must never be placed."""
    rng = np.random.default_rng(11)
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
    cv2.circle(mask, (300, 300), 30, 255, -1)
    frame[mask > 0] = rng.integers(0, 255, (int((mask > 0).sum()), 3), dtype=np.uint8)
    return frame, mask


# ── the geometric sanity check on a single transform ──

def test_sane_accepts_identity():
    assert mosaic._sane(np.eye(3), (10, 10, 40, 30))


def test_sane_rejects_collapse():
    tiny = np.diag([0.02, 0.02, 1.0])
    assert not mosaic._sane(tiny, (10, 10, 40, 30))


def test_sane_rejects_mirror():
    flip = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert not mosaic._sane(flip, (10, 10, 40, 30))


def test_sane_divides_out_the_expected_scale():
    # The rescue pass matches against an upscaled canvas, so a 3x transform is
    # correct there and a mis-registration only if the scale is NOT divided out.
    up = np.diag([3.0, 3.0, 1.0])
    assert mosaic._sane(up, (10, 10, 40, 30), scale=3.0)


# ── photometric verification: the test that actually rejects false matches ──

def test_verify_accepts_a_true_alignment():
    det, _, _ = mosaic._detector()
    f, m = _capture(*CHAIN[2])
    fa = mosaic._feat(det, f, m)
    ok, ncc, ov = mosaic._verify(fa, fa, np.eye(3))
    assert ok and ncc > 0.99, (ok, ncc, ov)


def test_verify_rejects_unrelated_patches():
    det, _, _ = mosaic._detector()
    fa = mosaic._feat(det, *_capture(*CHAIN[0]))
    nf, nm = _noise_capture()
    fb = mosaic._feat(det, nf, nm)
    ok, ncc, _ = mosaic._verify(fa, fb, np.eye(3))
    assert not ok, ncc


# ── seed choice ──

def test_seed_is_the_best_connected_capture():
    # A three-node chain 0-1-2: only the middle one touches both others, so it is
    # the only sane place to build from.
    pairs = {(0, 1): {"inliers": 9}, (1, 0): {"inliers": 9},
             (1, 2): {"inliers": 7}, (2, 1): {"inliers": 7}}
    seed, comp = mosaic.choose_seed([0, 1, 2], pairs, verbose=False)
    assert seed == 1, seed
    assert comp == {0, 1, 2}, comp


def test_seed_prefers_the_largest_group():
    # A tight pair (0-1, strong) versus a loose triple (2-3-4, weak).  More
    # captures in the mosaic beats a better-matched pair.
    pairs = {}
    for a, b, n in ((0, 1, 40), (2, 3, 6), (3, 4, 6)):
        pairs[(a, b)] = {"inliers": n}
        pairs[(b, a)] = {"inliers": n}
    seed, comp = mosaic.choose_seed([0, 1, 2, 3, 4], pairs, verbose=False)
    assert comp == {2, 3, 4}, comp
    assert seed == 3, seed          # the middle of the triple


# ── stitching end to end ──

def test_stitch_places_more_than_two():
    imgs, masks = _session()
    mos, info = mosaic.stitch(imgs, masks, verbose=False)
    assert mos is not None
    assert info["used"] >= 4, (info["used"], info["skipped"])


def test_stitch_grows_beyond_a_single_capture():
    imgs, masks = _session()
    mos, _ = mosaic.stitch(imgs, masks, verbose=False)
    out = mosaic.crop_to_content(mos)
    one = PATCH_R * 2 * mosaic.MOSAIC_UPSCALE
    assert max(out.shape[:2]) > one * 1.3, (out.shape, one)


def test_stitch_starts_from_a_central_capture():
    imgs, masks = _session()
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    assert info["seed"] in (1, 2, 3), info["seed"]      # never an end of the chain


def _shift(dx, dy):
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])


_PTS = np.float32([[20, 20], [60, 22], [58, 55], [22, 58],
                   [40, 30], [35, 50]]).reshape(-1, 1, 2)


def _pairing(H, inliers=9):
    """A pair record whose correspondences are exactly consistent with `H`."""
    return {"H": H, "inliers": inliers, "matches": inliers, "ncc": 0.9,
            "src": _PTS, "dst": cv2.perspectiveTransform(_PTS, H)}


def test_resection_chains_through_a_placed_neighbour():
    """A capture matching only a NON-seed placed image still gets positioned.

    This is what lets the mosaic grow past its seed's own overlaps: image 2 here
    has no link to the seed at all, only to image 1, which is already placed.
    """
    placed = {0: np.eye(3), 1: _shift(50, 0)}
    pairs = {(1, 2): _pairing(_shift(20, 5))}
    H, via, n = mosaic._resection(2, placed, pairs, (10, 10, 60, 50), verbose=False)
    assert H is not None and via == [1], (H, via)
    got = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), H).ravel()
    assert np.allclose(got, (70.0, 5.0), atol=0.5), got      # 50+20, 0+5


def test_resection_pools_every_matching_neighbour():
    """Matching TWO placed images fits one transform to both sets of points."""
    placed = {0: np.eye(3), 1: _shift(50, 0)}
    pairs = {(0, 2): _pairing(_shift(70, 5), inliers=7),
             (1, 2): _pairing(_shift(20, 5), inliers=11)}
    H, via, n = mosaic._resection(2, placed, pairs, (10, 10, 60, 50), verbose=False)
    assert sorted(via) == [0, 1], via
    assert n >= len(_PTS), n                     # points from both were used
    got = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), H).ravel()
    assert np.allclose(got, (70.0, 5.0), atol=0.5), got


def test_resection_outvotes_a_disagreeing_neighbour():
    """Two neighbours that contradict each other must not be averaged.

    The pooled RANSAC arbitrates: the majority evidence wins outright rather
    than the mosaic splitting the difference and placing the capture where
    neither neighbour says it belongs.
    """
    placed = {0: np.eye(3), 1: _shift(50, 0), 2: _shift(0, 40)}
    good = _shift(70, 5)
    pairs = {(0, 3): _pairing(good, inliers=9),
             (1, 3): _pairing(_shift(20, 5), inliers=9),      # same place: 50+20
             (2, 3): _pairing(_shift(300, 300), inliers=9)}   # nonsense
    H, via, n = mosaic._resection(3, placed, pairs, (10, 10, 60, 50), verbose=False)
    got = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), H).ravel()
    assert np.allclose(got, (70.0, 5.0), atol=2.0), (got, via)


def test_stitch_fits_jointly_when_several_neighbours_match():
    imgs, masks = _session()
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    joint = [p for p in info["placements"] if len(p.get("via", [])) > 1]
    assert joint, info["placements"]


def test_stitch_rejects_an_unrelated_capture():
    imgs, masks = _session()
    nf, nm = _noise_capture()
    imgs.append(nf)
    masks.append(nm)
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    assert len(imgs) - 1 not in info["order"], info["order"]


def test_stitch_registration_is_accurate():
    """Every placed capture must land where the ground-truth offsets say."""
    imgs, masks = _session()
    det, norm, _ = mosaic._detector()
    idxs = list(range(len(imgs)))
    bboxes = {i: mosaic.content_bbox(masks[i]) for i in idxs}
    feats = {i: mosaic._feat(det, imgs[i], masks[i]) for i in idxs}
    pairs = mosaic.match_graph(norm, feats, bboxes, idxs, verbose=False)
    seed, _ = mosaic.choose_seed(idxs, pairs, verbose=False)

    G = {seed: np.eye(3)}
    left = [i for i in idxs if i != seed]
    while left:
        strength = {j: sum(pairs[(i, j)]["inliers"] for i in G if (i, j) in pairs)
                    for j in left}
        j = max(left, key=lambda k: strength[k])
        if strength[j] == 0:
            break
        left.remove(j)
        H, _via, _n = mosaic._resection(j, G, pairs, bboxes[j], verbose=False)
        if H is not None:
            G[j] = H

    ctr = np.float32([[[float(CENTRE[0]), float(CENTRE[1])]]])
    sx, sy = CHAIN[seed][0], CHAIN[seed][1]
    for k in G:
        if k == seed:
            continue
        got = cv2.perspectiveTransform(ctr, G[k]).ravel()
        want = (CENTRE[0] + CHAIN[k][0] - sx, CENTRE[1] + CHAIN[k][1] - sy)
        err = float(np.hypot(got[0] - want[0], got[1] - want[1]))
        assert err < 6.0, f"capture {k} placed {err:.1f}px from truth"


# ── agglomeration: no single root, and nothing thrown away ──
# Two captures that overlap each other but nothing else must still be merged into
# their own mosaic, and that mosaic must survive to be retried against the rest.

# Two tight pairs, far enough apart that no capture matches across the gap.
# Group A reuses the chain's middle, whose overlap is known to register.
TWO_GROUPS = [(204, 216, -3, 1.03), (186, 220, 6, 0.98),     # group A
              (280, 130, 0, 1.0), (262, 134, -2, 1.0)]        # group B


def _two_group_session():
    imgs, masks = [], []
    for cx, cy, a, s in TWO_GROUPS:
        f, m = _capture(cx, cy, a, s)
        imgs.append(f)
        masks.append(m)
    return imgs, masks


def test_pieces_form_per_component_not_one_root():
    """Each overlapping group becomes its own mosaic; neither is discarded."""
    imgs, masks = _two_group_session()
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    groups = [set(p) for p in info["pieces"]]
    assert {0, 1} in groups, info["pieces"]
    assert {2, 3} in groups, info["pieces"]


def test_every_capture_survives_somewhere():
    """Nothing is dropped: a capture is either merged or still its own piece."""
    imgs, masks = _two_group_session()
    nf, nm = _noise_capture()
    imgs.append(nf)
    masks.append(nm)
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    seen = {i for p in info["pieces"] for i in p}
    assert seen == set(range(len(imgs))), (seen, info["pieces"])


def test_unmergeable_capture_is_never_placed_in_the_mosaic():
    imgs, masks = _two_group_session()
    nf, nm = _noise_capture()
    imgs.append(nf)
    masks.append(nm)
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    assert [len(imgs) - 1] in info["pieces"], "noise merged into something"


def test_passes_stop_when_nothing_more_merges():
    imgs, masks = _two_group_session()
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    assert info["passes"] >= 2, "never tried a second pass"
    assert info["passes"] <= mosaic.MAX_PASSES


def test_a_single_component_still_needs_a_confirming_pass():
    imgs, masks = _session()
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    assert info["passes"] >= 2, "stopped without confirming nothing else joins"


def test_members_are_carried_through_merges():
    imgs, masks = _two_group_session()
    _, info = mosaic.stitch(imgs, masks, verbose=False)
    flat = sorted(i for p in info["pieces"] for i in p)
    assert flat == sorted(set(flat)), "an index appears in two pieces"


def test_winner_is_rendered_at_full_upscale():
    """Passes run at native scale; the returned mosaic is upscaled once."""
    imgs, masks = _session()
    mos, info = mosaic.stitch(imgs, masks, verbose=False)
    assert info["used"] >= 4
    out = mosaic.crop_to_content(mos)
    one_patch = PATCH_R * 2
    assert max(out.shape[:2]) > one_patch * mosaic.MOSAIC_UPSCALE, out.shape


# ── keypoint visualisation ──

# ── display-only cropping ──
# A real extract is a ~45x32 px blob in a 480x640 frame (0.28% of it), so the
# viewers crop to the content.  These pin down that it is DISPLAY ONLY.

def test_patch_view_box_surrounds_the_content():
    frame, mask = _capture(243, 208)
    box = mosaic.patch_view_box(mask, frame.shape)
    x0, y0, x1, y1 = box
    bb = mosaic.content_bbox(mask)
    assert x0 <= bb[0] and y0 <= bb[1]
    assert x1 >= bb[0] + bb[2] and y1 >= bb[1] + bb[3]
    # ...and stays inside the frame.
    assert 0 <= x0 < x1 <= frame.shape[1]
    assert 0 <= y0 < y1 <= frame.shape[0]


def test_cropping_raises_the_share_of_the_view_that_is_content():
    frame, mask = _capture(243, 208)
    before = (mask > 0).mean()
    cm = mosaic.crop_box(mask, mosaic.patch_view_box(mask, frame.shape))
    after = (cm > 0).mean()
    assert after > 10 * before


def test_crop_box_and_view_box_handle_an_empty_mask():
    frame = np.zeros((640, 480, 3), np.uint8)
    empty = np.zeros((640, 480), np.uint8)
    assert mosaic.patch_view_box(empty, frame.shape) is None
    assert mosaic.crop_box(frame, None) is frame


def test_fit_for_display_enlarges_small_patches_but_caps_them():
    small = np.zeros((30, 40, 3), np.uint8)
    out = mosaic.fit_for_display(small, target=560)
    assert max(out.shape[:2]) > 300
    huge = np.zeros((4000, 3000, 3), np.uint8)
    assert max(mosaic.fit_for_display(huge, cap=900).shape[:2]) <= 900


def test_contact_sheet_cropping_does_not_change_the_canvas():
    imgs, masks = _session()
    a = mosaic.contact_sheet(imgs, cols=4, crop=False)
    b = mosaic.contact_sheet(imgs, cols=4, masks=masks)
    assert a.shape == b.shape
    # ...but fills far more of it.
    def fill(s):
        return (cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) > 8).mean()
    assert fill(b) > 3 * fill(a)


def test_display_cropping_does_not_affect_registration():
    """The whole point: viewers crop, the stitcher does not see it."""
    imgs, masks = _session()
    mos, info = mosaic.stitch(imgs, masks, verbose=False)
    # Cropping the inputs for display must not have mutated them.
    for im, m in zip(imgs, masks):
        mosaic.crop_box(im, mosaic.patch_view_box(m, im.shape))
    mos2, info2 = mosaic.stitch(imgs, masks, verbose=False)
    assert info["used"] == info2["used"]
    assert info["pieces"] == info2["pieces"]
    assert np.array_equal(mos, mos2)


def test_keypoint_view_draws_something():
    f, m = _capture(*CHAIN[2])
    vis, n = mosaic.keypoint_view(f, m)
    assert vis is not None and n > 0, n
    # Analysis space: cropped to the patch and enlarged, not the source frame.
    assert max(vis.shape[:2]) <= mosaic.ANALYSIS_SIZE + 2 * mosaic.ANALYSIS_PAD
    assert vis.shape[:2] != f.shape[:2]


def test_keypoint_view_handles_an_empty_patch():
    blank = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    vis, n = mosaic.keypoint_view(blank, np.zeros((FRAME_H, FRAME_W), np.uint8))
    assert vis is None and n == 0


def test_keypoint_sheet_labels_every_capture():
    imgs, masks = _session()
    sheet, labs = mosaic.keypoint_sheet(imgs, masks, verbose=False)
    assert sheet is not None
    assert len(labs) == len(imgs)
    assert all("kp" in s for s in labs), labs


def test_stitch_single_capture_is_returned_as_is():
    imgs, masks = _session()
    mos, info = mosaic.stitch(imgs[:1], masks[:1], verbose=False)
    assert mos is not None and info["used"] == 1


def test_stitch_handles_nothing_matching():
    nf, nm = _noise_capture()
    f0, m0 = _capture(*CHAIN[0])
    mos, info = mosaic.stitch([f0, nf], [m0, nm], verbose=False)
    assert mos is not None
    assert info["used"] == 1 and len(info["skipped"]) == 1


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
