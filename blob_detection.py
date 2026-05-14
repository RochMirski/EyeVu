import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ── Tunable constants ─────────────────────────────────────────────────────────
STEP           = 4      # downsample factor for 3D plots
LOCAL_KERNEL   = 50     # morphological neighbourhood for local-extrema search
MIN_CONTRAST   = 0.10   # min (rim-pit)/range to accept a volcano
MAX_RIM_R      = 220    # max rim radius searched (px)
N_ANGLES       = 180    # angular samples per radial ring
RING_SEP_RATIO     = 1.3    # rim_r must be ≥ this × pit_r for a valid two-ring volcano
CENTRE_TOL         = 55     # px — max centre distance for a blue/red match
RING_SIZE_TOL      = 0.50   # fractional tolerance on pit_r / rim_r agreement
SPECULAR_MASK_PCT      = 99.0  # top-N% of max-channel used to locate specular region
SPECULAR_INPAINT_R     = 7    # inpaint radius (px) for reflection removal

PIT_COL  = (0,   255, 255)   # yellow  — pit edge
RIM_COL  = (255,   0, 255)   # magenta — rim
PEAK_COL = (0,   165, 255)   # orange  — peak-volcano

# IMG_H / IMG_W are updated at the start of each image iteration
global IMG_H, IMG_W
IMG_H = IMG_W = 0

# ── Helper functions ─────────────────────────────────────────────────────────

def radial_profile(smooth, cx, cy, max_r):
    """Mean intensity at each integer radius 0..max_r from (cx, cy)."""
    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    prof = np.zeros(max_r, dtype=np.float32)
    for r in range(max_r):
        xs = np.clip((cx + r * cos_a).astype(int), 0, smooth.shape[1] - 1)
        ys = np.clip((cy + r * sin_a).astype(int), 0, smooth.shape[0] - 1)
        prof[r] = smooth[ys, xs].mean()
    return prof


def find_volcanos(channel_u8, ch_name):
    """Find pit- and peak-volcanos in a single-channel uint8 image."""
    smooth   = cv2.GaussianBlur(channel_u8.astype(np.float32), (31, 31), 0)
    ch_range = float(smooth.max() - smooth.min())
    if ch_range == 0:
        return []

    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (LOCAL_KERNEL, LOCAL_KERNEL))
    eroded       = cv2.erode(smooth,  kernel)
    dilated      = cv2.dilate(smooth, kernel)
    contrast_map = dilated - eroded
    local_min    = ((smooth - eroded)  < 1.5) & (contrast_map > ch_range * MIN_CONTRAST)
    local_max    = ((dilated - smooth) < 1.5) & (contrast_map > ch_range * MIN_CONTRAST)

    results = []
    for mask, kind in ((local_min.astype(np.uint8), 'pit'),
                       (local_max.astype(np.uint8), 'peak')):
        n_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask)
        for lbl in range(1, n_labels):
            cx, cy  = int(centroids[lbl][0]), int(centroids[lbl][1])
            safe_r  = min(cx, cy, IMG_W - cx - 1, IMG_H - cy - 1, MAX_RIM_R)
            if safe_r < 20:
                continue
            prof       = radial_profile(smooth, cx, cy, safe_r)
            center_val = prof[0]
            search_end = min(safe_r - 1, MAX_RIM_R)

            if kind == 'pit':
                rim_idx  = int(np.argmax(prof[5:search_end])) + 5
                contrast = (prof[rim_idx] - center_val) / ch_range
                if contrast < MIN_CONTRAST:
                    continue
                thresh = center_val + 0.2 * (prof[rim_idx] - center_val)
                above  = np.where(prof[1:rim_idx] > thresh)[0]
                pit_r  = int(above[0]) + 1 if len(above) else 5
                results.append(dict(cx=cx, cy=cy, kind='pit',
                                    pit_r=pit_r, rim_r=rim_idx,
                                    contrast=contrast, profile=prof))
            else:
                moat_idx = int(np.argmin(prof[5:search_end])) + 5
                drop     = (center_val - prof[moat_idx]) / ch_range
                if drop < MIN_CONTRAST or moat_idx + 5 >= search_end:
                    continue
                outer_idx = int(np.argmax(prof[moat_idx:search_end])) + moat_idx
                rise      = (prof[outer_idx] - prof[moat_idx]) / ch_range
                if rise < MIN_CONTRAST * 0.5:
                    continue
                results.append(dict(cx=cx, cy=cy, kind='peak',
                                    pit_r=moat_idx, rim_r=outer_idx,
                                    contrast=drop, profile=prof))

    results.sort(key=lambda v: -v['contrast'])
    unique = []
    for v in results:
        if not any(np.hypot(v['cx'] - u['cx'], v['cy'] - u['cy']) < 30 for u in unique):
            unique.append(v)

    print(f"  {ch_name}: {len(unique)} volcano(s)")
    for i, v in enumerate(unique):
        print(f"    {i+1}. {v['kind']:4s} ({v['cx']},{v['cy']})  "
              f"pit_r={v['pit_r']}  rim_r={v['rim_r']}  contrast={v['contrast']:.2f}")
    return unique


def draw_volcanos(base_img, volcanos, title):
    canvas = base_img.copy()
    for v in volcanos:
        cx, cy  = v['cx'], v['cy']
        pit_col = PIT_COL if v['kind'] == 'pit' else PEAK_COL
        cv2.circle(canvas, (cx, cy), v['pit_r'], pit_col, 2)
        cv2.circle(canvas, (cx, cy), v['rim_r'], RIM_COL,  2)
        cv2.circle(canvas, (cx, cy), 4,          pit_col, -1)
        cv2.putText(canvas, f"{v['kind']} c={v['contrast']:.2f}",
                    (cx - 40, cy - v['rim_r'] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, RIM_COL, 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')


def has_two_rings(v):
    return v['pit_r'] > 0 and v['rim_r'] > v['pit_r'] * RING_SEP_RATIO


# ── Main loop — process every image in the Images folder ─────────────────────

IMAGES_DIR  = 'Images'
EXTENSIONS  = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
image_files = sorted(f for f in os.listdir(IMAGES_DIR)
                     if f.lower().endswith(EXTENSIONS))

if not image_files:
    print(f"No images found in '{IMAGES_DIR}'.")

for fname in image_files:
    image_path = os.path.join(IMAGES_DIR, fname)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {fname}, skipping.")
        continue

   
    IMG_H, IMG_W = img.shape[:2]
    print(f"\n{'='*60}\n{fname}  ({IMG_W}×{IMG_H})\n{'='*60}")

    # ── Remove specular / Purkinje reflection via inpainting ─────────────────
    # Build a mask of the top-N% brightest pixels (max across channels),
    # dilate slightly to catch jagged edges, then inpaint.
    spec_max  = np.max(img, axis=2)
    thresh    = np.percentile(spec_max, SPECULAR_MASK_PCT)
    spec_mask = (spec_max > thresh).astype(np.uint8)
    spec_mask = cv2.dilate(spec_mask,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img = cv2.inpaint(img, spec_mask, SPECULAR_INPAINT_R, cv2.INPAINT_TELEA)

    # ── 3D surface plots ──────────────────────────────────────────────────────
    img_ds = img[::STEP, ::STEP]
    h, w   = img_ds.shape[:2]
    X, Y   = np.meshgrid(np.arange(w), np.arange(h))

    channel_names = ['Blue', 'Green', 'Red']
    channel_cmaps = ['Blues', 'Greens', 'Reds']

    fig_rgb, axes_rgb = plt.subplots(1, 3, figsize=(18, 6),
                                      subplot_kw={'projection': '3d'})
    fig_rgb.suptitle(f'RGB channels (inpainted) — {fname}', fontsize=13)
    for idx, (name, cmap) in enumerate(zip(channel_names, channel_cmaps)):
        ch = img_ds[:, :, idx].astype(np.float32) / 255.0
        ax = axes_rgb[idx]
        ax.plot_surface(X, Y, ch, cmap=cmap, linewidth=0, antialiased=False)
        ax.set_title(name)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Intensity')
        ax.invert_yaxis()
    plt.tight_layout()

    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_norm = gray[::STEP, ::STEP].astype(np.float32) / 255.0
    fig_bw    = plt.figure(figsize=(10, 7))
    ax_bw     = fig_bw.add_subplot(111, projection='3d')
    surf      = ax_bw.plot_surface(X, Y, gray_norm, cmap='gray',
                                    linewidth=0, antialiased=False)
    fig_bw.colorbar(surf, ax=ax_bw, shrink=0.5, label='Intensity')
    ax_bw.set_title(f'Grayscale (inpainted) — {fname}')
    ax_bw.set_xlabel('X'); ax_bw.set_ylabel('Y'); ax_bw.set_zlabel('Intensity')
    ax_bw.invert_yaxis()
    plt.tight_layout()
    plt.show()   # blocks — close both 3D windows to continue
    plt.close('all')  # release matplotlib before cv2 takes the event loop

    # ── Volcano detection ─────────────────────────────────────────────────────
    blue_volcanos = find_volcanos(img[:, :, 0], 'Blue')
    red_volcanos  = find_volcanos(img[:, :, 2], 'Red')

    draw_volcanos(img, blue_volcanos,
                  f'Blue volcanos — {fname}  (any key = next)')
    draw_volcanos(img, red_volcanos,
                  f'Red volcanos  — {fname}  (any key = next)')

    # ── Match blue & red ──────────────────────────────────────────────────────
    # Only peak-type volcanos produce the orange+magenta ring pair
    blue_2ring = [v for v in blue_volcanos if has_two_rings(v) and v['kind'] == 'peak']
    red_2ring  = [v for v in red_volcanos  if has_two_rings(v) and v['kind'] == 'peak']

    matches = []
    for bv in blue_2ring:
        for rv in red_2ring:
            if np.hypot(bv['cx'] - rv['cx'], bv['cy'] - rv['cy']) > CENTRE_TOL:
                continue
            for attr in ('pit_r', 'rim_r'):
                mean_r = (bv[attr] + rv[attr]) / 2
                if mean_r == 0 or abs(bv[attr] - rv[attr]) / mean_r > RING_SIZE_TOL:
                    break
            else:
                matches.append((bv, rv))

    print(f"  Matches: {len(matches)}")
    for i, (bv, rv) in enumerate(matches):
        print(f"    {i+1}. blue=({bv['cx']},{bv['cy']})  red=({rv['cx']},{rv['cy']})"
              f"  pit~{(bv['pit_r']+rv['pit_r'])//2}  rim~{(bv['rim_r']+rv['rim_r'])//2}")

    if matches:
        canvas = img.copy()
        for bv, rv in matches:
            cx  = (bv['cx']    + rv['cx'])    // 2
            cy  = (bv['cy']    + rv['cy'])    // 2
            p_r = (bv['pit_r'] + rv['pit_r']) // 2
            r_r = (bv['rim_r'] + rv['rim_r']) // 2
            cv2.circle(canvas, (bv['cx'], bv['cy']), bv['pit_r'], (255, 100,   0), 1)
            cv2.circle(canvas, (bv['cx'], bv['cy']), bv['rim_r'], (255, 100,   0), 1)
            cv2.circle(canvas, (rv['cx'], rv['cy']), rv['pit_r'], (  0,  80, 255), 1)
            cv2.circle(canvas, (rv['cx'], rv['cy']), rv['rim_r'], (  0,  80, 255), 1)
            cv2.circle(canvas, (cx, cy), p_r, (255, 255, 255), 2)
            cv2.circle(canvas, (cx, cy), r_r, (255, 255, 255), 2)
            cv2.circle(canvas, (cx, cy), 5,   (255, 255, 255), -1)
            cv2.putText(canvas, f"match {p_r}/{r_r}px",
                        (cx - 40, cy - r_r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Matched volcanos — {fname}')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close('all')
    else:
        print("  No blue/red matches — falling back to Green channel.")
        green_volcanos = find_volcanos(img[:, :, 1], 'Green')
        # Show only peak-type (orange+magenta) green volcanos directly
        green_peaks = [v for v in green_volcanos
                       if has_two_rings(v) and v['kind'] == 'peak']
        draw_volcanos(img, green_peaks,
                      f'Green peaks (fallback) — {fname}')

print("\nAll images processed.")
