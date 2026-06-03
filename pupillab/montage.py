"""Montage builder — lays out every detector's debug stages into one image.

Generalises the original ``test_pupil_detection.save_debug_stages`` grid: stages
are grouped per detector under a coloured header bar, and a final comparison
panel overlays every detector's pupil on one brightened flash frame.  Because a
detector's stages come straight from its ``DetectionResult.stages``, adding a new
module makes its block appear here automatically — no montage code to touch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import context as _ctx       # ensures cap is importable; gives us cap.cv2
from .base import DetectionContext, DetectionResult

cv2 = _ctx.cap.cv2

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _caption(img, text, bar_h=22, color=(255, 255, 255), bg=(0, 0, 0)):
    """Return ``img`` with a caption bar (``text``) stacked on top."""
    w = img.shape[1]
    bar = np.full((bar_h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (4, bar_h - 6), _FONT, 0.45, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _header(text, width, bar_h=28, bg=(60, 60, 60)):
    bar = np.full((bar_h, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (6, bar_h - 8), _FONT, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    return bar


def _scale_to_width(img, cell_w):
    h, w = img.shape[:2]
    return cv2.resize(img, (cell_w, max(1, int(h * cell_w / w))))


def _grid(cells, cols, cell_w):
    """Arrange labelled, equal-width cells into a `cols`-wide grid (BGR image)."""
    if not cells:
        return np.zeros((1, cols * cell_w, 3), dtype=np.uint8)
    row_h = max(c.shape[0] for c in cells)
    padded = [cv2.copyMakeBorder(c, 0, row_h - c.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
              for c in cells]
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _comparison_panel(ctx: DetectionContext, results, cell_w, cols):
    """One brightened-flash cell per detector with that detector's pupil drawn."""
    cells = []
    for res in results:
        vis = cv2.convertScaleAbs(ctx.flash, alpha=3.0)   # flash is dark; brighten
        _ctx.draw_roi(vis, ctx)                            # search box, if any
        ov = res.overlay()
        if ov is not None:
            _ctx.cap._draw_overlays(vis, [ov])
        tag = f"{res.detector}"
        if res.pupil is not None:
            (_, _), (MA, _), _ = res.pupil
            tag += f"  r={MA / 2:.0f} conf={res.confidence:.2f}"
        elif not res.ok:
            tag += "  (unavailable)"
        else:
            tag += "  (no pupil)"
        cells.append(_caption(_scale_to_width(vis, cell_w), tag, color=(0, 255, 255)))
    return _grid(cells, cols, cell_w)


def build_montage(ctx: DetectionContext, results, cell_w: int = 320,
                  cols: int = 3) -> np.ndarray:
    """Build the full grouped montage (BGR image) for a list of DetectionResults."""
    width = cols * cell_w
    blocks = []
    for res in results:
        header = f"{res.detector}"
        if res.notes:
            header += f"  -  {res.notes}"      # ASCII only: cv2 font has no em-dash
        blocks.append(_header(header, width))
        cells = [_caption(_scale_to_width(img, cell_w), name)
                 for name, img in res.stages]
        if cells:
            blocks.append(_grid(cells, cols, cell_w))
        else:
            blocks.append(_header("(no debug stages)", width, bg=(30, 30, 30)))

    blocks.append(_header("COMPARISON - pupil overlay on flash", width,
                          bar_h=30, bg=(20, 70, 20)))
    blocks.append(_comparison_panel(ctx, results, cell_w, cols))
    return np.vstack(blocks)


def save_montage(folder: str, ctx: DetectionContext, results,
                 filename: str = "debug_montage.jpg") -> str:
    """Write the montage into ``folder``; return the path."""
    import os
    montage = build_montage(ctx, results)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, montage)
    return path


def best_result(results) -> Optional[DetectionResult]:
    """Pick the result to drive annotated.jpg: highest-confidence valid pupil.

    Ties / no-confidence fall back to the first detector that found a pupil
    (registration order puts the proven baseline first)."""
    found = [r for r in results if r.ok and r.pupil is not None]
    if not found:
        return None
    return max(found, key=lambda r: (r.confidence, ))
