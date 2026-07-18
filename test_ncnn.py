#!/usr/bin/env python3
"""Run the ncnn RITnet pupil segmentation on ONE image — with an interactive cropper.

HEADLESS (prints + saves images, works over SSH):
    python3 test_ncnn.py                 # auto-picks the first image in the folder
    python3 test_ncnn.py myeye.jpg
    python3 test_ncnn.py myeye.jpg --size 640x400          # full-res (slow, may OOM a Pi Zero)
    python3 test_ncnn.py myeye.jpg --crop 640,560          # crop CENTRE
    python3 test_ncnn.py myeye.jpg --region 480,360,800,760  # explicit crop rectangle

VISUAL (needs a desktop / display — run it locally on the Pi):
    python3 test_ncnn.py myeye.jpg --gui
        arrow keys  move the crop box
        + / -       grow / shrink the crop box
        ENTER/space run RITnet on the current crop (overlays pupil + shows segmentation)
        r           reset the crop to the reflex anchor
        s           save the current overlay + stages
        q / ESC     quit

The ncnn RITnet graph is shape-flexible: the one model (ritnet.param/.bin) runs at any
input that is a multiple of 16.  The chosen crop is resized to ``--size`` before the net
(default 320x192 — fits a Pi Zero's RAM and is ~4x faster than 640x400, at some accuracy
cost).  Class map colours: black=bg, grey=sclera, orange=iris, red=pupil.
"""

import argparse
import glob
import os
import time

try:
    import resource
except ImportError:
    resource = None

import cv2
import numpy as np
import ncnn

import ncnn_infer as ri

HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Arrow-key codes from cv2.waitKeyEx (Linux/GTK on the Pi + Windows fallbacks).
K_LEFT = {65361, 2424832}
K_RIGHT = {65363, 2555904}
K_UP = {65362, 2490368}
K_DOWN = {65364, 2621440}


def _resolve(name):
    for d in (HERE, os.path.join(HERE, "models", "ritnet")):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _parse_size(s):
    try:
        w, h = (int(v) for v in s.lower().split("x"))
    except ValueError:
        raise SystemExit(f"bad --size {s!r}; use e.g. 320x192")
    return max(16, (w // 16) * 16), max(16, (h // 16) * 16)


def _find_image(arg):
    if arg:
        return arg if os.path.isfile(arg) else None
    cands = []
    for e in _IMG_EXTS:
        cands += glob.glob(os.path.join(HERE, "*" + e))
    cands = [c for c in sorted(cands)
             if "_ncnn_" not in os.path.basename(c)
             and not os.path.basename(c).startswith("_")]
    return cands[0] if cands else None


def load_net(param):
    binp = param[:-6] + ".bin"
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.lightmode = True
    net.load_param(param)
    net.load_model(binp)
    ri._auto_names(param)
    return net, ri._INPUT_NAME, ri._OUTPUT_NAME


def infer(green, rect, size, net, names, anchor):
    """Run RITnet on the crop rectangle `rect`=(x0,y0,x1,y1); resized to `size`=(W,H).

    Returns a dict with center/radius/conf, the full-frame pupil mask `pm`, the
    network-resolution colour class map `seg`, the fitted `ellipse`, the clamped
    `rect`, and `infer_s` (seconds).  None if the rect is empty.
    """
    inn, outn = names
    W, H = size
    h, w = green.shape[:2]
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
    src = ri.inpaint_reflex(green, None)
    crop = src[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    vh, vw = crop.shape[:2]
    net_in = ri.normalize(crop)
    feed = net_in if (vh, vw) == (H, W) else cv2.resize(net_in, (W, H), interpolation=cv2.INTER_AREA)

    t = time.time()
    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(feed),
                               ncnn.Mat.PixelType.PIXEL_GRAY, W, H)
    mat.substract_mean_normalize([127.5], [1.0 / 127.5])
    ex = net.create_extractor()
    ex.input(inn, mat)
    _, out = ex.extract(outn)
    logits = np.array(out)
    infer_s = time.time() - t

    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    prob = e / e.sum(axis=0, keepdims=True)
    cls = prob.argmax(axis=0).astype(np.uint8)
    pupp = prob[ri.PUPIL_CLASS]
    seg = ri._CLASS_COLORS[cls].copy()                    # net-resolution class map
    if (cls.shape[1], cls.shape[0]) != (vw, vh):
        cls = cv2.resize(cls, (vw, vh), interpolation=cv2.INTER_NEAREST)
        pupp = cv2.resize(pupp, (vw, vh), interpolation=cv2.INTER_LINEAR)
    pm = np.zeros((h, w), np.uint8)
    pm[y0:y0 + vh, x0:x0 + vw] = (cls == ri.PUPIL_CLASS).astype(np.uint8) * 255
    pp = np.zeros((h, w), np.float32)
    pp[y0:y0 + vh, x0:x0 + vw] = pupp

    cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = [c for c in cnts if cv2.contourArea(c) >= 60 and len(c) >= 5]
    center = radius = ellipse = None
    conf = 0.0
    if cand:
        c = ri._select_contour(cand, anchor)
        ellipse = cv2.fitEllipse(c)
        (ex2, ey2), (MA, ma), _ = ellipse
        center = (float(ex2), float(ey2))
        radius = float((MA + ma) / 4.0)
        sel = np.zeros((h, w), np.uint8)
        cv2.drawContours(sel, [c], -1, 255, -1)
        vals = pp[sel > 0]
        conf = float(vals.mean()) if vals.size else 0.0
    return dict(center=center, radius=radius, conf=conf, pm=pm, seg=seg,
                ellipse=ellipse, rect=(x0, y0, x0 + vw, y0 + vh), feed=feed, infer_s=infer_s)


def _reflex_anchor(gray):
    try:
        import cap
        rr = cap._find_corneal_reflex(gray)
        if rr is not None:
            return (int(rr[0]), int(rr[1]))
    except Exception:                        # noqa: BLE001
        pass
    return None


def _save_stages(stem, res, green):
    final = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = res["rect"]
    cv2.rectangle(final, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if res["ellipse"] is not None:
        cv2.ellipse(final, res["ellipse"], (0, 0, 255), 2)
        cx, cy = res["center"]
        cv2.circle(final, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(HERE, f"{stem}_ncnn_netinput.png"), res["feed"])
    cv2.imwrite(os.path.join(HERE, f"{stem}_ncnn_seg.png"), res["seg"])
    cv2.imwrite(os.path.join(HERE, f"{stem}_ncnn_mask.png"), res["pm"])
    cv2.imwrite(os.path.join(HERE, f"{stem}_ncnn_final.png"), final)
    print(f"saved: {stem}_ncnn_netinput.png / _seg.png / _mask.png / _final.png")


# ────────────────────────────── headless ──────────────────────────────────
def run_headless(green, gray, rect, size, net, names, stem):
    anchor = _reflex_anchor(gray)
    print("running inference...", flush=True)
    res = infer(green, rect, size, net, names, anchor)
    if res is None:
        print("ERROR: empty crop region"); return 1
    peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            if resource else float("nan"))
    print("\n=== RESULT ===")
    print(f"inference:   {res['infer_s']:.2f} s")
    print(f"peak memory: {peak:.0f} MB")
    c = res["center"]
    c = (round(c[0], 1), round(c[1], 1)) if c else None
    print(f"pupil:       center={c}  radius={round(res['radius'],1) if res['radius'] else None}"
          f"  confidence={res['conf']:.2f}" + ("" if c else "   (no pupil found)"))
    _save_stages(stem, res, green)
    return 0


# ──────────────────────────────── GUI ─────────────────────────────────────
def run_gui(green, gray, size, net, names, stem):
    h, w = green.shape[:2]
    anchor = _reflex_anchor(gray)
    cx, cy = anchor if anchor else (w // 2, h // 2)
    scale = 1.0                                          # crop = 640x400 * scale
    disp_scale = min(1.0, 1000.0 / max(h, w))
    base_gray = cv2.convertScaleAbs(cv2.cvtColor(green, cv2.COLOR_GRAY2BGR), alpha=1.6)
    win = "ncnn RITnet cropper"
    seg_win = "segmentation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("GUI: arrows=move  +/-=size  ENTER=run  r=reset  s=save  q=quit", flush=True)
    res = None
    step = 20
    while True:
        cw = max(64, int(ri.NET_W * scale))
        ch = max(40, int(ri.NET_H * scale))
        x0 = int(np.clip(cx - cw // 2, 0, max(0, w - cw)))
        y0 = int(np.clip(cy - ch // 2, 0, max(0, h - ch)))
        view = base_gray.copy()
        if res is not None and res["pm"] is not None:    # tint last pupil mask
            view[res["pm"] > 0] = (0, 0, 255)
        cv2.rectangle(view, (x0, y0), (x0 + cw, y0 + ch), (0, 255, 0), 2)
        cv2.circle(view, (cx, cy), 4, (0, 255, 255), -1)
        if res is not None and res["ellipse"] is not None:
            cv2.ellipse(view, res["ellipse"], (0, 165, 255), 2)
        msg = f"crop ({cx},{cy}) {cw}x{ch}  net {size[0]}x{size[1]}"
        if res is not None:
            cc = res["center"]
            msg += (f"  ->  pupil {('(%d,%d)' % (cc[0], cc[1])) if cc else 'NONE'}"
                    f" conf {res['conf']:.2f}  {res['infer_s']:.1f}s")
        cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        disp = cv2.resize(view, (int(w * disp_scale), int(h * disp_scale)))
        cv2.imshow(win, disp)

        k = cv2.waitKeyEx(20)
        if k in (ord('q'), 27):
            break
        elif k in K_LEFT:
            cx = max(0, cx - step)
        elif k in K_RIGHT:
            cx = min(w, cx + step)
        elif k in K_UP:
            cy = max(0, cy - step)
        elif k in K_DOWN:
            cy = min(h, cy + step)
        elif k in (ord('+'), ord('=')):
            scale = min(2.5, scale + 0.1)
        elif k in (ord('-'), ord('_')):
            scale = max(0.3, scale - 0.1)
        elif k in (ord('r'),):
            cx, cy = anchor if anchor else (w // 2, h // 2)
            scale = 1.0
        elif k in (13, 10, 32):                          # ENTER / space -> run
            print("running inference...", flush=True)
            res = infer(green, (x0, y0, x0 + cw, y0 + ch), size, net, names, anchor)
            if res is not None:
                cc = res["center"]
                print(f"  pupil={('(%.0f,%.0f)' % cc) if cc else 'NONE'} "
                      f"conf={res['conf']:.2f} {res['infer_s']:.1f}s", flush=True)
                cv2.imshow(seg_win, res["seg"])
        elif k in (ord('s'),) and res is not None:
            _save_stages(stem, res, green)
    cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Run ncnn RITnet on one image (headless or --gui).")
    ap.add_argument("image", nargs="?", help="eye image (default: first in this folder)")
    ap.add_argument("--size", default="320x192", help="network input WxH, multiple of 16")
    ap.add_argument("--model", default=None, help="ncnn .param (default: ritnet.param)")
    ap.add_argument("--crop", default=None, help="crop CENTRE CX,CY (headless)")
    ap.add_argument("--region", default=None, help="crop rectangle X0,Y0,X1,Y1 (headless)")
    ap.add_argument("--gui", action="store_true", help="interactive arrow-key cropper")
    args = ap.parse_args()

    param = args.model or _resolve("ritnet.param") or _resolve("ritnet_320.param")
    if not param or not os.path.isfile(param) or not os.path.isfile(param[:-6] + ".bin"):
        print("ERROR: no ncnn model found. Copy ritnet.param + ritnet.bin next to this script.")
        return 1
    size = _parse_size(args.size)

    img_path = _find_image(args.image)
    if not img_path:
        print("ERROR: no image. Put one here or pass it: python3 test_ncnn.py myeye.jpg")
        return 1
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: could not read image: {img_path}")
        return 1
    green = bgr[:, :, 1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = green.shape[:2]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    print(f"model {os.path.basename(param)}  net {size[0]}x{size[1]}  image {os.path.basename(img_path)} {w}x{h}")

    net, in_name, out_name = load_net(param)
    names = (in_name, out_name)

    if args.gui:
        return run_gui(green, gray, size, net, names, stem)

    # headless: build the crop rect from --region / --crop / reflex anchor / centre
    if args.region:
        try:
            x0, y0, x1, y1 = (int(v) for v in args.region.split(","))
        except ValueError:
            print("ERROR: bad --region; use X0,Y0,X1,Y1"); return 1
        rect = (x0, y0, x1, y1)
        print(f"crop region: {rect}")
    else:
        if args.crop:
            try:
                ccx, ccy = (int(v) for v in args.crop.split(","))
            except ValueError:
                print("ERROR: bad --crop; use CX,CY"); return 1
            note = "manual centre"
        else:
            a = _reflex_anchor(gray)
            ccx, ccy = a if a else (w // 2, h // 2)
            note = "reflex anchor" if a else "image centre"
        rect = (ccx - ri.NET_W // 2, ccy - ri.NET_H // 2,
                ccx + ri.NET_W // 2, ccy + ri.NET_H // 2)
        print(f"crop centre: ({ccx},{ccy})  {ri.NET_W}x{ri.NET_H} window  ({note})")
    return run_headless(green, gray, rect, size, net, names, stem)


if __name__ == "__main__":
    raise SystemExit(main())
