"""On-device ncnn RITnet timing + peak-memory test — run on the Pi.

Usage: python3 -u _pi_time.py [ambient_frame.jpg]
Sets ncnn memory-minimising options (no winograd/sgemm temp buffers, fp16
storage) to see whether full-res RITnet fits the Zero W's RAM.
"""
import time, glob, os, sys, resource, cv2

import ncnn, ncnn_infer, cap

if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    path = sys.argv[1]
else:
    cands = sorted(glob.glob("/tmp/eyevu_transfers/*/ambient.jpg"))
    if not cands:
        print("NO ambient frame; pass one as arg", flush=True); raise SystemExit(1)
    path = cands[-1]

g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(f"frame {path} shape={g.shape}", flush=True)
rr = cap._find_corneal_reflex(g)
anchor = (rr[0], rr[1]) if rr is not None else (g.shape[1] // 2, g.shape[0] // 2)
print(f"anchor {anchor}", flush=True)

# Build the ncnn net with memory-minimising options and inject it.
param = ncnn_infer.resolve_model()
net = ncnn.Net()
net.opt.lightmode = True                  # recycle blob memory (default)
net.opt.num_threads = 1
net.opt.use_winograd_convolution = False  # avoid large winograd temp buffers
net.opt.use_sgemm_convolution = False     # avoid im2col gemm buffer
net.opt.use_fp16_storage = True           # halve weight memory
net.load_param(param)
net.load_model(param[:-6] + ".bin")
ncnn_infer._auto_names(param)
ncnn_infer._net = net
print("net ready (mem-min opts); inferring...", flush=True)

t = time.time()
r = ncnn_infer.locate(g, crop_center=anchor)
dt = time.time() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # MB
print(f"INFER_SECONDS={dt:.1f} peak_rss_MB={peak:.0f} center={r.center} "
      f"conf={r.confidence:.2f} ok={r.ok} notes={r.notes}", flush=True)
print("DONE", flush=True)
