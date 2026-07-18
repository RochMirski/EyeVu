#!/usr/bin/env python3
"""Export RITnet to ONNX for ncnn conversion (run on the PC, where torch works).

Writes ``models/ritnet/ritnet.onnx`` at a fixed 1x1x400x640 input (NET_H x NET_W),
matching ritnet_infer's native crop — no dynamic axes, which keeps the ncnn graph
simple.  Then, on any machine with ncnn's tools (see NCNN_PI_SETUP.md):

    python -m onnxsim models/ritnet/ritnet.onnx models/ritnet/ritnet-sim.onnx
    onnx2ncnn models/ritnet/ritnet-sim.onnx \
              models/ritnet/ritnet.param models/ritnet/ritnet.bin

Copy ritnet.param + ritnet.bin next to cap.py on the Pi; ncnn_infer.py loads them.
"""

import os

import torch

import ritnet_infer


def main():
    model = ritnet_infer._load_model()
    if model is None:
        print("Could not load RITnet:", ritnet_infer._load_error)
        return 1
    model.eval()

    h, w = ritnet_infer.NET_H, ritnet_infer.NET_W          # 400 x 640
    dummy = torch.zeros(1, 1, h, w, dtype=torch.float32)

    out_path = os.path.join(os.path.dirname(ritnet_infer.resolve_weights()),
                            "ritnet.onnx")
    kw = dict(input_names=["input"], output_names=["logits"],
              opset_version=11, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kw)
    except TypeError:                                       # older torch: no dynamo kw
        torch.onnx.export(model, dummy, out_path, **kw)

    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"Wrote {out_path}  ({size_kb:.1f} KB)")

    # ── Simplify (fold shape/constant nodes so onnx2ncnn emits a clean graph) ──
    sim_path = out_path.replace(".onnx", "-sim.onnx")
    try:
        import onnx
        from onnxsim import simplify
        model_simp, ok = simplify(onnx.load(out_path))
        onnx.save(model_simp, sim_path)
        print(f"Simplified -> {sim_path}  (check_ok={ok})")
    except Exception as e:                                  # noqa: BLE001
        print(f"onnxsim skipped ({e}); use {out_path} directly")
        sim_path = out_path

    # ── Verify the ONNX matches torch numerically on a random input ──
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, h, w)).astype(np.float32)
    with torch.no_grad():
        y_t = model(torch.from_numpy(x)).numpy()
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(sim_path, providers=["CPUExecutionProvider"])
        y_o = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        diff = float(np.abs(y_t - y_o).max())
        am_t = y_t.argmax(1)
        am_o = y_o.argmax(1)
        agree = float((am_t == am_o).mean()) * 100.0
        print(f"torch vs onnxruntime: max|logit diff|={diff:.2e}, "
              f"argmax agreement={agree:.3f}%")
    except Exception as e:                                  # noqa: BLE001
        print(f"onnxruntime check skipped ({e})")
    print(f"torch output shape: {tuple(y_t.shape)}  (expect (1, 4, {h}, {w}))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
