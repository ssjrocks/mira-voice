"""Fixed-shape audio models for Qualcomm's NPU (QNN HTP needs static shapes and rejects some ops).

vocoder_npu.onnx    features [1, 1024, 200] -> [1, 1, 64000]    (one 4 s window, context frames included)
upsampler_npu.onnx  wav16 [1, 1, 16640] -> [1, 1, 49920]        (one 1 s window + 320 samples context each side)

The window sizes are the largest that fit the S23 NPU's 8 MB on-chip memory: QNN 2.51 keeps a whole row of
each activation there (width x 128 bytes), and refuses the graph ("Requires ... bytes of TCM") above that.

Both use x*x instead of Pow(x, 2) (the NPU computes Pow as exp(y*log x), wrong for negative x). The upsampler's
weight-norm/Expand weights are constant-folded and its depthwise convolutions (rejected by the NPU) become ordinary
convolutions with block-diagonal weights, which is the same maths. Compile them with qnn_context.py.

usage: export_npu.py                 both models
       export_npu.py --up-seconds N  only the upsampler, N s window, as upsampler_npu<N>s.onnx
       export_npu.py --voc-frames N  only the vocoder, N frames (50 per second), as vocoder_npu<N>f.onnx
"""
import argparse
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import common as C

B = C.BUILD
ap = argparse.ArgumentParser()
ap.add_argument("--up-seconds", type=int)
ap.add_argument("--voc-frames", type=int)
args = ap.parse_args()
DO_VOC, DO_UP = not args.up_seconds, not args.voc_frames
VOC_FRAMES = args.voc_frames or 200
VOC_NAME = f"vocoder_npu{args.voc_frames}f" if args.voc_frames else "vocoder_npu"
UP_SAMPLES = (args.up_seconds or 1) * 16000 + 2 * 320
UP_NAME = f"upsampler_npu{args.up_seconds}s" if args.up_seconds else "upsampler_npu"


def fix_shape(src, dst, name, shape):
    subprocess.run([sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed", "--input_name", name,
                    "--input_shape", ",".join(map(str, shape)), src, dst], check=True)


def fold(src, dst):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


def densify_depthwise(path):
    """Depthwise Conv (group == channels) -> ordinary Conv with a block-diagonal weight. The NPU's validator rejects
    these 12-tap depthwise convolutions (error 3110), strided or not; ordinary convolutions run fine there."""
    m = onnx.load(path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    n_fixed = 0
    for n in g.node:
        at = {a.name: a for a in n.attribute}
        w = inits.get(n.input[1]) if n.op_type == "Conv" and len(n.input) > 1 else None
        if w is None or "group" not in at or at["group"].i <= 1 or at["group"].i != w.dims[0] or w.dims[1] != 1:
            continue
        wd = numpy_helper.to_array(w)
        dense = np.zeros((wd.shape[0], wd.shape[0], wd.shape[2]), wd.dtype)
        for c in range(wd.shape[0]):
            dense[c, c] = wd[c, 0]
        name = f"{n.name}_dense_w"
        g.initializer.append(numpy_helper.from_array(dense, name))
        n.input[1] = name
        at["group"].i = 1
        n_fixed += 1
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return n_fixed


def check(name, ref_model, new_model, inp, x):
    a = ort.InferenceSession(ref_model, providers=["CPUExecutionProvider"]).run(None, {inp: x})[0]
    b = ort.InferenceSession(new_model, providers=["CPUExecutionProvider"]).run(None, {inp: x})[0]
    print(f"{name}: out {b.shape}, max diff vs original {np.abs(a - b).max():.2e}")
    x.astype(np.float32).tofile(new_model.replace(".onnx", "_in.f32"))
    a.astype(np.float32).tofile(new_model.replace(".onnx", "_out.f32"))


if DO_VOC:
    fix_shape(f"{B}/vocoder_m.onnx", f"{B}/{VOC_NAME}.onnx", "features", [1, 1024, VOC_FRAMES])
if DO_UP:
    fix_shape(f"{B}/upsampler_m.onnx", f"{B}/upsampler_npu_tmp.onnx", "wav16", [1, 1, UP_SAMPLES])
    fold(f"{B}/upsampler_npu_tmp.onnx", f"{B}/{UP_NAME}.onnx")
    print("upsampler: depthwise convs made dense:", densify_depthwise(f"{B}/{UP_NAME}.onnx"))

import json
ref = json.load(open(f"{B}/ref/ex1.json"))
codes = np.array([ref["codes"]], np.int64)
proc = ort.InferenceSession(f"{B}/processer.onnx", providers=["CPUExecutionProvider"])
feats = proc.run(None, {"speech_tokens": codes, "context_tokens": np.array(ref["global_tokens"], np.int32).reshape(1, 1, 32)})[0]
f = np.zeros((1, 1024, VOC_FRAMES), np.float32)
f[:, :, :feats.shape[2]] = feats[:, :, :VOC_FRAMES]
if DO_VOC:
    check(VOC_NAME, f"{B}/vocoder.onnx", f"{B}/{VOC_NAME}.onnx", "features", f)
if DO_UP:
    w = ort.InferenceSession(f"{B}/vocoder.onnx", providers=["CPUExecutionProvider"]).run(None, {"features": feats})[0].reshape(-1)
    u = np.zeros((1, 1, UP_SAMPLES), np.float32)
    u[0, 0, :min(len(w), UP_SAMPLES)] = w[:UP_SAMPLES]
    check(UP_NAME, f"{B}/upsampler.onnx", f"{B}/{UP_NAME}.onnx", "wav16", u)
    import os
    os.remove(f"{B}/upsampler_npu_tmp.onnx")
