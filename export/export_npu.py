"""Fixed-shape audio models for Qualcomm's NPU (QNN HTP needs static shapes and rejects some ops).

vocoder_npu.onnx    features [1, 1024, 412] -> [1, 1, 131840]   (one 8 s window: 400 codes + 8/4 context)
upsampler_npu.onnx  wav16 [1, 1, 128640] -> [1, 1, 385920]      (one 8 s window + 320 samples context each side)

Both use x*x instead of Pow(x, 2) (the NPU computes Pow as exp(y*log x), wrong for negative x). The upsampler's
weight-norm/Expand weights are constant-folded and its strided depthwise convolutions (rejected by the NPU) become
stride-1 convolutions + Slice(step 2), which is the same maths.
"""
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import common as C

B = C.BUILD
VOC_FRAMES = 412
UP_SAMPLES = 8 * 16000 + 2 * 320


def fix_shape(src, dst, name, shape):
    subprocess.run([sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed", "--input_name", name,
                    "--input_shape", ",".join(map(str, shape)), src, dst], check=True)


def fold(src, dst):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


def unstride_depthwise(path):
    m = onnx.load(path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    nodes, n_fixed = [], 0
    for n in g.node:
        at = {a.name: a for a in n.attribute}
        w = inits.get(n.input[1]) if n.op_type == "Conv" and len(n.input) > 1 else None
        if (w is not None and "group" in at and at["group"].i > 1 and at["group"].i == w.dims[0]
                and "strides" in at and list(at["strides"].ints) == [2]):
            at["strides"].ints[:] = [1]
            full = n.output[0] + "_s1"
            out = n.output[0]
            n.output[0] = full
            nodes.append(n)
            consts = {k: numpy_helper.from_array(np.array(v, np.int64), f"{n.name}_{k}") for k, v in
                      [("starts", [0]), ("ends", [2 ** 31 - 1]), ("axes", [2]), ("steps", [2])]}
            g.initializer.extend(consts.values())
            nodes.append(helper.make_node("Slice", [full] + [c.name for c in consts.values()], [out], name=n.name + "_step2"))
            n_fixed += 1
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return n_fixed


def check(name, ref_model, new_model, inp, x):
    a = ort.InferenceSession(ref_model, providers=["CPUExecutionProvider"]).run(None, {inp: x})[0]
    b = ort.InferenceSession(new_model, providers=["CPUExecutionProvider"]).run(None, {inp: x})[0]
    print(f"{name}: out {b.shape}, max diff vs original {np.abs(a - b).max():.2e}")
    x.astype(np.float32).tofile(new_model.replace(".onnx", "_in.f32"))
    a.astype(np.float32).tofile(new_model.replace(".onnx", "_out.f32"))


fix_shape(f"{B}/vocoder_m.onnx", f"{B}/vocoder_npu.onnx", "features", [1, 1024, VOC_FRAMES])
fix_shape(f"{B}/upsampler_m.onnx", f"{B}/upsampler_npu_tmp.onnx", "wav16", [1, 1, UP_SAMPLES])
fold(f"{B}/upsampler_npu_tmp.onnx", f"{B}/upsampler_npu.onnx")
print("upsampler: strided depthwise convs rewritten:", unstride_depthwise(f"{B}/upsampler_npu.onnx"))

import json
ref = json.load(open(f"{B}/ref/ex1.json"))
codes = np.array([ref["codes"]], np.int64)
proc = ort.InferenceSession(f"{B}/processer.onnx", providers=["CPUExecutionProvider"])
feats = proc.run(None, {"speech_tokens": codes, "context_tokens": np.array(ref["global_tokens"], np.int32).reshape(1, 1, 32)})[0]
f = np.zeros((1, 1024, VOC_FRAMES), np.float32)
f[:, :, :feats.shape[2]] = feats[:, :, :VOC_FRAMES]
check("vocoder_npu", f"{B}/vocoder.onnx", f"{B}/vocoder_npu.onnx", "features", f)
w = ort.InferenceSession(f"{B}/vocoder.onnx", providers=["CPUExecutionProvider"]).run(None, {"features": feats})[0].reshape(-1)
u = np.zeros((1, 1, UP_SAMPLES), np.float32)
u[0, 0, :min(len(w), UP_SAMPLES)] = w[:UP_SAMPLES]
check("upsampler_npu", f"{B}/upsampler.onnx", f"{B}/upsampler_npu.onnx", "wav16", u)
import os
os.remove(f"{B}/upsampler_npu_tmp.onnx")
