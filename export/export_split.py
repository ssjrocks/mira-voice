"""Split decoder for streaming: processer.onnx (codes+voice -> 1024-d features, fp32) and vocoder.onnx
(features -> 16 kHz audio, written by export_onnx.py). Verifies processer+vocoder == decoder.onnx."""
import json

import numpy as np
import onnx
import onnxruntime as ort

import common as C
from export_onnx import fp16_to_fp32, to_opset, OPSET

proc = to_opset(fp16_to_fp32(onnx.load(f"{C.DEC_DIR}/processer.onnx")), OPSET)
onnx.checker.check_model(proc)
onnx.save(proc, f"{C.BUILD}/processer.onnx")

ref = json.load(open(f"{C.BUILD}/ref/ex1.json"))
codes = np.array([ref["codes"]], np.int64)
ctx = np.array(ref["global_tokens"], np.int32).reshape(1, 1, 32)
s = lambda n: ort.InferenceSession(f"{C.BUILD}/{n}.onnx", providers=["CPUExecutionProvider"])
f = s("processer").run(None, {"speech_tokens": codes, "context_tokens": ctx})[0]
w = s("vocoder").run(None, {"features": f})[0].reshape(-1)
d = s("decoder").run(None, {"speech_tokens": codes, "context_tokens": ctx})[0].reshape(-1)
print("features", f.shape, f.dtype, "| split vs merged decoder max diff %.2e" % np.abs(w - d).max())
