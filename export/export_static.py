"""Fixed-shape copies of the vocoder and upsampler for NPU execution (QNN/NNAPI need static shapes).

Streaming decodes 50 codes at a time; the vocoder sees 8 codes of context before and 4 after (62 frames),
the upsampler 320 samples either side of the 16000-sample chunk.
"""
import subprocess
import sys

import numpy as np
import onnxruntime as ort

import common as C

VOC_FRAMES = 62
UP_SAMPLES = 16000 + 2 * 320

def fix(src, dst, name, shape):
    subprocess.run([sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed", "--input_name", name,
                    "--input_shape", ",".join(map(str, shape)), f"{C.BUILD}/{src}", f"{C.BUILD}/{dst}"], check=True)

fix("vocoder.onnx", "vocoder_s62.onnx", "features", [1, 1024, VOC_FRAMES])
fix("upsampler.onnx", "upsampler_s16640.onnx", "wav16", [1, 1, UP_SAMPLES])
for dyn, st, name, shape in [("vocoder", "vocoder_s62", "features", (1, 1024, VOC_FRAMES)),
                             ("upsampler", "upsampler_s16640", "wav16", (1, 1, UP_SAMPLES))]:
    x = np.random.randn(*shape).astype(np.float32) * (0.5 if name == "features" else 0.1)
    a = ort.InferenceSession(f"{C.BUILD}/{dyn}.onnx", providers=["CPUExecutionProvider"]).run(None, {name: x})[0]
    b = ort.InferenceSession(f"{C.BUILD}/{st}.onnx", providers=["CPUExecutionProvider"]).run(None, {name: x})[0]
    print(st, "out", b.shape, "max diff vs dynamic %.2e" % np.abs(a - b).max())
    np.save(f"{C.BUILD}/{st}_in.npy", x); np.save(f"{C.BUILD}/{st}_out.npy", a)
    x.tofile(f"{C.BUILD}/{st}_in.f32"); a.astype(np.float32).tofile(f"{C.BUILD}/{st}_out.f32")
