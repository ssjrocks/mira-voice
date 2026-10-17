"""Sub-models of the static vocoder (cut after each top-level stage) plus a Sin-only probe, each with
real-data test vectors, to find which part the NPU computes wrongly."""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

import common as C

B = C.BUILD
src = f"{B}/vocoder_s62.onnx"
m = onnx.load(src)
x = np.fromfile(f"{B}/vocoder_s62_in.f32", dtype=np.float32).reshape(1, 1024, 62)
# last node output of each top-level stage /model/model.K/...
last = {}
for n in m.graph.node:
    parts = n.name.split("/")
    if len(parts) > 2 and parts[1] == "model" and parts[2].startswith("model."):
        last[int(parts[2].split(".")[1])] = n.output[0]
print("stages:", sorted(last))
cuts = []
for k in [0, 1, 2, 3, 4, 5]:
    if k not in last:
        continue
    name = f"vcut{k}"
    onnx.utils.extract_model(src, f"{B}/{name}.onnx", ["features"], [last[k]])
    y = ort.InferenceSession(f"{B}/{name}.onnx", providers=["CPUExecutionProvider"]).run(None, {"features": x})[0]
    x.tofile(f"{B}/{name}_in.f32"); y.astype(np.float32).tofile(f"{B}/{name}_out.f32")
    cuts.append(name); print(name, "->", last[k], y.shape, "range %.2f..%.2f" % (y.min(), y.max()))

# Sin probe: sin(x * 3) over the same input (|x*3| up to ~25)
inp = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1024, 62])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1024, 62])
k = helper.make_tensor("k", TensorProto.FLOAT, [], [3.0])
g = helper.make_graph([helper.make_node("Mul", ["features", "k"], ["z"]), helper.make_node("Sin", ["z"], ["y"])],
                      "sinprobe", [inp], [out], [k])
mp = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); mp.ir_version = 9
onnx.save(mp, f"{B}/sinprobe.onnx")
x.tofile(f"{B}/sinprobe_in.f32"); np.sin(x * 3).astype(np.float32).tofile(f"{B}/sinprobe_out.f32")
print("cuts:", " ".join(cuts), "sinprobe")
