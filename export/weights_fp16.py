"""Store large fp32 weights as fp16 + a Cast back to fp32 (ONNX Runtime folds the casts at load): halves the
file, computes in fp32. usage: weights_fp16.py IN.onnx OUT.onnx"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

m = onnx.load(sys.argv[1])
g = m.graph
casts, n = [], 0
for i, t in enumerate(list(g.initializer)):
    if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= 4096:
        name = t.name  # t is the live proto: read it before overwriting
        h = numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), name + "_fp16")
        g.initializer[i].CopyFrom(h)
        casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT, name=name + "_cast"))
        n += 1
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print(f"{n} weights stored as fp16 -> {sys.argv[2]}")
