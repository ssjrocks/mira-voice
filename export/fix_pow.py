"""Replace Pow(x, 2) with Mul(x, x): Qualcomm's NPU evaluates Pow as exp(y*log(x)), wrong for negative x
(the snake activations square sin(), which is negative half the time). Same maths on every backend.

usage: fix_pow.py IN.onnx OUT.onnx [IN2.onnx OUT2.onnx ...]
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fix(src, dst):
    m = onnx.load(src)
    g = m.graph
    consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    nodes, count = [], 0
    for n in g.node:
        e = consts.get(n.input[1]) if n.op_type == "Pow" else None
        if e is not None and e.size == 1 and float(e.reshape(-1)[0]) == 2.0:
            nodes.append(helper.make_node("Mul", [n.input[0], n.input[0]], list(n.output), name=n.name + "_sq"))
            count += 1
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"{src.split('/')[-1]} -> {dst.split('/')[-1]}: replaced {count} Pow(x, 2)")


args = sys.argv[1:]
for i in range(0, len(args), 2):
    fix(args[i], args[i + 1])
