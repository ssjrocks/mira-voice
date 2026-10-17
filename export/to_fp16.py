"""fp16 copies of the audio models (inputs/outputs stay fp32) for CPUs with fast fp16 arithmetic."""
import sys

import onnx
from onnxconverter_common import float16

import common as C

for name in sys.argv[1:]:
    m = onnx.load(f"{C.BUILD}/{name}.onnx")
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, f"{C.BUILD}/{name}_fp16.onnx")
    print(name, "->", f"{name}_fp16.onnx")
