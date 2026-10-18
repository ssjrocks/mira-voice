"""Compiles the fixed-shape NPU models (export_npu.py) for the Galaxy S23's NPU (Snapdragon 8 Gen 2, Hexagon v73) on
this PC. The phone then loads the result in about a second instead of compiling for minutes, and needs none of the
compile's memory (the upsampler's on-phone compile ran the phone out of memory).

Uses Qualcomm's ONNX Runtime QNN plugin (pip onnxruntime-qnn, in build/qnn-venv). Its QNN version must equal the
app's com.qualcomm.qti:qnn-runtime: both 2.51.0 (onnxruntime-qnn 2.7.0).

Notes from getting this to work:
- "Failed to finalize QNN graph ... MEM_ALLOC" means a tensor row doesn't fit the NPU's 8 MB on-chip memory; --verbose
  shows QNN's own reason ("Requires 0x848000 bytes of TCM"). export_npu.py's window sizes are chosen to fit.
- fp16 needs soc_model; 43 is SM8550 (QNN's log names it).
- QNN's x86 emulation (--check) works for the vocoder (32 dB vs the CPU, as on the phone) but crashes on the upsampler.

usage: build/qnn-venv/bin/python export/qnn_context.py [--check] MODEL.onnx...
       writes build/qnn/<stem>_ctx.onnx (or --out DIR) + its QNN binary; --check also runs <stem>_in.f32 through the
       compiled model on the PC (HTP emulation) and compares with <stem>_out.f32; --check-only skips compiling
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn as oq

OUT = Path(__file__).resolve().parent.parent / "build" / "qnn"
# Same settings as the app's NpuBench.qnnOptions, plus the target chip (no NPU to detect it from on the PC)
QNN_OPTIONS = {
    "backend_path": oq.get_qnn_htp_path(),
    "htp_arch": "73",
    "soc_model": "43",  # SM8550
    "enable_htp_fp16_precision": "1",
    "htp_graph_finalization_optimization_mode": "3",
}

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--check-only", action="store_true", help="skip compiling: check an existing <stem>_ctx.onnx")
ap.add_argument("--opt", action="append", default=[], help="override a QNN option, key=value ('key=' removes it)")
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--out", type=Path, default=OUT)
ap.add_argument("models", nargs="+", type=Path)
args = ap.parse_args()
if args.verbose:
    ort.set_default_logger_severity(0)  # QNN logs through the default logger
for kv in args.opt:
    k, v = kv.split("=", 1)
    QNN_OPTIONS.pop(k, None) if v == "" else QNN_OPTIONS.__setitem__(k, v)

ort.register_execution_provider_library("QNNExecutionProvider", oq.get_library_path())
qnn = next(d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider")


def options(**config):
    so = ort.SessionOptions()
    so.log_severity_level = 0 if args.verbose else 3
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")  # the whole graph must run on the NPU
    for k, v in config.items():
        so.add_session_config_entry(k, v)
    so.add_provider_for_devices([qnn], QNN_OPTIONS)
    return so


args.out.mkdir(parents=True, exist_ok=True)
for model in args.models:
    ctx = args.out / f"{model.stem}_ctx.onnx"
    if not args.check_only:
        t = time.time()
        ort.InferenceSession(str(model), options(**{"ep.context_enable": "1", "ep.context_file_path": str(ctx),
                                                    "ep.context_embed_mode": "0"}))
        files = sorted(p.name for p in args.out.glob(f"{model.stem}_ctx*"))
        print(f"{model.name}: compiled in {time.time() - t:.0f} s, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"
              f" -> {', '.join(files)}", flush=True)
        for n in onnx.load(str(ctx)).graph.node:
            print("  ", n.op_type, n.name, {a.name: a.s.decode() if a.type == onnx.AttributeProto.STRING else a.i for a in n.attribute}, flush=True)
    if args.check or args.check_only:
        x_file, y_file = model.with_name(f"{model.stem}_in.f32"), model.with_name(f"{model.stem}_out.f32")
        s = ort.InferenceSession(str(ctx), options())
        inp = s.get_inputs()[0]
        x = np.fromfile(x_file, np.float32).reshape(inp.shape)
        t = time.time()
        y = s.run(None, {inp.name: x})[0].ravel()
        ref = np.fromfile(y_file, np.float32)
        snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((y - ref) ** 2), 1e-30))
        print(f"  check: {time.time() - t:.0f} s emulated, SNR {snr:.1f} dB vs CPU, max diff {np.abs(y - ref).max():.4f}", flush=True)
sys.exit(0)
