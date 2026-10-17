"""How closely does chunked (streaming) decoding match decoding everything at once?

Chunk k covers tokens [s, e); it is decoded with LEFT tokens of past context and RIGHT tokens of
lookahead, and only the audio for [s, e) is kept (joined with a short crossfade).
"""
import json

import numpy as np
import onnxruntime as ort

import common as C

ref = json.load(open(f"{C.BUILD}/ref/ex1.json"))
codes = np.array(ref["codes"], dtype=np.int64)
ctx = np.array(ref["global_tokens"], dtype=np.int32).reshape(1, 1, 32)
dec = ort.InferenceSession(f"{C.BUILD}/decoder.onnx", providers=["CPUExecutionProvider"])
ups = ort.InferenceSession(f"{C.BUILD}/upsampler.onnx", providers=["CPUExecutionProvider"])
HOP = 320


def decode(c):
    return dec.run(None, {"speech_tokens": c[None], "context_tokens": ctx})[0].reshape(-1)


def upsample(w):
    return ups.run(None, {"wav16": w.reshape(1, 1, -1)})[0].reshape(-1)


full = decode(codes)
full48 = upsample(full)


def snr(a, b):
    n = min(len(a), len(b))
    return 10 * np.log10((b[:n] ** 2).sum() / ((a[:n] - b[:n]) ** 2).sum())


def chunked(chunk, left, right, fade=64, up_ctx=0):
    out16, out48 = np.zeros(0, np.float32), np.zeros(0, np.float32)
    n = len(codes)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = max(0, s - left), min(n, e + right)
        w = decode(codes[a:b])
        seg = w[(s - a) * HOP:(e - a) * HOP]
        out16 = np.concatenate([out16, seg])
        # upsample this segment with some 16 kHz context on both sides
        ua, ub = max(0, (s - a) * HOP - up_ctx), min(len(w), (e - a) * HOP + up_ctx)
        u = upsample(w[ua:ub])
        u = u[((s - a) * HOP - ua) * 3:((s - a) * HOP - ua) * 3 + len(seg) * 3]
        out48 = np.concatenate([out48, u])
    return out16, out48


print(f"{len(codes)} codes; peak of full-decode wav16 {np.abs(full).max():.3f}, wav48 {np.abs(full48).max():.3f}")
for chunk, left, right, up_ctx in [(50, 0, 0, 0), (50, 8, 4, 0), (50, 16, 8, 0), (50, 16, 8, 640), (50, 24, 12, 1280),
                                   (25, 16, 8, 640), (100, 16, 8, 640)]:
    a16, a48 = chunked(chunk, left, right, up_ctx=up_ctx)
    print(f"chunk {chunk:3d} left {left:2d} right {right:2d} up_ctx {up_ctx:4d}: SNR16 {snr(a16, full):5.1f} dB  SNR48 {snr(a48, full48):5.1f} dB")
