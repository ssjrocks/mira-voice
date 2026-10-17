"""Check the exported ONNX files against the original pieces, and time them at phone-like thread counts."""
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

import common as C

B = C.BUILD
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def sess(path, n=threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = n
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---- speaker encoder: same 32 tokens as mel(torchaudio) + s_encoder on every example clip
orig = sess(f"{C.DEC_DIR}/s_encoder.onnx")
spk = sess(f"{B}/speaker_encoder.onnx")
for ex in ["example1", "example2", "example3"]:
    clip = C.ref_clip(C.load_ref_audio(f"{C.MODEL_DIR}/{ex}.wav"))
    a = orig.run(None, {"mel_spectrogram": C.torch_mel(clip)})[0].reshape(-1)
    t = time.time()
    b = spk.run(None, {"wav": clip[None]})[0].reshape(-1)
    print(f"speaker_encoder {ex}: {int((a == b).sum())}/32 tokens equal, {time.time() - t:.3f}s")

# ---- decoder vs original processer(fp16 onnx) + torch fp32 vocoder, on the reference's tokens
ref = json.load(open(f"{B}/ref/ex1.json"))
codes = np.array([ref["codes"]], dtype=np.int64)
ctx = np.array(ref["global_tokens"], dtype=np.int32).reshape(1, 1, 32)
proc = sess(f"{C.DEC_DIR}/processer.onnx")
feats = proc.run(None, {"speech_tokens": codes, "context_tokens": ctx})[0].astype(np.float32)
with torch.inference_mode():
    wav_ref = C.load_torch_vocoder()(torch.from_numpy(feats)).numpy().reshape(-1)
dec = sess(f"{B}/decoder.onnx")
dec.run(None, {"speech_tokens": codes[:, :50], "context_tokens": ctx})  # warm-up
t = time.time()
wav = dec.run(None, {"speech_tokens": codes, "context_tokens": ctx})[0].reshape(-1)
dt = time.time() - t
dur = len(wav) / 16000
snr = 10 * np.log10((wav_ref ** 2).sum() / ((wav - wav_ref) ** 2).sum())
print(f"decoder: {len(wav)} samples (ref {len(wav_ref)}), SNR vs original {snr:.1f} dB, "
      f"{dt:.2f}s for {dur:.2f}s audio (RTF {dt / dur:.3f}) @ {threads} threads")
C.write_wav(f"{B}/ref/ex1_onnx_16k.wav", wav, 16000)

# ---- upsampler vs torch FlashSR
with torch.inference_mode():
    up_ref = C.load_torch_upsampler()(torch.from_numpy(wav).view(1, 1, -1)).numpy().reshape(-1)
ups = sess(f"{B}/upsampler.onnx")
t = time.time()
up = ups.run(None, {"wav16": wav.reshape(1, 1, -1)})[0].reshape(-1)
dt = time.time() - t
snr = 10 * np.log10((up_ref ** 2).sum() / ((up - up_ref) ** 2).sum())
print(f"upsampler: {len(up)} samples, SNR vs torch {snr:.1f} dB, {dt:.2f}s (RTF {dt / dur:.3f})")
C.write_wav(f"{B}/ref/ex1_onnx_48k.wav", C.peak_normalize(up), 48000)
