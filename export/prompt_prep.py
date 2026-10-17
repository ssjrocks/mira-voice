"""Prepare a "prompted" voice: the reference clip's own speech codes + its transcript, cut at a word boundary.

Spark-TTS-style continuation cloning: the LLM gets  transcript + new text  and the reference's speech codes
as the start of its answer, so it carries on in that speaker's style (accent, rhythm) rather than only
copying timbre from the 32 voice tokens.

usage: prompt_prep.py REF.wav OUT.json [max_seconds]
"""
import json
import sys

import numpy as np
import onnxruntime as ort
import torch

import common as C

ref, out = sys.argv[1], sys.argv[2]
max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
audio = C.load_ref_audio(ref, duration=30)

# transcript with word timings; cut at the last word ending before max_s
from faster_whisper import WhisperModel
asr = WhisperModel("medium.en", device="cpu", compute_type="int8", cpu_threads=8)
segs, _ = asr.transcribe(audio, language="en", beam_size=5, word_timestamps=True)
words = [w for s in segs for w in s.words]
keep = [w for w in words if w.end <= max_s]
cut = keep[-1].end + 0.05 if keep else max_s
transcript = "".join(w.word for w in keep).strip()
clip = audio[:int(cut * 16000)]
print(f"full transcript: {''.join(w.word for w in words).strip()!r}")
print(f"prompt: {cut:.2f}s -> {transcript!r}")

# voice tokens (same as the app: whole clip, volume-normalised, tiled to 6 s)
spk = ort.InferenceSession(f"{C.BUILD}/speaker_encoder.onnx", providers=["CPUExecutionProvider"])
glob = spk.run(None, {"wav": C.ref_clip(audio)[None]})[0].reshape(-1).tolist()

# semantic speech codes of the prompt clip: wav2vec2-large-xlsr-53 layers 11/14/16 averaged -> q_encoder
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()
x = C.audio_volume_normalize(clip)
with torch.inference_mode():
    iv = proc(x, sampling_rate=16000, return_tensors="pt").input_values
    hs = w2v(iv, output_hidden_states=True).hidden_states
    feat = ((hs[11] + hs[14] + hs[16]) / 3).float().numpy()
q = ort.InferenceSession(f"{C.DEC_DIR}/q_encoder.onnx", providers=["CPUExecutionProvider"])
codes = q.run(["semantic_tokens"], {"features": feat})[0].reshape(-1).tolist()
print(f"{len(codes)} prompt codes ({len(codes) / 50:.2f}s), voice tokens {glob[:6]}...")
json.dump({"ref": ref, "cut_s": cut, "transcript": transcript, "global_tokens": glob, "prompt_codes": codes},
          open(out, "w"))
