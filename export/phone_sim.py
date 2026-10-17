"""Run the exact phone pipeline on the PC (ONNX files + native engine via mira_cli) and score it.

Scores: word error rate from Whisper (did it say the text?) and speaker similarity to the
reference clip from a speaker-verification model (does it sound like the person?).

usage: phone_sim.py QUANTS THREADS [n_voices] [n_texts]   e.g. phone_sim.py f16,Q8_0,Q4_K_M 4
"""
import json
import re
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

import common as C

TEXTS = [
    "Hey there! This is a quick test of voice cloning, running entirely on a CPU.",
    "The weather today is sunny with a light breeze, perfect for a walk in the park.",
    "I can't believe it's already October. Where did the year go?",
    "Please remember to pick up milk, eggs, and bread on your way home tonight.",
]
VOICES = ["example1", "example2", "example3"]

quants = sys.argv[1].split(",")
threads = int(sys.argv[2])
VOICES = VOICES[:int(sys.argv[3])] if len(sys.argv) > 3 else VOICES
TEXTS = TEXTS[:int(sys.argv[4])] if len(sys.argv) > 4 else TEXTS
OUT = f"{C.BUILD}/sim"
import os
os.makedirs(OUT, exist_ok=True)


def sess(name):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(f"{C.BUILD}/{name}.onnx", so, providers=["CPUExecutionProvider"])


spk_enc, decoder, upsampler = sess("speaker_encoder"), sess("decoder"), sess("upsampler")

from faster_whisper import WhisperModel
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)

import torch
from transformers import AutoFeatureExtractor, WavLMForXVector
sv_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()


def embed(wav16):
    with torch.inference_mode():
        x = sv_fe(wav16, sampling_rate=16000, return_tensors="pt")
        e = sv(**x).embeddings[0]
    return torch.nn.functional.normalize(e, dim=-1)


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


rows = []
for voice in VOICES:
    raw = C.load_ref_audio(f"{C.MODEL_DIR}/{voice}.wav")
    ref_emb = embed(raw)
    clip = C.ref_clip(raw)
    g = spk_enc.run(None, {"wav": clip[None]})[0].reshape(-1).tolist()
    for ti, text in enumerate(TEXTS):
        for q in quants:
            r = subprocess.run([f"{C.BUILD}/native-host/mira_cli", f"{C.BUILD}/mira-{q}.gguf", f"{C.BUILD}/mira_head.bin", str(threads), "0", text,
                                ",".join(map(str, g))], capture_output=True, text=True)
            res = json.loads(r.stdout.strip().splitlines()[-1])
            codes = np.array([res["codes"]], dtype=np.int64)
            t = time.time()
            wav16 = decoder.run(None, {"speech_tokens": codes, "context_tokens": np.array(g, np.int32).reshape(1, 1, 32)})[0].reshape(-1)
            t_dec = time.time() - t
            t = time.time()
            wav48 = upsampler.run(None, {"wav16": wav16.reshape(1, 1, -1)})[0].reshape(-1)
            t_up = time.time() - t
            name = f"{voice}_t{ti}_{q}"
            C.write_wav(f"{OUT}/{name}.wav", C.peak_normalize(wav48), 48000)
            hyp = " ".join(s.text for s in asr.transcribe(wav16, language="en", beam_size=1)[0]).strip()
            sim = float(embed(wav16) @ ref_emb)
            dur = len(wav16) / 16000
            row = dict(name=name, quant=q, dur=round(dur, 2), tok_s=round(res["generated_tokens"] / res["generate_ms"] * 1000, 1),
                       prefill_ms=res["prefill_ms"], dec_rtf=round(t_dec / dur, 3), up_rtf=round(t_up / dur, 3),
                       wer=round(wer(text, hyp), 3), spk_sim=round(sim, 3), hyp=hyp)
            rows.append(row)
            print(json.dumps(row), flush=True)

json.dump(rows, open(f"{OUT}/results.json", "w"), indent=1)
print("\nsummary (mean over voices x texts):")
for q in quants:
    rs = [r for r in rows if r["quant"] == q]
    print(f"  {q:7s} WER {np.mean([r['wer'] for r in rs]):.3f}  speaker-sim {np.mean([r['spk_sim'] for r in rs]):.3f}  "
          f"LLM {np.mean([r['tok_s'] for r in rs]):.0f} tok/s  decoder RTF {np.mean([r['dec_rtf'] for r in rs]):.3f}  "
          f"upsampler RTF {np.mean([r['up_rtf'] for r in rs]):.3f}  (@{threads} threads)")
