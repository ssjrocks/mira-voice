"""Default cloning (voice tokens only) vs prompted cloning (+ reference codes and transcript), scored for
words (Whisper), voice similarity (WavLM) and accent (CommonAccent ECAPA classifier).

usage: accent_test.py PROMPT.json OUT_DIR [model.gguf]
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

import common as C

prompt = json.load(open(sys.argv[1]))
out = sys.argv[2]
gguf = sys.argv[3] if len(sys.argv) > 3 else f"{C.BUILD}/mira-Q8_0.gguf"
os.makedirs(out, exist_ok=True)
TEXTS = [
    "I reckon we should head down to the beach this arvo, the weather's looking great.",
    "Can you grab a coffee on your way past? I'll have a flat white, thanks.",
    "The meeting has been moved to half past four on Thursday afternoon.",
]
g = prompt["global_tokens"]
codes_file = f"{out}/prompt_codes.csv"
open(codes_file, "w").write(",".join(map(str, prompt["prompt_codes"])))

dec = ort.InferenceSession(f"{C.BUILD}/decoder.onnx", providers=["CPUExecutionProvider"])
ups = ort.InferenceSession(f"{C.BUILD}/upsampler.onnx", providers=["CPUExecutionProvider"])

from faster_whisper import WhisperModel
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
from transformers import AutoFeatureExtractor, WavLMForXVector
sv_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
try:
    from speechbrain.inference.classifiers import EncoderClassifier
    accent = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa",
                                            savedir=f"{C.W}/models/accent-id-commonaccent_ecapa")
    labels = accent.hparams.label_encoder
except Exception as e:  # classifier is optional
    print("accent classifier unavailable:", e)
    accent = None


def emb(w):
    with torch.inference_mode():
        return torch.nn.functional.normalize(sv(**sv_fe(w, sampling_rate=16000, return_tensors="pt")).embeddings[0], dim=-1)


def accent_of(w):
    if accent is None:
        return {}
    with torch.inference_mode():
        out_prob, _, _, _ = accent.classify_batch(torch.from_numpy(w).float().unsqueeze(0))
    p = torch.softmax(out_prob[0], -1) if out_prob[0].sum().abs() > 1.01 or out_prob[0].min() < 0 else out_prob[0]
    p = p.exp() / p.exp().sum() if p.max() <= 0 else p
    top = torch.topk(p, 3)
    return {labels.decode_ndim(int(i)): round(float(v), 2) for v, i in zip(top.values, top.indices)}


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


ref_audio = C.load_ref_audio(prompt["ref"], duration=30)
ref_emb = emb(ref_audio)
print(f"reference: accent {accent_of(ref_audio)}")
rows = []
for ti, text in enumerate(TEXTS):
    for mode in ["default", "prompted"]:
        for seed in [0, 1]:
            env = dict(os.environ)
            full = text
            if mode == "prompted":
                env["MIRA_PROMPT_CODES"] = codes_file
                full = prompt["transcript"] + " " + text
            r = subprocess.run([f"{C.BUILD}/native-host/mira_cli", gguf, f"{C.BUILD}/mira_head.bin", "8", str(seed), full,
                                ",".join(map(str, g))], capture_output=True, text=True, env=env)
            codes = json.loads(r.stdout.strip().splitlines()[-1])["codes"]
            w16 = dec.run(None, {"speech_tokens": np.array([codes], np.int64),
                                 "context_tokens": np.array(g, np.int32).reshape(1, 1, 32)})[0].reshape(-1)
            w48 = C.peak_normalize(ups.run(None, {"wav16": w16.reshape(1, 1, -1)})[0].reshape(-1))
            name = f"t{ti}_{mode}_s{seed}"
            C.write_wav(f"{out}/{name}.wav", w48, 48000)
            hyp = " ".join(s.text for s in asr.transcribe(w16, language="en", beam_size=1)[0]).strip()
            row = dict(name=name, mode=mode, dur=round(len(w16) / 16000, 2), wer=round(wer(text, hyp), 2),
                       sim=round(float(emb(w16) @ ref_emb), 3), accent=accent_of(w16), hyp=hyp)
            rows.append(row)
            print(json.dumps(row), flush=True)

json.dump(rows, open(f"{out}/results.json", "w"), indent=1)
for mode in ["default", "prompted"]:
    rs = [r for r in rows if r["mode"] == mode]
    aus = np.mean([r["accent"].get("australia", 0.0) for r in rs]) if accent else float("nan")
    print(f"{mode:9s} WER {np.mean([r['wer'] for r in rs]):.2f}  voice-sim {np.mean([r['sim'] for r in rs]):.3f}  "
          f"P(australia) {aus:.2f}")
