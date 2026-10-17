"""Accent-mode experiments: (a) transcript running past the clip's audio, (b) repetition penalty on clip codes."""
import json, os, re, subprocess, sys
import numpy as np, onnxruntime as ort
import common as C
from faster_whisper import WhisperModel
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

pr = json.load(open(f"{C.BUILD}/accent/user_prompt.json"))
codes_file = f"{C.BUILD}/accent/run1/prompt_codes.csv"
g = ",".join(map(str, pr["global_tokens"]))
full_transcript = ("The Arduino Nano Matterboard has always been about the low power side of the smart home. "
                   "Until now your sketches had two ways into the smart home, matter over threads.")
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
dec = ort.InferenceSession(f"{C.BUILD}/decoder.onnx", providers=["CPUExecutionProvider"])
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
def emb(w):
    with torch.inference_mode():
        return torch.nn.functional.normalize(sv(**fe(w, sampling_rate=16000, return_tensors="pt")).embeddings[0], dim=-1)
ref = emb(C.load_ref_audio(f"{C.BUILD}/accent/user_ref.wav", duration=30))
TEXTS = ["The meeting has been moved to half past four on Thursday afternoon.",
         "Can you grab a coffee on your way past? I'll have a flat white, thanks.",
         "We drove up the coast on Saturday and stopped for fish and chips at the beach."]
def words(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
def wer(r, h):
    r, h = words(r), words(h); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j-1] + 1, prev + (r[i-1] != h[j-1])); prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)
def run(transcript, text, seed, penalty=True):
    env = dict(os.environ, MIRA_PROMPT_CODES=codes_file, MIRA_PROMPT_PENALTY="1" if penalty else "0")
    r = subprocess.run([f"{C.BUILD}/native-host/mira_cli", f"{C.BUILD}/mira-Q8_0.gguf", f"{C.BUILD}/mira_head.bin", "8",
                        str(seed), transcript + " " + text, g], capture_output=True, text=True, env=env)
    codes = json.loads(r.stdout.strip().splitlines()[-1])["codes"]
    if len(codes) < 5: return len(codes) / 50, "", 1.0, 0.0
    w = dec.run(None, {"speech_tokens": np.array([codes], np.int64), "context_tokens": np.array(pr["global_tokens"], np.int32).reshape(1,1,32)})[0].reshape(-1)
    h = " ".join(s.text for s in asr.transcribe(w, language="en", beam_size=1)[0]).strip()
    return len(w) / 16000, h, wer(text, h), float(emb(w) @ ref)
for name, transcript, penalty in [("matched transcript, penalty on ", pr["transcript"], True),
                                  ("matched transcript, penalty off", pr["transcript"], False),
                                  ("transcript longer than clip    ", full_transcript, True)]:
    rows = []
    for t in TEXTS:
        for seed in range(3):
            dur, h, e, sim = run(transcript, t, seed, penalty)
            rows.append((dur, e, sim))
            if name.startswith("transcript longer"): print(f"   {dur:5.2f}s  {h[:90]}")
    a = np.array(rows)
    print(f"{name}: mean dur {a[:,0].mean():.2f}s, near-silent (<1 s) {int((a[:,0] < 1).sum())}/9, WER {a[:,1].mean():.2f}, voice-sim {a[:,2].mean():.3f}", flush=True)
