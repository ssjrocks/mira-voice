"""Score generated wavs: Whisper transcript + speaker similarity to a reference clip.

usage: score_wav.py REF_AUDIO OUT1.wav [OUT2.wav ...]
"""
import sys

import librosa
import torch
from faster_whisper import WhisperModel
from transformers import AutoFeatureExtractor, WavLMForXVector

ref, outs = sys.argv[1], sys.argv[2:]
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()


def emb(w):
    with torch.inference_mode():
        return torch.nn.functional.normalize(sv(**fe(w, sampling_rate=16000, return_tensors="pt")).embeddings[0], dim=-1)


r = emb(librosa.load(ref, sr=16000)[0])
for o in outs:
    w = librosa.load(o, sr=16000)[0]
    text = " ".join(s.text for s in asr.transcribe(w, language="en", beam_size=1)[0]).strip()
    print(f"{o.split('/')[-1]}: {len(w) / 16000:.2f}s  speaker-sim {float(emb(w) @ r):.3f}  \"{text}\"")
