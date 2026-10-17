"""Ground truth: run original MiraTTS on CPU (transformers instead of lmdeploy).

Usage: python reference.py REF_AUDIO "text" OUT_PREFIX [seed]
Writes OUT_PREFIX_16k.wav, OUT_PREFIX_48k.wav and OUT_PREFIX.json (tokens).
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

import common as C

ref_path, text, out = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.manual_seed(seed)

# 1. speaker (context) tokens
clip = C.ref_clip(C.load_ref_audio(ref_path))
mel = C.torch_mel(clip)
s_enc = ort.InferenceSession(f"{C.DEC_DIR}/s_encoder.onnx", providers=["CPUExecutionProvider"])
global_tokens = s_enc.run(["global_tokens"], {"mel_spectrogram": mel})[0].reshape(-1)
print("global tokens", global_tokens.tolist())

# 2. LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(C.MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(C.MODEL_DIR, dtype=torch.float32).eval()
ids_str = tok(C.prompt_string(text, global_tokens), add_special_tokens=False).input_ids
ids = C.build_prompt_ids(tok, text, global_tokens)
assert ids == ids_str, "manual prompt ids differ from tokenizing the prompt string"
t = time.time()
with torch.inference_mode():
    gen = model.generate(torch.tensor([ids]), do_sample=True, top_p=0.95, top_k=50, temperature=0.8,
                         repetition_penalty=1.2, min_p=0.05, max_new_tokens=1024,
                         eos_token_id=C.IM_END, pad_token_id=C.IM_END)
new = gen[0, len(ids):].tolist()
dt = time.time() - t
codes = C.speech_ids_to_codes(new)
print(f"LLM: {len(new)} tokens in {dt:.1f}s ({len(new)/dt:.1f} tok/s), {len(codes)} speech codes, "
      f"last token {new[-1]} (eos={C.IM_END})")
non_speech = [i for i in new if not (C.SPEECH_TOKEN_BASE <= i < C.SPEECH_TOKEN_BASE + C.N_SPEECH_TOKENS)]
print("non-speech generated tokens:", non_speech[:20])

# 3. decode: processer (onnx) -> vocoder (torch) -> upsampler (torch)
proc = ort.InferenceSession(f"{C.DEC_DIR}/processer.onnx", providers=["CPUExecutionProvider"])
feats = proc.run(["preprocessed_output"], {
    "speech_tokens": np.array([codes], dtype=np.int64),
    "context_tokens": global_tokens.reshape(1, 1, -1).astype(np.int32)})[0]
with torch.inference_mode():
    wav16 = C.load_torch_vocoder()(torch.from_numpy(feats.astype(np.float32))).squeeze().numpy()
    wav48 = C.load_torch_upsampler()(torch.from_numpy(wav16).view(1, 1, -1)).squeeze().numpy()
wav48 = C.peak_normalize(wav48)
C.write_wav(f"{out}_16k.wav", wav16, 16000)
C.write_wav(f"{out}_48k.wav", wav48, 48000)
json.dump({"text": text, "ref": ref_path, "global_tokens": global_tokens.tolist(), "prompt_ids": ids,
           "generated_ids": new, "codes": codes}, open(f"{out}.json", "w"))
print(f"audio {len(wav16)/16000:.2f}s -> {out}_48k.wav")
