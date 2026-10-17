"""Write mira_head.bin: the LM-head rows for the 8192 speech tokens + eos, fp16, row-major [8193, 896].

The engine computes only these logits from the final hidden state instead of letting llama.cpp
score all 166k vocabulary entries (the head is tied to the input embeddings).
"""
import numpy as np
from safetensors import safe_open

import common as C

with safe_open(f"{C.MODEL_DIR}/model.safetensors", "pt") as f:
    keys = [k for k in f.keys() if "embed_tokens" in k or k == "lm_head.weight"]
    emb = f.get_tensor(keys[0]).float().numpy()
rows = list(range(C.SPEECH_TOKEN_BASE, C.SPEECH_TOKEN_BASE + C.N_SPEECH_TOKENS)) + [C.IM_END]
head = emb[rows].astype(np.float16)
head.tofile(f"{C.BUILD}/mira_head.bin")
print(keys[0], emb.shape, "->", head.shape, f"{head.nbytes / 1e6:.1f} MB")
