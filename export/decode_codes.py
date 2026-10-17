"""Decode mira_cli JSON outputs (speech codes) on the PC and score them.

usage: decode_codes.py REF_AUDIO GLOBAL_CSV OUT.json [OUT2.json ...]
Each json is mira_cli's last line; writes OUT.wav (16 kHz) next to it.
"""
import json
import sys

import numpy as np
import onnxruntime as ort

import common as C

ref, g, files = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
dec = ort.InferenceSession(f"{C.BUILD}/decoder.onnx", providers=["CPUExecutionProvider"])
wavs = []
for f in files:
    d = json.loads(open(f).read().strip().splitlines()[-1])
    codes = np.array([d["codes"]], dtype=np.int64)
    w = dec.run(None, {"speech_tokens": codes, "context_tokens": np.array(g, np.int32).reshape(1, 1, 32)})[0].reshape(-1)
    C.write_wav(f.replace(".json", ".wav"), w, 16000)
    wavs.append(f.replace(".json", ".wav"))
    print(f"{f.split('/')[-1]}: {len(d['codes'])} codes, {d['generated_tokens'] / d['generate_ms'] * 1000:.1f} tok/s, "
          f"first codes {d['codes'][:8]}")
import subprocess
subprocess.run([sys.executable, f"{C.W}/export/score_wav.py", ref] + wavs)
