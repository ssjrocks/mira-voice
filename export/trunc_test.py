"""How often does prompted (accent) mode stop early? Two sentences per LLM call vs one at a time."""
import json, os, re, subprocess, sys
import numpy as np
import common as C
from faster_whisper import WhisperModel

prompt = json.load(open(f"{C.BUILD}/accent/user_prompt.json"))
codes_file = f"{C.BUILD}/accent/run1/prompt_codes.csv"
g = ",".join(map(str, prompt["global_tokens"]))
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
import onnxruntime as ort
dec = ort.InferenceSession(f"{C.BUILD}/decoder.onnx", providers=["CPUExecutionProvider"])
S1 = "I reckon we should head down to the beach this arvo, the weather's looking great."
S2 = "Can you grab a coffee on your way past? I'll have a flat white, thanks."
def gen(text, seed):
    env = dict(os.environ, MIRA_PROMPT_CODES=codes_file)
    r = subprocess.run([f"{C.BUILD}/native-host/mira_cli", f"{C.BUILD}/mira-Q8_0.gguf", f"{C.BUILD}/mira_head.bin", "8", str(seed),
                        prompt["transcript"] + " " + text, g], capture_output=True, text=True, env=env)
    return json.loads(r.stdout.strip().splitlines()[-1])["codes"]
def words(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
def heard(codes):
    w = dec.run(None, {"speech_tokens": np.array([codes], np.int64), "context_tokens": np.array(prompt["global_tokens"], np.int32).reshape(1,1,32)})[0].reshape(-1)
    return " ".join(s.text for s in asr.transcribe(w, language="en", beam_size=1)[0])
for mode in ["two-per-call", "one-per-call"]:
    complete = 0
    for seed in range(6):
        if mode == "two-per-call":
            h = heard(gen(S1 + " " + S2, seed))
        else:
            h = heard(gen(S1, seed)) + " " + heard(gen(S2, seed))
        ok = words(h)[-2:] == ["white", "thanks"] or words(h)[-1:] == ["thanks"]
        complete += ok
        print(f"{mode} seed {seed}: {'complete' if ok else 'CUT   '} | {h.strip()[-70:]}", flush=True)
    print(f"== {mode}: {complete}/6 complete")
