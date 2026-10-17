"""semantic_encoder.onnx: 16 kHz clip -> speech codes, for prompted ("accent") cloning on the phone.

= HF Wav2Vec2FeatureExtractor normalisation + wav2vec2-large-xlsr-53 (only the first 16 of 24 layers are
needed: upstream averages hidden_states 11, 14, 16 = outputs of layers 10, 13, 15) + MiraTTS q_encoder.onnx.
Writes an fp32 and an int8 (dynamic MatMul quantisation) copy, and checks both against the PC pipeline.

  wav [1, T] f32 (volume-normalised)  ->  semantic_tokens [1, ~T/320] i64
"""
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import common as C
from export_onnx import export, merge, to_opset, OPSET

B = C.BUILD


class Wav2VecFeatures(torch.nn.Module):
    def __init__(self, w2v):
        super().__init__()
        self.fe = w2v.feature_extractor
        self.fp = w2v.feature_projection
        self.pos = w2v.encoder.pos_conv_embed
        self.layers = torch.nn.ModuleList(w2v.encoder.layers[:16])

    def forward(self, wav):
        x = (wav - wav.mean()) / torch.sqrt(wav.var(unbiased=False) + 1e-7)
        h, _ = self.fp(self.fe(x).transpose(1, 2))
        h = h + self.pos(h)
        acc = None
        for i, layer in enumerate(self.layers):
            h = layer(h)[0]
            if i in (10, 13, 15):
                acc = h if acc is None else acc + h
        return acc / 3


def main():
    w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53", attn_implementation="eager").eval()
    mod = Wav2VecFeatures(w2v).eval()
    clip = C.audio_volume_normalize(C.load_ref_audio(f"{C.MODEL_DIR}/example1.wav"))
    x = torch.from_numpy(clip).unsqueeze(0)
    with torch.inference_mode():
        proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        hs = w2v(proc(clip, sampling_rate=16000, return_tensors="pt").input_values, output_hidden_states=True).hidden_states
        ref = ((hs[11] + hs[14] + hs[16]) / 3).numpy()
        mine = mod(x).numpy()
    print("features: max abs diff vs HF %.2e (max %.2f)" % (np.abs(mine - ref).max(), np.abs(ref).max()))

    if not os.path.exists(f"{B}/semantic_encoder_fp32.onnx"):
        w_m = export(mod, (x,), f"{B}/w2v16.onnx", ["wav"], ["features"], {"wav": {1: "samples"}, "features": {1: "frames"}})
        q = to_opset(onnx.load(f"{C.DEC_DIR}/q_encoder.onnx"), OPSET)
        sem = merge(w_m, q, [("features", "features")])
        onnx.save(sem, f"{B}/semantic_encoder_fp32.onnx", save_as_external_data=False)
        os.remove(f"{B}/w2v16.onnx")
    else:  # reuse; just make sure the opset list has no duplicates
        sem = onnx.load(f"{B}/semantic_encoder_fp32.onnx")
        seen = {}
        for o in sem.opset_import:
            seen[o.domain] = max(seen.get(o.domain, 0), o.version)
        del sem.opset_import[:]
        sem.opset_import.extend(onnx.helper.make_opsetid(d, v) for d, v in seen.items())
        onnx.save(sem, f"{B}/semantic_encoder_fp32.onnx")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(f"{B}/semantic_encoder_fp32.onnx", f"{B}/semantic_encoder.onnx",
                     op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8, per_channel=True)

    # agreement with the PC pipeline (prompt_prep.py) on the user's clip and on example1
    qs = ort.InferenceSession(f"{C.DEC_DIR}/q_encoder.onnx", providers=["CPUExecutionProvider"])
    checks = [("example1", clip)]
    if os.path.exists(f"{B}/accent/user_ref.wav"):  # optional: your own voice clip
        checks.append(("user", C.audio_volume_normalize(C.load_ref_audio(f"{B}/accent/user_ref.wav", duration=30)[:int(7.92 * 16000)])))
    for name, audio in checks:
        with torch.inference_mode():
            hs = w2v(proc(audio, sampling_rate=16000, return_tensors="pt").input_values, output_hidden_states=True).hidden_states
        ref_codes = qs.run(None, {"features": ((hs[11] + hs[14] + hs[16]) / 3).numpy()})[0].reshape(-1)
        for f in ["semantic_encoder_fp32", "semantic_encoder"]:
            s = ort.InferenceSession(f"{B}/{f}.onnx", providers=["CPUExecutionProvider"])
            codes = s.run(None, {"wav": audio[None].astype(np.float32)})[0].reshape(-1)
            n = min(len(codes), len(ref_codes))
            print(f"{name:8s} {f:22s}: {len(codes)} codes, {np.mean(codes[:n] == ref_codes[:n]) * 100:.1f}% equal to HF pipeline")
    for f in ["semantic_encoder_fp32", "semantic_encoder"]:
        print(f, "%.0f MB" % (os.path.getsize(f"{B}/{f}.onnx") / 1e6))


if __name__ == "__main__":
    main()
