"""Export the audio side of MiraTTS to self-contained fp32 ONNX files for the phone.

  speaker_encoder.onnx  wav [1, 96000] f32 (16 kHz, volume-normalised, tiled)  -> global_tokens [1,1,32] i32
  decoder.onnx          speech_tokens [1,N] i64, context_tokens [1,1,32] i32    -> wav16 [1,1,N*320] f32
  upsampler.onnx        wav16 [1,1,T] f32                                       -> wav48 [1,1,3T] f32 (not peak-normalised)
"""
import numpy as np
import onnx
import torch
import torchaudio
from onnx import compose, helper, numpy_helper, TensorProto

import common as C

OPSET = 18
B = C.BUILD


# ---------------------------------------------------------------- mel spectrogram
class Mel(torch.nn.Module):
    """torchaudio MelSpectrogram(16k, n_fft=1024, win=640, hop=320, f_min=10, 128 mels, power=1,
    slaney/slaney, center=True, reflect pad) as conv1d, so it exports to plain ONNX ops."""

    def __init__(self):
        super().__init__()
        n_fft, win, self.hop = 1024, 640, 320
        w = torch.hann_window(win, periodic=True, dtype=torch.float64)
        lpad = (n_fft - win) // 2
        window = torch.zeros(n_fft, dtype=torch.float64)
        window[lpad:lpad + win] = w
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        ang = 2 * torch.pi * k[:, None] * n[None, :] / n_fft
        self.register_buffer("cos", (torch.cos(ang) * window).float().unsqueeze(1))
        self.register_buffer("sin", (-torch.sin(ang) * window).float().unsqueeze(1))
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 10.0, 8000.0, 128, 16000, norm="slaney",
                                                    mel_scale="slaney")
        self.register_buffer("fb", fb.float())  # [513, 128]
        self.pad = n_fft // 2

    def forward(self, wav):  # [1, T]
        x = torch.nn.functional.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        re = torch.nn.functional.conv1d(x, self.cos, stride=self.hop)
        im = torch.nn.functional.conv1d(x, self.sin, stride=self.hop)
        mag = torch.sqrt(re * re + im * im)          # [1, 513, frames]
        return torch.matmul(mag.transpose(1, 2), self.fb)  # [1, frames, 128]


def export(module, args, path, input_names, output_names, dynamic_axes=None):
    torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=OPSET, dynamo=False)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    return m


def merge(m1, m2, io_map):
    """Chain two models; internal names get prefixes, graph inputs/outputs keep theirs."""
    m1 = compose.add_prefix(m1, "a_", rename_inputs=False, rename_outputs=False)
    m2 = compose.add_prefix(m2, "b_", rename_inputs=False, rename_outputs=False)
    m1.ir_version = m2.ir_version = max(m1.ir_version, m2.ir_version)
    m = compose.merge_models(m1, m2, io_map=io_map)
    seen = {}
    for o in m.opset_import:  # merge_models can list the same domain twice
        seen[o.domain] = max(seen.get(o.domain, 0), o.version)
    del m.opset_import[:]
    m.opset_import.extend(helper.make_opsetid(d, v) for d, v in seen.items())
    onnx.checker.check_model(m)
    return m


def to_opset(m, version):
    if m.opset_import[0].version != version:
        m = onnx.version_converter.convert_version(m, version)
    return m


def fp16_to_fp32(m):
    """Convert an fp16 model to fp32: initializers, Constant nodes, Cast targets and declared types."""
    g = m.graph
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for node in g.node:
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
            if node.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    return m


def main():
    # ---- speaker encoder = mel + s_encoder
    mel = Mel().eval()
    wav = torch.from_numpy(C.ref_clip(C.load_ref_audio(f"{C.MODEL_DIR}/example1.wav"))).unsqueeze(0)
    ref_mel = C.torch_mel(wav[0].numpy())
    with torch.no_grad():
        my_mel = mel(wav).numpy()
    print("mel max abs diff vs torchaudio: %.2e (max value %.2f)" % (np.abs(my_mel - ref_mel).max(), ref_mel.max()))
    mel_m = export(mel, (wav,), f"{B}/mel.onnx", ["wav"], ["mel"])
    s_enc = to_opset(onnx.load(f"{C.DEC_DIR}/s_encoder.onnx"), OPSET)
    spk = merge(mel_m, s_enc, [("mel", "mel_spectrogram")])
    onnx.save(spk, f"{B}/speaker_encoder.onnx")

    # ---- decoder = processer (fp16 -> fp32) + vocoder
    proc = fp16_to_fp32(onnx.load(f"{C.DEC_DIR}/processer.onnx"))
    proc = to_opset(proc, OPSET)
    voc = C.load_torch_vocoder()
    # the jit-scripted snake() doesn't export cleanly; same maths in plain torch
    from ncodec.decoder import layers

    def snake_fwd(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
    layers.Snake1d.forward = snake_fwd
    feats = torch.randn(1, 1024, 50)
    voc_m = export(voc, (feats,), f"{B}/vocoder.onnx", ["features"], ["wav16"],
                   {"features": {2: "frames"}, "wav16": {2: "samples"}})
    dec = merge(proc, voc_m, [("preprocessed_output", "features")])
    onnx.save(dec, f"{B}/decoder.onnx")

    # ---- upsampler: FlashSR's own (dynamo) ONNX export of the identical upsampler.pth,
    # with its I/O renamed (the legacy exporter can't handle its edge-padding resampler)
    ups = onnx.load(f"{C.W}/upstream/FlashSR/models/model.onnx")
    rename = {"x": "wav16", "tanh": "wav48"}
    for v in list(ups.graph.input) + list(ups.graph.output):
        v.name = rename.get(v.name, v.name)
    for node in ups.graph.node:
        node.input[:] = [rename.get(n, n) for n in node.input]
        node.output[:] = [rename.get(n, n) for n in node.output]
    onnx.checker.check_model(ups)
    onnx.save(ups, f"{B}/upsampler.onnx")
    import os
    for f in ["speaker_encoder", "decoder", "upsampler"]:
        print(f, "%.1f MB" % (os.path.getsize(f"{B}/{f}.onnx") / 1e6))


if __name__ == "__main__":
    main()
