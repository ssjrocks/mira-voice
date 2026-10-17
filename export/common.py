"""Shared paths and the original (PyTorch/ONNX) MiraTTS pieces, run on CPU.

This mirrors upstream MiraTTS (mira/model.py + FastBiCodec + FlashSR) but without
lmdeploy/CUDA, so it serves as the reference the phone pipeline is checked against.
"""
import os
import re
import sys

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = f"{W}/models/MiraTTS"
DEC_DIR = f"{MODEL_DIR}/decoders"
BUILD = f"{W}/build"
sys.path.insert(0, f"{W}/upstream/FastBiCodec")
sys.path.insert(0, f"{W}/upstream/FlashSR")

# Token id layout (from added_tokens.json): contiguous blocks.
CONTEXT_TOKEN_BASE = 151665   # <|context_token_0|> .. 4095
SPEECH_TOKEN_BASE = 155761    # <|speech_token_0|> .. 8191
N_SPEECH_TOKENS = 8192
IM_END = 151645               # eos
TASK_TTS = 165137
START_TEXT = 165146
END_TEXT = 165152
CONTEXT_AUDIO_START = 165150
CONTEXT_AUDIO_END = 165156
PROMPT_SPEECH_START = 165151

REF_SAMPLES = 96000  # 6 s at 16 kHz, what the speaker encoder sees


def audio_volume_normalize(audio, coeff=0.2):
    from ncodec.encoder.model import audio_volume_normalize as f
    return f(audio, coeff)


def load_ref_audio(path, duration=8):
    import librosa
    audio, _ = librosa.load(path, duration=duration, sr=16000)
    return audio


def ref_clip(audio):
    audio = audio_volume_normalize(audio)
    n = len(audio)
    if REF_SAMPLES > n:
        audio = np.tile(audio, REF_SAMPLES // n + 1)
    return audio[:REF_SAMPLES].astype(np.float32)


def torch_mel(wav_1d):
    """Reference mel spectrogram, exactly as FastBiCodec's AudioEncoder computes it."""
    import torch
    from ncodec.encoder.model import get_mel_transformer
    mel = get_mel_transformer()(torch.from_numpy(wav_1d).unsqueeze(0).float()).squeeze(1)
    return np.ascontiguousarray(mel.transpose(1, 2).numpy())  # [1, T, 128]


def build_prompt_ids(tokenizer, text, global_tokens):
    ids = [TASK_TTS, START_TEXT]
    ids += tokenizer.encode(text, add_special_tokens=False)
    ids += [END_TEXT, CONTEXT_AUDIO_START]
    ids += [CONTEXT_TOKEN_BASE + int(g) for g in global_tokens]
    ids += [CONTEXT_AUDIO_END, PROMPT_SPEECH_START]
    return ids


def prompt_string(text, global_tokens):
    ctx = "".join(f"<|context_token_{int(i)}|>" for i in global_tokens)
    return (f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{ctx}"
            f"<|context_audio_end|><|prompt_speech_start|>")


def speech_ids_to_codes(ids):
    return [i - SPEECH_TOKEN_BASE for i in ids
            if SPEECH_TOKEN_BASE <= i < SPEECH_TOKEN_BASE + N_SPEECH_TOKENS]


def speech_text_to_codes(text):
    return [int(t) for t in re.findall(r"speech_token_(\d+)", text)]


def load_torch_vocoder():
    """The vocoder ('detokenizer') as upstream builds it, but fp32 on CPU."""
    from safetensors.torch import load_file
    from ncodec.decoder.model_utils import Decoder, remove_weight_norm_recursive
    cfg = {'input_channel': 1024, 'channels': 1536, 'rates': [8, 5, 4, 2], 'kernel_sizes': [16, 11, 8, 4]}
    dec = Decoder(**cfg)
    dec.apply(remove_weight_norm_recursive)
    missing, unexpected = dec.load_state_dict(load_file(f"{DEC_DIR}/detokenizer.safetensors"), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return dec.eval().float()


def load_torch_upsampler():
    from FastAudioSR import FASR
    return FASR(f"{DEC_DIR}/upsampler.pth").model.eval().float()


def peak_normalize(x):
    return x / np.abs(x).max() * 0.999


def write_wav(path, wav, sr):
    import soundfile as sf
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)
