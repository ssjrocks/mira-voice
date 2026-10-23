# Mira Voice

Voice cloning text-to-speech that runs entirely on an Android phone, built on
[MiraTTS](https://huggingface.co/YatharthS/MiraTTS). Record or pick a short clip of a voice, type some
text, and the phone speaks it in that voice: no server, no internet after the one-time model download. Optionally,
a PC with an NVIDIA GPU on your Wi-Fi can do the speaking instead, with bigger models ([On my PC](#on-my-pc-optional)).

Tested on a Samsung Galaxy S23 (Snapdragon 8 Gen 2): about 2.5× slower than real time with 48 kHz output,
so a 8 s sentence takes about 20 s to make.

## Install

Download `MiraVoice-<version>.apk` from the [latest release](https://github.com/ssjrocks/mira-voice/releases/latest)
and open it on the phone (allow installing from your browser or file manager when asked). On first start the
app downloads its models (1.5 GB) from the [models release](https://github.com/ssjrocks/mira-voice/releases/tag/models-1).
Later versions are offered in the app: it checks this repository's releases page on start
(Settings → *Check for updates*).

## Using it

1. **Add a voice** (Voices card), one of three ways:
   - *Record: read the example*: read the sentence shown at your normal pace;
   - *Record speech*: say anything, or play the speech you want to copy (a video, a voice message) near the phone;
   - *Pick file*: any audio file.

   The whole clip is used, up to 25 s (longer clips are cut at a pause). Speech recognition (whisper.cpp,
   `small.en`, on the phone) then writes down what was said; check it word for word against the clip (Play is in
   the same dialog), fix anything misheard, name the voice and save. Voices are kept in a library: pick one to use,
   play its clip, rename, delete, or edit its transcript (*Transcribe again* redoes it from the clip).
2. With **Match accent and speaking style** on, the model continues from the clip's own speech, which keeps
   the speaker's accent (e.g. Australian) instead of the model's default American one. This needs the transcript to
   match the clip **exactly**, including its last word: extra words get spoken at the start of every output, and
   speech missing from the transcript (even a trailing "s") makes outputs come out almost silent. After you save a
   voice or edit its transcript the app test-speaks a short sentence and warns you if they don't match; any part
   that still comes out almost silent is redone without accent matching.
3. Type text and tap **Speak**. The whole text is generated first, then played.

### On my PC (optional)

With a PC with an NVIDIA GPU on the same Wi-Fi, tap **On my PC**: speech is then made on the PC with bigger
models, much faster, and voice transcripts come from Whisper large-v3-turbo. Voices stay in the app; their clips
and transcripts are sent to the PC the first time they're used there, and the text to speak is sent with each request.

| PC voice model | speed on an RTX 5080 | VRAM | output |
|---|---|---|---|
| [dots.tts-soar](https://huggingface.co/dots-studio/dots.tts-soar) (2B, default) | ~4–5× faster than real time | ~7 GB | 48 kHz |
| [Qwen3-TTS 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | ~1.7× faster than real time | ~5 GB | 24 kHz |

Set up the PC once (Linux, [uv](https://docs.astral.sh/uv/), CUDA 13 driver): `server/install.sh` installs the server
as a systemd user service that starts at login, or at boot with `loginctl enable-linger` (port 8790). The app finds it
on the LAN by mDNS, or you type its address. The PC tells the app all its addresses (home network and Tailscale), and
the app keeps them and uses whichever answers, so the same setup works at home and, with Tailscale on the phone, away
from home. Models load on the first request (about a minute;
the app starts this as soon as you switch to the PC) and are unloaded after 15 idle minutes, so the GPU is free
otherwise. There is no pairing or login: traffic is plain HTTP, and anything that can reach port 8790 can use the
server, so keep that port off the internet (LAN and tailnet only).

Each app session writes a log (device, timings, memory, errors) to
`Android/data/dev.local.miravoice/files/logs`; the last 25 sessions are kept.

From 0.4.1, crashes (including native crashes in llama.cpp/ONNX Runtime) and freezes are reported the next time
the app opens, to the developer's server,
with the current and previous session logs attached. Settings → *Report a problem* sends the same with your note.
Reports never include your recordings, voices or the text you speak. Read them with `app-reports` on the build PC.

## How it works

MiraTTS is a fine-tune of [Spark-TTS](https://github.com/SparkAudio/Spark-TTS): a 0.5B Qwen2 language model
that writes BiCodec speech tokens (50 per second of audio), plus a codec and an upsampler.

| stage | file | runs on |
|---|---|---|
| voice clip → 32 voice tokens (timbre) | `speaker_encoder.onnx` (mel + BiCodec speaker encoder) | ONNX Runtime |
| voice clip → its speech tokens (accent mode) | `semantic_encoder.onnx` (wav2vec2-xlsr-53 layers 1–16 + quantiser, fp16 weights) | ONNX Runtime |
| text + voice → speech tokens | `mira.gguf` (Q8_0) + `mira_head.bin` | llama.cpp |
| voice clip → its transcript (voice setup) | `whisper-small.en-q5_1.bin` | whisper.cpp (sharing llama.cpp's ggml) |
| speech tokens → 16 kHz audio | `processer.onnx` + `vocoder.onnx` | ONNX Runtime |
| 16 kHz → 48 kHz | `upsampler.onnx` (FlashSR) | ONNX Runtime |

Things that were needed to make it work and fast on a phone:

- **Speech-only output head.** Only the 8,192 speech tokens and end-of-speech can be produced, so llama.cpp runs
  in "last-token embedding" mode (no 166k-entry LM head) and the engine scores those 8,193 rows itself
  (`native/mira_engine.cpp`), about 30–40% faster per token.
- **f32 attention.** Qwen2's attention activations overflow fp16; on CPUs with native fp16 arithmetic (most
  phones) llama.cpp produced NaN with an f16 KV cache or flash attention.
- **No repacked/KleidiAI weights on ARM.** llama.cpp's ARM weight layouts gave wrong hidden states for this
  model on the Snapdragon 8 Gen 2, while plain Q8_0 kernels (i8mm) match x86 and were the fastest option.
- **Accent mode** (Spark-TTS prompt continuation): the clip's transcript is prepended to the text and the clip's
  own speech tokens start the answer, so generation continues in the speaker's style.

## Building

```bash
git clone --recursive https://github.com/ssjrocks/mira-voice.git
cd mira-voice
```

**App** (Android SDK with NDK 29.0.14206865 and CMake 3.31.6, JDK 17+):

```bash
cd android && ./gradlew assembleRelease
```

Without the models release, push the model files to the app's folder instead:
`adb push <files> /sdcard/Android/data/dev.local.miravoice/files/models/` (open the app once first so it
creates the folder).

**Models** (Python 3.12, CPU is enough; see `export/requirements.txt`): download `YatharthS/MiraTTS` into
`models/MiraTTS`, then from `export/`:

| script | makes |
|---|---|
| `export_onnx.py` | `speaker_encoder.onnx`, `vocoder.onnx`, `upsampler.onnx` |
| `export_split.py` | `processer.onnx` |
| `export_semantic.py`, `weights_fp16.py` | `semantic_encoder.onnx` |
| `export_head.py` | `mira_head.bin` |
| `fix_pow.py` | rewrites `Pow(x, 2)` as `x*x` (needed for Qualcomm's NPU) |
| `upstream/llama.cpp/convert_hf_to_gguf.py` + `llama-quantize … Q8_0` | `mira.gguf` |

**NPU graphs** (experimental, `npu` branch): `export_npu.py` makes fixed-shape vocoder/upsampler models and
`qnn_context.py` compiles them on the PC for the Snapdragon 8 Gen 2's NPU with Qualcomm's `onnxruntime-qnn`, so the
phone doesn't have to (that compile took minutes there and ran the phone out of memory for the upsampler).

`reference.py`, `phone_sim.py`, `verify_onnx.py`, `accent_test.py` check each step against the original
PyTorch model (Whisper word error rate, speaker similarity). `tools/` has the adb test helpers and
`release.sh`.

## Credits and licences

- Code in this repository: MIT (see `LICENSE`).
- [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) by Yatharth Sharma, model weights **CC BY-NC-SA 4.0**:
  the converted model files in the releases are derived from them and carry the same licence (non-commercial,
  share-alike, with attribution). Built on [Spark-TTS / BiCodec](https://github.com/SparkAudio/Spark-TTS).
- [FlashSR](https://github.com/ysharma3501/FlashSR) upsampler, [FastBiCodec](https://github.com/ysharma3501/FastBiCodec).
- [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) (Apache 2.0).
- PC server: [dots.tts](https://github.com/studio-dots-ai/dots.tts) (Apache 2.0), [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
  (Apache 2.0), [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT), [Whisper](https://github.com/openai/whisper) weights (MIT), [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT).

Only clone voices of people who have agreed to it.
