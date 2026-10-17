# Mira Voice

Voice cloning text-to-speech that runs entirely on an Android phone, built on
[MiraTTS](https://huggingface.co/YatharthS/MiraTTS). Record or pick a short clip of a voice, type some
text, and the phone speaks it in that voice: no server, no internet after the one-time model download.

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

Each app session writes a log (device, timings, memory, errors) to
`Android/data/dev.local.miravoice/files/logs`; the last 25 sessions are kept.

From 0.4.1, crashes (including native crashes in llama.cpp/ONNX Runtime) and freezes are reported the next time
the app opens, to `https://ssjrocks.com/api/reports/` ([ssjrocks/app-reports](https://github.com/ssjrocks/app-reports)),
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
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT), [Whisper](https://github.com/openai/whisper) weights (MIT), [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT).

Only clone voices of people who have agreed to it.
