"""Mira Voice PC server: the phone app's "Use PC" mode sends text and voice clips here over the LAN, and this PC's GPU
does the speaking (dots.tts or Qwen3-TTS, both bigger than the phone's model) and the speech recognition (Whisper
large-v3-turbo).

Models load on the first request and unload after IDLE_UNLOAD_S without requests, so the GPU is free for other
work (Ollama, games) the rest of the time. The phone finds the server by mDNS (_miravoice._tcp) or a typed address
(e.g. the PC's Tailscale IP). There is no pairing: anything that can reach the port can use it.

API (JSON unless noted):
  GET  /v1/info                       server, GPU, engines, and the addresses it can be reached at
  GET  /v1/voices/{id}                200 if the voice is stored here, else 404
  PUT  /v1/voices/{id}  {wav_b64, transcript}   stores a voice clip (WAV) and what's said in it
  POST /v1/warmup       {engine}      starts loading an engine (returns at once)
  POST /v1/speak        {voice, text, engine, seed}  -> audio/wav (16-bit mono)
  POST /v1/transcribe   {wav_b64}     -> {text}

usage: python mira_server.py [--host 0.0.0.0] [--port 8790]
"""
import argparse
import asyncio
import base64
import io
import logging
import os
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

VERSION = "1.2.0"
SERVICE_TYPE = "_miravoice._tcp.local."
DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "mira-voice-server"
VOICES = DATA / "voices"
IDLE_UNLOAD_S = int(os.environ.get("MIRA_IDLE_UNLOAD_S", "900"))
MAX_TEXT = 5000
CHUNK_CHARS = 300  # longer texts are spoken in parts of up to this many characters, split between sentences
VOICE_ID = re.compile(r"^[0-9a-f]{16,64}$")

log = logging.getLogger("mira-server")
gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")  # all model work on one thread
last_used = time.time()


def free_vram_gb() -> float:
    """Free GPU memory without creating a CUDA context (nvidia-smi)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0]) / 1024
    except Exception:  # noqa: BLE001
        return 0.0


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip().replace("NVIDIA GeForce ", "")
    except Exception:  # noqa: BLE001
        return "no GPU"


# ------------------------------------------------------------------------------------------------ engines

class Engine:
    id = name = ""
    vram_gb = 0.0
    model = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        raise NotImplementedError

    def unload(self):
        if self.model is None:
            return
        self.model = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        log.info("unloaded %s", self.id)


class DotsEngine(Engine):
    """dots.tts-soar: 2B continuous autoregressive TTS, continuation cloning (clip + its transcript), 48 kHz."""
    id, name, vram_gb = "dots", "dots.tts (best)", 7.5

    def load(self):
        import sys
        from loguru import logger  # dots.tts logs every generation step at DEBUG
        from dots_tts.runtime import DotsTtsRuntime
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        # optimize=True: ~5x faster generation after a warm-up that compiles kernels (minutes the first time,
        # cached on disk afterwards)
        self.model = DotsTtsRuntime.from_pretrained("dots-studio/dots.tts-soar", precision="bfloat16",
                                                    optimize=os.environ.get("MIRA_DOTS_OPTIMIZE", "1") == "1")

    def speak(self, text, voice, seed):
        import torch
        torch.manual_seed(seed)
        r = self.model.generate(text=text, prompt_audio_path=str(voice / "ref.wav"),
                                prompt_text=(voice / "transcript.txt").read_text() or None)
        return r["audio"].float().cpu().squeeze().numpy(), r["sample_rate"]


class QwenEngine(Engine):
    """Qwen3-TTS 1.7B Base: voice cloning from clip + transcript, 24 kHz."""
    id, name, vram_gb = "qwen", "Qwen3-TTS", 5.5
    prompts: dict = {}

    def load(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.prompts = {}
        self.model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                                   dtype=torch.bfloat16, attn_implementation="sdpa")

    def speak(self, text, voice, seed):
        import torch
        torch.manual_seed(seed)
        key = voice.name
        if key not in self.prompts:
            transcript = (voice / "transcript.txt").read_text()
            self.prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=str(voice / "ref.wav"), ref_text=transcript or None, x_vector_only_mode=not transcript)
        wavs, sr = self.model.generate_voice_clone(text=text, language="English", voice_clone_prompt=self.prompts[key])
        return np.asarray(wavs[0], np.float32), sr


class WhisperEngine(Engine):
    id, name, vram_gb = "whisper", "Whisper large-v3-turbo", 2.0

    def load(self):
        import torch
        from transformers import pipeline
        self.model = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                              torch_dtype=torch.float16, device="cuda:0")

    def transcribe(self, wav16k):
        r = self.model({"raw": wav16k, "sampling_rate": 16000}, return_timestamps=True,
                       generate_kwargs={"language": "en", "task": "transcribe"})
        return r["text"].strip()


ENGINES = {e.id: e for e in (DotsEngine(), QwenEngine(), WhisperEngine())}
TTS = ("dots", "qwen")


def ensure_loaded(engine: Engine):
    """Loads [engine], first unloading the other voice model (one at a time, to leave the GPU room for other work)
    and, if the GPU still lacks room, Whisper too. Runs on the GPU thread."""
    if engine.loaded:
        return
    if engine.id in TTS:
        for other in TTS:
            if other != engine.id:
                ENGINES[other].unload()
    need = engine.vram_gb
    if free_vram_gb() < need:
        for other in ENGINES.values():
            if other is not engine:
                other.unload()
    free = free_vram_gb()
    if free < need:
        raise HTTPException(503, f"The PC's GPU is busy: {free:.1f} GB free, {engine.name} needs about {need:.0f} GB. "
                                 "Something else is using it (Ollama, a game?).")
    t = time.time()
    log.info("loading %s (%.1f GB free)", engine.id, free)
    engine.load()
    log.info("loaded %s in %.1f s", engine.id, time.time() - t)


def with_room(engine: Engine, fn, *a):
    """Runs fn(*a) for [engine]; if the GPU runs out of memory (a long clip's cache on top of Whisper, say), unloads the
    other models and tries once more. Runs on the GPU thread."""
    import gc
    import torch
    try:
        return fn(*a)
    except torch.OutOfMemoryError:
        others = [e for e in ENGINES.values() if e is not engine and e.loaded]
        if not others:
            raise
    # retry outside the except block: the exception's traceback holds the failed attempt's tensors
    log.warning("%s ran out of GPU memory; unloading %s and trying again", engine.id, ", ".join(e.id for e in others))
    for e in others:
        e.unload()
    gc.collect()
    torch.cuda.empty_cache()
    return fn(*a)


async def on_gpu(fn, *a):
    global last_used
    last_used = time.time()
    try:
        return await asyncio.get_running_loop().run_in_executor(gpu, fn, *a)
    finally:
        last_used = time.time()


async def idle_unloader():
    while True:
        await asyncio.sleep(60)
        if time.time() - last_used > IDLE_UNLOAD_S and any(e.loaded for e in ENGINES.values()):
            await asyncio.get_running_loop().run_in_executor(gpu, lambda: [e.unload() for e in ENGINES.values()])


# ------------------------------------------------------------------------------------------------ text and audio

def split_text(text: str):
    """Sentences grouped into parts of at most CHUNK_CHARS (a longer sentence is split at commas, then spaces)."""
    sentences = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    parts, cur = [], ""
    for s in filter(None, (s.strip() for s in sentences)):
        while len(s) > CHUNK_CHARS:
            cut = max(s.rfind(", ", 0, CHUNK_CHARS), s.rfind(" ", 0, CHUNK_CHARS))
            cut = cut if cut > CHUNK_CHARS // 3 else CHUNK_CHARS
            head, s = s[:cut + 1].strip(), s[cut + 1:].strip()
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(head)
        if cur and len(cur) + 1 + len(s) > CHUNK_CHARS:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def trim_silence(audio: np.ndarray, sr: int, keep_s=0.25, floor=0.01) -> np.ndarray:
    """Cuts leading/trailing silence beyond [keep_s] (models sometimes add a long pause at the end of a part)."""
    loud = np.flatnonzero(np.abs(audio) > floor)
    if len(loud) == 0:
        return audio
    keep = int(keep_s * sr)
    return audio[max(0, loud[0] - keep):loud[-1] + keep]


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def read_wav16k(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


# ------------------------------------------------------------------------------------------------ HTTP

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(idle_unloader())
    mdns = await start_mdns(app.state.port)
    yield
    if mdns:
        await mdns[0].async_unregister_service(mdns[1])
        await mdns[0].async_close()


class AccessLog:
    """Logs every request with the device's address, so a phone that can't connect shows up (or doesn't) here."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t, status = time.time(), [0]

        async def send_status(msg):
            if msg["type"] == "http.response.start":
                status[0] = msg["status"]
            await send(msg)
        try:
            await self.app(scope, receive, send_status)
        finally:
            log.info("%s %s from %s -> %s (%.1f s)", scope["method"], scope["path"],
                     (scope.get("client") or ["?"])[0], status[0] or "no answer", time.time() - t)


app = FastAPI(title="Mira Voice server", version=VERSION, lifespan=lifespan)
app.add_middleware(AccessLog)


@app.get("/v1/info")
async def info():
    return {"name": "Mira Voice server", "version": VERSION, "host": socket.gethostname(), "gpu": gpu_name(),
            "engines": [{"id": i, "name": ENGINES[i].name, "loaded": ENGINES[i].loaded} for i in TTS],
            "asr": ENGINES["whisper"].name, "free_vram_gb": round(free_vram_gb(), 1),
            "addresses": addresses(app.state.port)}


@app.get("/v1/voices/{vid}")
async def voice_get(vid: str):
    if not VOICE_ID.match(vid) or not (VOICES / vid / "ref.wav").is_file():
        raise HTTPException(404, "Unknown voice")
    return {"id": vid}


@app.put("/v1/voices/{vid}")
async def voice_put(vid: str, request: Request):
    if not VOICE_ID.match(vid):
        raise HTTPException(400, "Bad voice id")
    body = await request.json()
    audio = read_wav16k(base64.b64decode(body["wav_b64"]))
    if len(audio) < 16000:
        raise HTTPException(400, "The clip is shorter than a second")
    d = VOICES / vid
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "ref.wav", audio, 16000, subtype="PCM_16")
    (d / "transcript.txt").write_text(str(body.get("transcript", "")).strip())
    ENGINES["qwen"].prompts.pop(vid, None)
    log.info("stored voice %s: %.1f s", vid, len(audio) / 16000)
    return {"id": vid}


@app.post("/v1/warmup")
async def warmup(request: Request):
    """Starts loading an engine without waiting, so it's ready by the time the phone sends text."""
    body = await request.json()
    engine = ENGINES.get(body.get("engine", "dots"))
    if engine is None or engine.id not in TTS:
        raise HTTPException(400, f"Unknown engine; choose one of {', '.join(TTS)}")
    if not engine.loaded:
        task = asyncio.create_task(on_gpu(ensure_loaded, engine))
        task.add_done_callback(lambda t: t.exception() and log.warning("warm-up of %s failed: %s", engine.id, t.exception()))
    return {"engine": engine.id, "loaded": engine.loaded}


@app.post("/v1/speak")
async def speak(request: Request):
    body = await request.json()
    text = str(body.get("text", "")).strip()[:MAX_TEXT]
    engine = ENGINES.get(body.get("engine", "dots"))
    voice = VOICES / str(body.get("voice", ""))
    if not text:
        raise HTTPException(400, "No text")
    if engine is None or engine.id not in TTS:
        raise HTTPException(400, f"Unknown engine; choose one of {', '.join(TTS)}")
    if not VOICE_ID.match(voice.name) or not (voice / "ref.wav").is_file():
        raise HTTPException(404, "Unknown voice")
    seed = int(body.get("seed", 0))
    t0 = time.time()
    await on_gpu(ensure_loaded, engine)
    load_s = time.time() - t0
    out, sr = [], 48000
    parts = split_text(text)
    for i, part in enumerate(parts):
        if await request.is_disconnected():
            log.info("speak cancelled by the phone after %d of %d parts", i, len(parts))
            return Response(status_code=499)
        a, sr = await on_gpu(with_room, engine, engine.speak, part, voice, seed + i)
        out.append(trim_silence(a, sr))
        out.append(np.zeros(int(sr * 0.2), np.float32))
    audio = np.concatenate(out[:-1])
    gen_s = time.time() - t0 - load_s
    log.info("spoke %d chars in %d parts with %s: %.1f s of audio in %.1f s (+%.1f s loading)",
             len(text), len(parts), engine.id, len(audio) / sr, gen_s, load_s)
    return Response(wav_bytes(audio, sr), media_type="audio/wav", headers={
        "X-Audio-Seconds": f"{len(audio) / sr:.2f}", "X-Gen-Seconds": f"{gen_s:.2f}",
        "X-Load-Seconds": f"{load_s:.2f}", "X-Parts": str(len(parts))})


@app.post("/v1/transcribe")
async def transcribe(request: Request):
    body = await request.json()
    audio = read_wav16k(base64.b64decode(body["wav_b64"]))
    w = ENGINES["whisper"]
    t0 = time.time()
    await on_gpu(ensure_loaded, w)
    text = await on_gpu(with_room, w, w.transcribe, audio)
    log.info("transcribed %.1f s in %.1f s", len(audio) / 16000, time.time() - t0)
    return {"text": text}


@app.exception_handler(HTTPException)
async def http_error(_, e: HTTPException):
    return JSONResponse({"error": e.detail}, status_code=e.status_code)


# ------------------------------------------------------------------------------------------------ mDNS and main

def lan_ip() -> str:
    """Source address of the default route (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def addresses(port: int) -> list[str]:
    """Where the phone can reach this server: the home network address, then Tailscale's (for when it's away)."""
    ips = [lan_ip()]
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True, timeout=5).stdout
        ips += [f[3].split("/")[0] for f in (line.split() for line in out.splitlines()) if f[1].startswith("tailscale")]
    except Exception:  # noqa: BLE001
        pass
    return [f"{ip}:{port}" for ip in dict.fromkeys(ips) if not ip.startswith("127.")]


async def start_mdns(port: int):
    """Advertises the server so the app finds it without typing an address."""
    ip = lan_ip()
    try:
        from zeroconf import IPVersion, ServiceInfo
        from zeroconf.asyncio import AsyncZeroconf
        host = socket.gethostname().split(".")[0].lower()
        info = ServiceInfo(SERVICE_TYPE, f"Mira Voice on {host}"[:60] + "." + SERVICE_TYPE,
                           addresses=[socket.inet_aton(ip)], port=port, properties={"version": VERSION},
                           server=f"miravoice-{host}.local.")
        azc = AsyncZeroconf(interfaces=[ip], ip_version=IPVersion.V4Only)
        await azc.async_register_service(info, allow_name_change=True)
        log.info("advertising %s on %s:%d", SERVICE_TYPE, ip, port)
        return azc, info
    except Exception as e:  # noqa: BLE001
        log.warning("mDNS advertisement failed (%s); type the address %s:%d in the app", e, ip, port)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    VOICES.mkdir(parents=True, exist_ok=True)
    app.state.port = args.port
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
