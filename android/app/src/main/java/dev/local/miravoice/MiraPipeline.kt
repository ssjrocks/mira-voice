package dev.local.miravoice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class GenParams(
    val temperature: Float = 0.8f,
    val topK: Int = 50,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.2f,
    val maxTokens: Int = 1024,
    val seed: Int = 0,
)

/**
 * A cloned voice. [tokens] (32 "context" tokens) carry the timbre. If the clip's own speech codes and
 * transcript are known, generation continues from them, which carries the speaker's accent and pacing too.
 */
class Voice(val tokens: IntArray, val promptCodes: IntArray?, val transcript: String?) {
    val accentMode get() = promptCodes != null && promptCodes.isNotEmpty() && !transcript.isNullOrBlank()
}

/** Per-chunk timings. */
class ChunkStats(val text: String) {
    /** Accent matching was dropped for this part because it came out almost silent (clip/transcript mismatch). */
    var accentFallback = false
    var promptTokens = 0
    var speechTokens = 0
    var prefillMs = 0.0
    var llmMs = 0.0
    var decodeMs = 0.0
    var upsampleMs = 0.0
    val audioSec get() = speechTokens / 50.0
}

/**
 * The MiraTTS pipeline:
 *   voice clip -> speaker_encoder.onnx -> 32 voice tokens   (+ semantic_encoder.onnx -> clip speech codes)
 *   text + voice -> LLM (llama.cpp) -> speech codes (50 per second of audio)
 *   speech codes + voice tokens -> processer.onnx -> vocoder.onnx -> 16 kHz -> upsampler.onnx -> 48 kHz
 */
class MiraPipeline(
    private val dir: File, val gguf: String, val llmThreads: Int, val onnxThreads: Int,
) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val speaker: OrtSession
    private val processer: OrtSession
    private val vocoder: OrtSession
    private val upsampler: OrtSession
    private val llm: Long
    val loadMs: Long

    private fun opts() = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(onnxThreads)
        // No memory arena: it kept the peak of every decode (6+ GB after a 20 s clip on a 7 GB phone).
        setCPUArenaAllocator(false)
        setMemoryPatternOptimization(false)
    }

    init {
        val t0 = SystemClock.elapsedRealtime()
        speaker = env.createSession(File(dir, ModelFiles.SPEAKER).path, opts())
        processer = env.createSession(File(dir, ModelFiles.PROCESSER).path, opts())
        vocoder = env.createSession(File(dir, ModelFiles.VOCODER).path, opts())
        upsampler = env.createSession(File(dir, ModelFiles.UPSAMPLER).path, opts())
        llm = MiraNative.load(File(dir, gguf).path, File(dir, ModelFiles.HEAD).path, llmThreads)
        loadMs = SystemClock.elapsedRealtime() - t0
    }

    /** 16 kHz mono clip -> the 32 voice tokens. Mirrors FastBiCodec's AudioEncoder.encode. */
    fun encodeVoiceTokens(wav16k: FloatArray): IntArray {
        val clip = tileTo(volumeNormalize(wav16k.copyOf(min(wav16k.size, 8 * 16000))), REF_SAMPLES)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(clip), longArrayOf(1, REF_SAMPLES.toLong())).use { t ->
            speaker.run(mapOf("wav" to t)).use { r -> return ints(r) }
        }
    }

    /**
     * The clip's own speech codes (wav2vec2 + quantiser), for accent mode. Loads the ~0.5 GB encoder only
     * for this call. [wav16k] must be exactly the audio the transcript describes.
     */
    fun encodePromptCodes(wav16k: FloatArray): IntArray {
        val clip = volumeNormalize(wav16k)
        env.createSession(File(dir, ModelFiles.SEMANTIC).path, opts()).use { s ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(clip), longArrayOf(1, clip.size.toLong())).use { t ->
                s.run(mapOf("wav" to t)).use { r ->
                    val b = (r[0] as OnnxTensor).longBuffer
                    return IntArray(b.remaining()) { b.get().toInt() }
                }
            }
        }
    }

    /** What was said in a 16 kHz clip (speech recognition), or null if the model isn't downloaded. */
    fun transcribe(wav16k: FloatArray): String? {
        val f = File(dir, ModelFiles.ASR)
        if (!f.isFile) return null
        val h = MiraNative.asrLoad(f.path)  // ~0.3 GB, only for this call
        try {
            return MiraNative.asrTranscribe(h, wav16k, llmThreads).trim()
        } finally {
            MiraNative.asrFree(h)
        }
    }

    /**
     * Speaks [text]: chunk by chunk (sentences, merged up to ~200 characters) through the LLM and decoder,
     * returning all the audio at once (16 or 48 kHz, peak-normalised).
     */
    fun synthesize(
        text: String, voice: Voice, hd: Boolean, p: GenParams,
        isCancelled: () -> Boolean,
        onStatus: (String) -> Unit,
        stats: MutableList<ChunkStats>,
    ): FloatArray {
        val chunks = splitText(text)
        val out = FloatList()
        val rate = if (hd) 48000 else 16000
        for ((i, chunk) in chunks.withIndex()) {
            if (isCancelled()) break
            val st = ChunkStats(chunk).also { stats.add(it) }
            val label = if (chunks.size > 1) "Part ${i + 1} of ${chunks.size}: " else ""
            onStatus("${label}generating speech…")
            val raw = DoubleArray(4)
            // Guard against the model never emitting end-of-speech: ~3.5 codes per character is typical.
            val maxTokens = min(p.maxTokens, 100 + 12 * chunk.length)
            val gen = { prompted: Boolean ->
                val llmText = if (prompted) voice.transcript!!.trim() + " " + chunk else chunk
                MiraNative.generate(llm, llmText, voice.tokens, p.temperature, p.topK, p.topP, p.minP,
                    p.repeatPenalty, maxTokens, p.seed + i, if (prompted) voice.promptCodes else null, { _, n ->
                        if (n % 25 == 0) onStatus("${label}generating speech… %.1f s".format(n / 50.0))
                        !isCancelled()
                    }, raw)
            }
            var codes = gen(voice.accentMode)
            st.promptTokens = raw[0].toInt()
            st.prefillMs = raw[2]
            st.llmMs = raw[3]
            // Almost no speech in accent mode means the clip and its transcript don't match (the model thinks
            // the text was already spoken): redo this part from the voice tokens alone.
            if (voice.accentMode && !isCancelled() && codes.size < 0.3 * expectedCodes(chunk)) {
                onStatus("${label}accent matching failed for this part, retrying without it…")
                codes = gen(false)
                st.accentFallback = true
                st.promptTokens += raw[0].toInt()
                st.prefillMs += raw[2]
                st.llmMs += raw[3]
            }
            st.speechTokens = codes.size
            if (isCancelled() || codes.isEmpty()) break
            onStatus("${label}decoding audio…")
            var t = SystemClock.elapsedRealtimeNanos()
            val wav16 = decode(codes, voice.tokens)
            st.decodeMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
            t = SystemClock.elapsedRealtimeNanos()
            val audio = if (hd) upsample(wav16) else wav16
            st.upsampleMs = if (hd) (SystemClock.elapsedRealtimeNanos() - t) / 1e6 else 0.0
            if (out.size > 0) out.addAll(FloatArray(rate / 8))  // 125 ms between parts
            out.addAll(audio)
        }
        return peakNormalize(out.toArray())
    }

    /**
     * Test-speaks a short sentence and compares its length with what the text needs: about 1x when the clip's
     * transcript matches its audio, under 0.6x when the clip holds more speech than the transcript, over 2x when the
     * transcript has words the clip doesn't (they leak into every output). Returns the ratio.
     */
    fun checkVoice(voice: Voice): Double {
        val raw = DoubleArray(4)
        val codes = MiraNative.generate(llm, voice.transcript!!.trim() + " " + CHECK_TEXT, voice.tokens, 0.8f, 50, 0.95f,
            0.05f, 1.2f, 600, 0, voice.promptCodes, null, raw)
        return codes.size / expectedCodes(CHECK_TEXT)
    }

    /** Speech codes -> 16 kHz audio, in windows of up to 8 s so memory stays small for long texts. */
    private fun decode(codes: IntArray, voiceTokens: IntArray): FloatArray {
        val n = codes.size
        if (n <= DEC_WINDOW) return decodeRange(codes, voiceTokens, 0, n)
        val out = FloatList()
        var s = 0
        while (s < n) {
            val e = min(n, s + DEC_WINDOW)
            out.addAll(decodeRange(codes, voiceTokens, s, e))
            s = e
        }
        return out.toArray()
    }

    /** Audio for codes [s, e), decoded with enough context either side to join seamlessly (checked on the PC:
     *  the processer needs ~96 codes of past and 32 of future context, the vocoder 8/4). */
    private fun decodeRange(codes: IntArray, voiceTokens: IntArray, s: Int, e: Int): FloatArray {
        val pa = max(0, s - 96)
        val pb = min(codes.size, e + 32)
        val n = pb - pa
        val c = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n) { codes[pa + it].toLong() }), longArrayOf(1, n.toLong()))
        val v = OnnxTensor.createTensor(env, IntBuffer.wrap(voiceTokens), longArrayOf(1, 1, voiceTokens.size.toLong()))
        val feats = c.use { v.use { processer.run(mapOf("speech_tokens" to c, "context_tokens" to v)).use { floats(it) } } }
        val va = max(pa, s - 8)
        val vb = min(pb, e + 4)
        val tv = vb - va
        val f = if (va == pa && vb == pb) feats else FloatArray(FEAT_DIM * tv).also {
            for (ch in 0 until FEAT_DIM) System.arraycopy(feats, ch * n + (va - pa), it, ch * tv, tv)
        }
        val wav = OnnxTensor.createTensor(env, FloatBuffer.wrap(f), longArrayOf(1, FEAT_DIM.toLong(), tv.toLong())).use { t ->
            vocoder.run(mapOf("features" to t)).use { floats(it) }
        }
        return wav.copyOfRange((s - va) * HOP, (e - va) * HOP)
    }

    /** 16 -> 48 kHz in windows of up to 8 s (the upsampler is local; 20 ms of context either side is plenty). */
    private fun upsample(wav16: FloatArray): FloatArray {
        val n = wav16.size
        val out = FloatList(n * 3)
        var s = 0
        while (s < n) {
            val e = min(n, s + UP_WINDOW)
            val a = max(0, s - UP_CTX)
            val b = min(n, e + UP_CTX)
            val up = OnnxTensor.createTensor(env, FloatBuffer.wrap(wav16.copyOfRange(a, b)), longArrayOf(1, 1, (b - a).toLong())).use { t ->
                upsampler.run(mapOf("wav16" to t)).use { floats(it) }
            }
            out.addAll(up.copyOfRange((s - a) * 3, (e - a) * 3))
            s = e
        }
        return out.toArray()
    }

    override fun close() {
        speaker.close()
        processer.close()
        vocoder.close()
        upsampler.close()
        MiraNative.free(llm)
    }

    companion object {
        const val REF_SAMPLES = 96000           // the speaker encoder sees exactly 6 s
        /** Longest voice clip used (accent mode re-reads it for every part of a text: ~0.4 s per clip second there). */
        const val MAX_CLIP_SAMPLES = 25 * 16000
        private const val FEAT_DIM = 1024
        private const val HOP = 320                // 16 kHz samples per speech code
        private const val DEC_WINDOW = 400         // codes decoded per window (8 s)
        private const val UP_WINDOW = 8 * 16000
        private const val UP_CTX = 320
        private const val CHECK_TEXT = "This is a quick test of my voice."

        /** Rough number of speech codes a text needs (~15 characters per second, 50 codes per second). */
        fun expectedCodes(text: String) = text.length / 15.0 * 50

        private fun floats(r: OrtSession.Result): FloatArray {
            val b = (r[0] as OnnxTensor).floatBuffer
            return FloatArray(b.remaining()).also { b.get(it) }
        }

        private fun ints(r: OrtSession.Result): IntArray {
            val b = (r[0] as OnnxTensor).intBuffer
            return IntArray(b.remaining()) { b.get() }
        }

        /** Scales to a 0.95 peak (upstream peak-normalises its 48 kHz output). */
        fun peakNormalize(x: FloatArray): FloatArray {
            var peak = 1e-6f
            for (s in x) peak = max(peak, abs(s))
            val g = 0.95f / peak
            return FloatArray(x.size) { x[it] * g }
        }

        /** Port of FastBiCodec's audio_volume_normalize (coeff 0.2), quirks included: the loudness
         *  estimate uses the magnitudes from before the quiet-clip boost, as upstream does. */
        fun volumeNormalize(audio: FloatArray, coeff: Float = 0.2f): FloatArray {
            var a = audio
            val sorted = FloatArray(a.size) { abs(a[it]) }.also { it.sort() }
            if (sorted.isEmpty()) return a
            if (sorted.last() < 0.1f) {
                val s = max(sorted.last(), 1e-3f)
                a = FloatArray(a.size) { a[it] / s * 0.1f }
            }
            val loud = sorted.filter { it > 0.01f }
            val n = loud.size
            if (n <= 10) return a
            var sum = 0.0
            val from = (0.9 * n).toInt()
            val to = (0.99 * n).toInt()
            for (k in from until to) sum += loud[k]
            val volume = (sum / max(1, to - from)).toFloat()
            val g = (coeff / volume).coerceIn(0.1f, 10f)
            val out = FloatArray(a.size) { a[it] * g }
            var peak = 0f
            for (v in out) peak = max(peak, abs(v))
            if (peak > 1f) for (k in out.indices) out[k] /= peak
            return out
        }

        /** Repeats the clip until it covers [n] samples, then cuts to n (upstream get_ref_clip). */
        fun tileTo(x: FloatArray, n: Int): FloatArray {
            if (x.isEmpty()) return FloatArray(n)
            return FloatArray(n) { x[it % x.size] }
        }

        /** Sentence split like upstream split_text, then merge neighbours up to ~200 characters per chunk. */
        fun splitText(text: String): List<String> {
            val sentences = text.trim().split(Regex("(?<=[.!?])\\s+")).map { it.trim() }.filter { it.isNotEmpty() }
            val out = mutableListOf<String>()
            for (s in sentences) {
                if (out.isNotEmpty() && out.last().length + s.length < 200) out[out.size - 1] = out.last() + " " + s
                else out.add(s)
            }
            return out
        }
    }
}
