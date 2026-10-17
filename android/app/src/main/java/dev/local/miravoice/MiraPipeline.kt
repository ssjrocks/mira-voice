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
            val prompted = voice.accentMode
            val llmText = if (prompted) voice.transcript!!.trim() + " " + chunk else chunk
            val codes = MiraNative.generate(llm, llmText, voice.tokens, p.temperature, p.topK, p.topP, p.minP,
                p.repeatPenalty, maxTokens, p.seed + i, if (prompted) voice.promptCodes else null, { _, n ->
                    if (n % 25 == 0) onStatus("${label}generating speech… %.1f s".format(n / 50.0))
                    !isCancelled()
                }, raw)
            st.promptTokens = raw[0].toInt()
            st.speechTokens = codes.size
            st.prefillMs = raw[2]
            st.llmMs = raw[3]
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

    private fun decode(codes: IntArray, voiceTokens: IntArray): FloatArray {
        val c = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(codes.size) { codes[it].toLong() }),
            longArrayOf(1, codes.size.toLong()))
        val v = OnnxTensor.createTensor(env, IntBuffer.wrap(voiceTokens), longArrayOf(1, 1, voiceTokens.size.toLong()))
        val feats = c.use { v.use { processer.run(mapOf("speech_tokens" to c, "context_tokens" to v)).use { floats(it) } } }
        val n = codes.size
        OnnxTensor.createTensor(env, FloatBuffer.wrap(feats), longArrayOf(1, FEAT_DIM.toLong(), n.toLong())).use { t ->
            vocoder.run(mapOf("features" to t)).use { return floats(it) }
        }
    }

    private fun upsample(wav16: FloatArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(wav16), longArrayOf(1, 1, wav16.size.toLong())).use { t ->
            upsampler.run(mapOf("wav16" to t)).use { return floats(it) }
        }
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
        const val PROMPT_MAX_SAMPLES = 12 * 16000 // accent mode: at most 12 s of clip (~600 codes per prompt)
        private const val FEAT_DIM = 1024

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
