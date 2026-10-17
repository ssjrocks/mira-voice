package dev.local.miravoice

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

object AudioIO {

    /** Decodes any audio file Android can read to mono float PCM at [targetRate], first [maxSeconds] only. */
    fun decodeToMono(context: Context, uri: Uri, targetRate: Int = 16000, maxSeconds: Float = 30f): FloatArray {
        val ex = MediaExtractor()
        ex.setDataSource(context, uri, null)
        val track = (0 until ex.trackCount).firstOrNull {
            ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: throw IllegalArgumentException("no audio track in file")
        ex.selectTrack(track)
        val fmt = ex.getTrackFormat(track)
        val codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(fmt, null, null, 0)
        codec.start()

        var rate = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var encoding = AudioFormat.ENCODING_PCM_16BIT
        val mono = FloatList()
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        while (!outputDone) {
            if (!inputDone) {
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx >= 0) {
                    val buf = codec.getInputBuffer(inIdx)!!
                    val n = ex.readSampleData(buf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, n, ex.sampleTime, 0)
                        ex.advance()
                    }
                }
            }
            val outIdx = codec.dequeueOutputBuffer(info, 10_000)
            if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val of = codec.outputFormat
                rate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = of.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } else if (outIdx >= 0) {
                val buf = codec.getOutputBuffer(outIdx)!!.order(ByteOrder.LITTLE_ENDIAN)
                buf.position(info.offset)
                buf.limit(info.offset + info.size)
                if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
                    val fb = buf.asFloatBuffer()
                    while (fb.remaining() >= channels) {
                        var s = 0f
                        repeat(channels) { s += fb.get() }
                        mono.add(s / channels)
                    }
                } else {
                    val sb = buf.asShortBuffer()
                    while (sb.remaining() >= channels) {
                        var s = 0f
                        repeat(channels) { s += sb.get() / 32768f }
                        mono.add(s / channels)
                    }
                }
                codec.releaseOutputBuffer(outIdx, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                if (mono.size > (maxSeconds * rate).toInt()) outputDone = true
            }
        }
        codec.stop()
        codec.release()
        ex.release()
        val pcm = mono.toArray().let { it.copyOf(min(it.size, (maxSeconds * rate).toInt())) }
        return resample(pcm, rate, targetRate)
    }

    /** Windowed-sinc resampler (Hann window, 32 zero crossings), anti-aliased when downsampling. */
    fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || x.isEmpty()) return x
        val ratio = to.toDouble() / from
        val cutoff = min(1.0, ratio) * 0.95
        val half = 32 / cutoff
        val out = FloatArray(floor(x.size * ratio).toInt())
        for (i in out.indices) {
            val t = i / ratio
            val lo = max(0, ceil(t - half).toInt())
            val hi = min(x.size - 1, floor(t + half).toInt())
            var acc = 0.0
            for (j in lo..hi) {
                val d = t - j
                val sinc = if (abs(d) < 1e-9) 1.0 else sin(PI * cutoff * d) / (PI * cutoff * d)
                val w = 0.5 + 0.5 * cos(PI * d / half)
                acc += x[j] * cutoff * sinc * w
            }
            out[i] = acc.toFloat()
        }
        return out
    }

    /** Drops leading/trailing near-silence (20 ms frames below 2% of the loudest frame's RMS). */
    fun trimSilence(x: FloatArray, rate: Int = 16000): FloatArray {
        val frame = rate / 50
        val n = x.size / frame
        if (n < 3) return x
        val rms = FloatArray(n) { f ->
            var s = 0f
            for (k in f * frame until (f + 1) * frame) s += x[k] * x[k]
            sqrt(s / frame)
        }
        val thresh = rms.max() * 0.02f
        val first = rms.indexOfFirst { it > thresh }
        val last = rms.indexOfLast { it > thresh }
        if (first < 0) return x
        val a = max(0, (first - 5) * frame)          // keep 100 ms of context
        val b = min(x.size, (last + 6) * frame)
        return x.copyOfRange(a, b)
    }

    /** Records 16 kHz mono from the mic until [keepGoing] returns false or [maxSeconds] pass. */
    @SuppressLint("MissingPermission")
    fun record(maxSeconds: Float = 15f, keepGoing: () -> Boolean, onLevel: (Float, Float) -> Unit): FloatArray {
        val rate = 16000
        val minBuf = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, max(minBuf, rate))
        val out = FloatList()
        val buf = ShortArray(rate / 10)
        rec.startRecording()
        try {
            while (keepGoing() && out.size < maxSeconds * rate) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) continue
                var peak = 0f
                for (k in 0 until n) {
                    val v = buf[k] / 32768f
                    out.add(v)
                    peak = max(peak, abs(v))
                }
                onLevel(peak, out.size / rate.toFloat())
            }
        } finally {
            rec.stop()
            rec.release()
        }
        return out.toArray()
    }

    fun wavBytes(samples: FloatArray, rate: Int): ByteArray {
        val data = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (s in samples) data.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt(36 + data.capacity()).put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2)
            .putShort(2).putShort(16)
        h.put("data".toByteArray()).putInt(data.capacity())
        return ByteArrayOutputStream().apply { write(h.array()); write(data.array()) }.toByteArray()
    }

    fun writeWav(file: File, samples: FloatArray, rate: Int) = file.writeBytes(wavBytes(samples, rate))

    /** Saves to Music/MiraVoice via MediaStore; returns the content Uri. */
    fun saveToMusic(context: Context, name: String, samples: FloatArray, rate: Int): Uri {
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, name)
            put(MediaStore.Audio.Media.MIME_TYPE, "audio/wav")
            put(MediaStore.Audio.Media.RELATIVE_PATH, Environment.DIRECTORY_MUSIC + "/MiraVoice")
        }
        val uri = context.contentResolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("could not create file in Music")
        context.contentResolver.openOutputStream(uri).use { os: OutputStream? -> os!!.write(wavBytes(samples, rate)) }
        return uri
    }

    fun readWav16(file: File): FloatArray {
        val b = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        b.position(44)
        val sb = b.asShortBuffer()
        return FloatArray(sb.remaining()) { sb.get() / 32768f }
    }
}

/** Growable float buffer without boxing. */
class FloatList(cap: Int = 1 shl 16) {
    private var a = FloatArray(cap)
    var size = 0
        private set

    fun add(v: Float) {
        if (size == a.size) a = a.copyOf(a.size * 2)
        a[size++] = v
    }

    fun addAll(v: FloatArray) {
        if (size + v.size > a.size) a = a.copyOf(max(a.size * 2, size + v.size))
        System.arraycopy(v, 0, a, size, v.size)
        size += v.size
    }

    fun toArray(): FloatArray = a.copyOf(size)
}
