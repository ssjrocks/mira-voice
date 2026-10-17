package dev.local.miravoice

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper
import kotlin.concurrent.thread

/**
 * Plays one clip at a time: starting a clip stops whatever was playing. [onChange] runs on the main
 * thread whenever playback starts or ends, so buttons can switch between Play and Stop.
 */
class ClipPlayer(private val onChange: () -> Unit) {
    private val main = Handler(Looper.getMainLooper())
    private var track: AudioTrack? = null

    /** Which clip is playing ("voice", "output"), or null. */
    var playing: String? = null
        private set

    fun play(id: String, samples: FloatArray, rate: Int) {
        stop()
        val minBuf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val t = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(minBuf, rate * 4 / 2))
            .build()
        track = t
        playing = id
        onChange()
        t.notificationMarkerPosition = samples.size
        t.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
            override fun onMarkerReached(at: AudioTrack) {
                if (track === at) stop()
            }

            override fun onPeriodicNotification(at: AudioTrack) {}
        }, main)
        t.play()
        thread(name = "mira-play") {
            var off = 0
            while (off < samples.size && track === t) {
                val n = t.write(samples, off, minOf(rate / 4, samples.size - off), AudioTrack.WRITE_BLOCKING)
                if (n <= 0) break
                off += n
            }
        }
    }

    fun stop() {
        val t = track ?: return
        track = null
        playing = null
        runCatching { t.pause(); t.flush(); t.stop() }
        t.release()
        if (Looper.myLooper() == Looper.getMainLooper()) onChange() else main.post(onChange)
    }
}
