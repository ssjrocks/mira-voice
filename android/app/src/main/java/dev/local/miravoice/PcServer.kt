package dev.local.miravoice

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Base64
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume

/**
 * "On my PC" mode: the Mira Voice server on the user's PC (server/mira_server.py) speaks and transcribes with bigger
 * models on its GPU, over the LAN or Tailscale. The server is found by mDNS (_miravoice._tcp) or a typed address; there
 * is no pairing, the address is all it needs. Network calls block: use them off the main thread.
 */
class PcServer(context: Context) {
    private val prefs = context.getSharedPreferences("pc", Context.MODE_PRIVATE)
    private val nsd = context.getSystemService(NsdManager::class.java)

    data class Info(val host: String, val gpu: String, val version: String, val engines: List<Pair<String, String>>)
    class Spoken(val audio: FloatArray, val rate: Int, val genSeconds: Double, val loadSeconds: Double, val parts: Int)

    /** "host:port" of the PC server, or null before one is found. */
    var address: String?
        get() = prefs.getString("address", null)
        set(v) = prefs.edit().putString("address", v).apply()
    val connected get() = address != null
    var engine: String
        get() = prefs.getString("engine", "dots") ?: "dots"
        set(v) = prefs.edit().putString("engine", v).apply()

    fun forget() = prefs.edit().remove("token").remove("address").apply()

    fun info(addr: String = address ?: throw IOException("No PC set")): Info {
        val j = JSONObject(String(request("GET", "/v1/info", null, addr = addr, timeoutMs = 4000)))
        val engines = j.getJSONArray("engines").let { a -> (0 until a.length()).map { a.getJSONObject(it).let { e -> e.getString("id") to e.getString("name") } } }
        return Info(j.getString("host"), j.getString("gpu"), j.getString("version"), engines)
    }

    /** Starts loading the voice model on the PC (it takes up to a minute), so it's ready when text arrives. */
    fun warmup() = request("POST", "/v1/warmup", JSONObject().put("engine", engine), timeoutMs = 8000)

    /** Speaks [text] in the voice whose clip is [refWav] (uploaded first if the PC doesn't have it). */
    fun speak(refWav: File, transcript: String, text: String, seed: Int, cancelled: () -> Boolean): Spoken {
        val id = ensureVoice(refWav, transcript)
        val conn = open("POST", "/v1/speak", JSONObject().put("voice", id).put("text", text).put("engine", engine).put("seed", seed),
            timeoutMs = 10 * 60_000)
        // Stop: closing the connection makes the PC stop after the part it's on
        val watcher = Thread {
            while (!Thread.currentThread().isInterrupted) {
                if (cancelled()) { conn.disconnect(); break }
                try { Thread.sleep(200) } catch (_: InterruptedException) { break }
            }
        }.apply { isDaemon = true; start() }
        try {
            val body = read(conn)
            val (audio, rate) = parseWav(body)
            return Spoken(audio, rate, conn.getHeaderField("X-Gen-Seconds")?.toDoubleOrNull() ?: 0.0,
                conn.getHeaderField("X-Load-Seconds")?.toDoubleOrNull() ?: 0.0, conn.getHeaderField("X-Parts")?.toIntOrNull() ?: 1)
        } finally {
            watcher.interrupt()
        }
    }

    /** Speech recognition on the PC (Whisper large-v3-turbo). */
    fun transcribe(wav16k: FloatArray): String {
        val body = JSONObject().put("wav_b64", Base64.encodeToString(AudioIO.wavBytes(wav16k, 16000), Base64.NO_WRAP))
        return JSONObject(String(request("POST", "/v1/transcribe", body, timeoutMs = 3 * 60_000))).getString("text")
    }

    /** The PC keeps voices under a hash of clip + transcript, so an edited transcript is sent again. */
    private fun ensureVoice(refWav: File, transcript: String): String {
        val wav = refWav.readBytes()
        val id = MessageDigest.getInstance("SHA-256").digest(wav + "\n$transcript".toByteArray())
            .joinToString("") { "%02x".format(it) }.take(24)
        val c = open("GET", "/v1/voices/$id", null, timeoutMs = 8000)
        if (c.responseCode == 404) {
            c.disconnect()
            request("PUT", "/v1/voices/$id", JSONObject().put("wav_b64", Base64.encodeToString(wav, Base64.NO_WRAP))
                .put("transcript", transcript), timeoutMs = 60_000)
            SessionLog.i("pc", "uploaded voice $id (%.1f s)".format((wav.size - 44) / 32000f))
        } else read(c)
        return id
    }

    // ---------------------------------------------------------------------------------------- HTTP

    private fun open(method: String, path: String, body: JSONObject?, addr: String = address ?: throw IOException("No PC set"),
                     timeoutMs: Int = 15_000): HttpURLConnection {
        val c = URL("http://$addr$path").openConnection() as HttpURLConnection
        c.requestMethod = method
        c.connectTimeout = 4000
        c.readTimeout = timeoutMs
        if (body != null) {
            c.doOutput = true
            c.setRequestProperty("Content-Type", "application/json")
            c.outputStream.use { it.write(body.toString().toByteArray()) }
        }
        return c
    }

    private fun read(c: HttpURLConnection): ByteArray {
        try {
            val code = c.responseCode
            if (code in 200..299) return c.inputStream.use { it.readBytes() }
            val msg = runCatching { JSONObject(String(c.errorStream.use { it.readBytes() })).getString("error") }.getOrNull()
            throw IOException(msg ?: "The PC answered HTTP $code")
        } finally {
            c.disconnect()
        }
    }

    private fun request(method: String, path: String, body: JSONObject?, addr: String = address ?: throw IOException("No PC set"),
                        timeoutMs: Int = 15_000) = read(open(method, path, body, addr, timeoutMs))

    /** 16-bit PCM WAV (what the server sends) -> samples and sample rate. */
    private fun parseWav(b: ByteArray): Pair<FloatArray, Int> {
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12
        var rate = 48000
        var channels = 1
        while (pos + 8 <= b.size) {
            val id = String(b, pos, 4)
            val len = bb.getInt(pos + 4)
            if (id == "fmt ") {
                channels = bb.getShort(pos + 10).toInt()
                rate = bb.getInt(pos + 12)
            } else if (id == "data") {
                val n = minOf(len, b.size - pos - 8) / 2 / channels
                return FloatArray(n) { bb.getShort(pos + 8 + it * 2 * channels) / 32768f } to rate
            }
            pos += 8 + len + (len and 1)
        }
        throw IOException("The PC sent audio this app can't read")
    }

    // ---------------------------------------------------------------------------------------- discovery

    /** Looks for the server on the LAN by mDNS; returns "host:port", or null after [timeoutMs]. */
    suspend fun discover(timeoutMs: Long = 6000): String? = withTimeoutOrNull(timeoutMs) {
        suspendCancellableCoroutine { cont ->
            var listener: NsdManager.DiscoveryListener? = null
            val done = AtomicBoolean(false)  // several resolves can finish at once
            fun finish(addr: String?) {
                if (!done.compareAndSet(false, true)) return
                listener?.let { l -> runCatching { nsd.stopServiceDiscovery(l) } }
                listener = null
                cont.resume(addr)
            }
            val l = object : NsdManager.DiscoveryListener {
                override fun onDiscoveryStarted(serviceType: String) {}
                override fun onDiscoveryStopped(serviceType: String) {}
                override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) = finish(null)
                override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {}
                override fun onServiceLost(info: NsdServiceInfo) {}
                override fun onServiceFound(info: NsdServiceInfo) {
                    @Suppress("DEPRECATION")  // resolveService is deprecated on API 34 but works everywhere
                    nsd.resolveService(info, object : NsdManager.ResolveListener {
                        override fun onResolveFailed(si: NsdServiceInfo, errorCode: Int) {}
                        override fun onServiceResolved(si: NsdServiceInfo) {
                            @Suppress("DEPRECATION")
                            val host = si.host?.hostAddress ?: return
                            finish("$host:${si.port}")
                        }
                    })
                }
            }
            listener = l
            runCatching { nsd.discoverServices("_miravoice._tcp", NsdManager.PROTOCOL_DNS_SD, l) }.onFailure { finish(null) }
            cont.invokeOnCancellation { listener?.let { x -> runCatching { nsd.stopServiceDiscovery(x) } } }
        }
    }
}
