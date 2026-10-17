package dev.local.miravoice

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** Model files: where they live on the phone and which ones this app version needs. */
object ModelFiles {
    const val GGUF = "mira.gguf"
    const val HEAD = "mira_head.bin"
    const val SPEAKER = "speaker_encoder.onnx"
    const val SEMANTIC = "semantic_encoder.onnx"
    const val PROCESSER = "processer.onnx"
    const val VOCODER = "vocoder.onnx"
    const val UPSAMPLER = "upsampler.onnx"
    /** Speech recognition for voice setup (whisper small.en); optional: without it transcripts are typed. */
    const val ASR = "whisper-small.en-q5_1.bin"

    /** Needed to speak at all; the semantic encoder is only needed to set up accent matching. */
    val REQUIRED = listOf(GGUF, HEAD, SPEAKER, PROCESSER, VOCODER, UPSAMPLER)

    fun dir(context: Context): File = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }

    /** Language-model files present (normally just mira.gguf; extra ones can be pushed for experiments). */
    fun ggufs(context: Context): List<String> =
        dir(context).listFiles { f -> f.isFile && f.name.endsWith(".gguf") }?.map { it.name }?.sorted() ?: emptyList()

    fun missingRequired(context: Context): List<String> = REQUIRED.filter { !File(dir(context), it).isFile }
}

/** One downloadable model file, from the manifest bundled in the APK (assets/models.json). */
data class ModelEntry(val name: String, val url: String, val size: Long, val sha256: String)

object ModelStore {
    fun manifest(context: Context): List<ModelEntry> {
        val json = runCatching { context.assets.open("models.json").bufferedReader().readText() }.getOrNull() ?: return emptyList()
        val files = JSONObject(json).getJSONArray("files")
        return (0 until files.length()).map {
            val f = files.getJSONObject(it)
            ModelEntry(f.getString("name"), f.getString("url"), f.getLong("size"), f.getString("sha256"))
        }
    }

    /** Manifest entries whose file is absent or a different size than this version expects. */
    fun outdated(context: Context): List<ModelEntry> {
        val dir = ModelFiles.dir(context)
        return manifest(context).filter { e -> File(dir, e.name).let { !it.isFile || it.length() != e.size } }
    }

    /**
     * Downloads [entries] into the models folder (resuming partial downloads), checking size and SHA-256.
     * onProgress(fileName, bytesDoneOverall, bytesTotalOverall).
     */
    fun download(context: Context, entries: List<ModelEntry>, isCancelled: () -> Boolean,
                 onProgress: (String, Long, Long) -> Unit) {
        val dir = ModelFiles.dir(context)
        val total = entries.sumOf { it.size }
        var before = 0L
        for (e in entries) {
            val part = File(dir, e.name + ".part")
            var have = if (part.isFile) part.length() else 0L
            SessionLog.i("models", "downloading ${e.name}: ${e.size} bytes, resuming at $have")
            if (have > e.size) { part.delete(); have = 0 }
            if (have < e.size) {
                val conn = (URL(e.url).openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = true
                    connectTimeout = 20_000
                    readTimeout = 60_000
                    if (have > 0) setRequestProperty("Range", "bytes=$have-")
                }
                val code = conn.responseCode
                if (code == 200 && have > 0) have = 0  // server ignored the range: start over
                else if (code != 200 && code != 206) throw IOException("download of ${e.name} failed: HTTP $code")
                RandomAccessFile(part, "rw").use { raf ->
                    raf.setLength(have)
                    raf.seek(have)
                    conn.inputStream.use { inp ->
                        val buf = ByteArray(1 shl 16)
                        var lastReport = 0L
                        while (true) {
                            if (isCancelled()) throw InterruptedException("cancelled")
                            val n = inp.read(buf)
                            if (n < 0) break
                            raf.write(buf, 0, n)
                            have += n
                            if (have - lastReport > (4 shl 20)) {
                                lastReport = have
                                onProgress(e.name, before + have, total)
                            }
                        }
                    }
                }
                conn.disconnect()
            }
            onProgress("${e.name} (checking)", before + have, total)
            if (part.length() != e.size || sha256(part) != e.sha256) {
                SessionLog.e("models", "${e.name}: size ${part.length()} / checksum mismatch, deleting the partial file")
                part.delete()
                throw IOException("${e.name} is corrupt (size or checksum mismatch); try again")
            }
            SessionLog.i("models", "${e.name} verified")
            val dst = File(dir, e.name)
            dst.delete()
            if (!part.renameTo(dst)) throw IOException("could not write $dst")
            before += e.size
        }
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { inp ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = inp.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
