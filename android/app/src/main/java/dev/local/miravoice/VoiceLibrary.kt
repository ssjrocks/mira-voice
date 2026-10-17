package dev.local.miravoice

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** A voice in the library: its clip (ref.wav) plus everything derived from it. */
class SavedVoice(
    val id: String,
    var name: String,
    val created: Long,
    val source: String,
    var seconds: Float,
    val tokens: IntArray,
    var promptCodes: IntArray?,
    var transcript: String,
    /** Last transcript check (test-sentence length / expected); null = not checked since the transcript changed. */
    var checkRatio: Double? = null,
) {
    val transcriptMismatch get() = checkRatio?.let { it < 0.6 || it > 2.0 } == true

    fun toVoice() = Voice(tokens, promptCodes, transcript)

    fun toJson() = JSONObject().apply {
        put("id", id); put("name", name); put("created", created); put("source", source); put("seconds", seconds.toDouble())
        put("tokens", JSONArray(tokens.toList()))
        put("prompt_codes", promptCodes?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
        put("transcript", transcript)
        put("check_ratio", checkRatio ?: JSONObject.NULL)
    }

    companion object {
        fun fromJson(j: JSONObject): SavedVoice {
            fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
            return SavedVoice(j.getString("id"), j.getString("name"), j.getLong("created"), j.optString("source"),
                j.optDouble("seconds", 0.0).toFloat(), ints(j.getJSONArray("tokens")),
                j.optJSONArray("prompt_codes")?.let { ints(it) }, j.optString("transcript"),
                if (j.isNull("check_ratio")) null else j.optDouble("check_ratio"))
        }
    }
}

/** Voices stored under <files>/voices/<id>/ (voice.json + ref.wav). Not thread-safe: use from the main thread. */
class VoiceLibrary(context: Context) {
    private val root = File(context.filesDir, "voices").apply { mkdirs() }

    fun list(): List<SavedVoice> = root.listFiles { f -> f.isDirectory }?.mapNotNull { d ->
        runCatching { SavedVoice.fromJson(JSONObject(File(d, "voice.json").readText())) }.getOrNull()
    }?.sortedBy { it.created } ?: emptyList()

    fun get(id: String?): SavedVoice? = id?.let { i ->
        runCatching { SavedVoice.fromJson(JSONObject(File(File(root, i), "voice.json").readText())) }.getOrNull()
    }

    fun refWav(id: String) = File(File(root, id), "ref.wav")

    /** Replaces a voice's clip (e.g. after shortening it to the longest usable length). */
    fun saveClip(id: String, wav16k: FloatArray) = AudioIO.writeWav(refWav(id), wav16k, 16000)

    fun add(name: String, source: String, wav16k: FloatArray, tokens: IntArray, codes: IntArray?, transcript: String): SavedVoice {
        val id = "v" + System.currentTimeMillis()
        val v = SavedVoice(id, name, System.currentTimeMillis(), source, wav16k.size / 16000f, tokens, codes, transcript)
        val d = File(root, id).apply { mkdirs() }
        AudioIO.writeWav(File(d, "ref.wav"), wav16k, 16000)
        save(v)
        return v
    }

    fun save(v: SavedVoice) {
        val d = File(root, v.id).apply { mkdirs() }
        val tmp = File(d, "voice.json.tmp")
        tmp.writeText(v.toJson().toString())
        tmp.renameTo(File(d, "voice.json"))
    }

    fun delete(id: String) {
        File(root, id).deleteRecursively()
    }

    /** A name not used yet: "Voice 1", "Voice 2", … */
    fun nextName(prefix: String = "Voice"): String {
        val names = list().map { it.name }.toSet()
        var n = 1
        while ("$prefix $n" in names) n++
        return "$prefix $n"
    }

    /** Moves the single voice of version 0.2 (prefs + voice_ref.wav) into the library. Returns its id or null. */
    fun migrateLegacy(prefs: SharedPreferences, legacyRef: File): String? {
        val tokens = prefs.getString("voice_tokens", null)?.split(",")?.mapNotNull { it.toIntOrNull() }?.toIntArray()
        if (tokens == null || tokens.size != 32) return null
        val codes = prefs.getString("voice_codes", null)?.split(",")?.mapNotNull { it.toIntOrNull() }?.toIntArray()
            ?.takeIf { it.isNotEmpty() }
        val wav = if (legacyRef.isFile) AudioIO.readWav16(legacyRef) else FloatArray(0)
        val v = add("My voice", "Recorded", wav, tokens, codes,
            prefs.getString("voice_transcript", "") ?: "")
        prefs.edit().remove("voice_tokens").remove("voice_codes").remove("voice_transcript").remove("voice_label").apply()
        legacyRef.delete()
        return v.id
    }
}
