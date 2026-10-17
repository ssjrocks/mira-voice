package dev.local.miravoice

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import android.provider.Settings
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import dev.local.miravoice.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.math.min

/** Survives activity re-creation so the models are loaded once per process. */
object Engine {
    /** All model work runs on this one thread: the pipeline is not thread-safe. */
    val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "mira-worker") }.asCoroutineDispatcher()
    var pipeline: MiraPipeline? = null
    var configKey: String? = null
    var updateChecked = false
}

class MainActivity : AppCompatActivity() {
    private lateinit var b: ActivityMainBinding
    private val prefs by lazy { getSharedPreferences("mira", MODE_PRIVATE) }
    private val library by lazy { VoiceLibrary(this) }
    private var voices: List<SavedVoice> = emptyList()
    private var current: SavedVoice? = null
    private var lastAudio: FloatArray? = null
    private var lastRate = 48000
    private val recording = AtomicBoolean(false)
    private val cancel = AtomicBoolean(false)
    private var busy = false
    private var pendingIntent: Intent? = null
    private var showingVoice = false  // suppresses transcript saves while the field is filled programmatically
    private val player = ClipPlayer { updatePlayButtons() }

    private val pickAudio = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) loadVoiceFromUri(uri, displayName(uri), transcript = null, askName = true)
    }
    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startRecording() else toast("Microphone permission is needed to record a voice")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        SessionLog.i("ui", "main screen created")

        b.threadsSlider.value = prefs.getInt("threads", autoThreads()).toFloat().coerceIn(1f, 8f)
        b.hdSwitch.isChecked = prefs.getBoolean("hd", true)
        b.randomSeedSwitch.isChecked = prefs.getBoolean("random_seed", true)
        b.accentSwitch.isChecked = prefs.getBoolean("accent", true)
        b.versionLabel.text = "Version ${BuildConfig.VERSION_NAME}"
        b.logsLabel.text = "Logs: last ${SessionLog.KEEP} sessions in Android/data/$packageName/files/logs"
        updateThreadsLabel()
        b.sysInfo.text = "CPU cores: ${Runtime.getRuntime().availableProcessors()} (auto threads: ${autoThreads()})\n" +
            MiraNative.systemInfo().trim()

        library.migrateLegacy(prefs, File(filesDir, "voice_ref.wav"))?.let {
            prefs.edit().putString("current_voice", it).apply()
            SessionLog.i("voice", "moved the saved voice of version 0.2 into the library as $it")
        }
        refreshVoices(prefs.getString("current_voice", null))

        b.recordButton.setOnClickListener {
            if (recording.get()) recording.set(false)
            else if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording()
            else askMic.launch(Manifest.permission.RECORD_AUDIO)
        }
        b.pickButton.setOnClickListener { pickAudio.launch("audio/*") }
        b.playVoiceButton.setOnClickListener {
            val v = current ?: return@setOnClickListener
            if (player.playing == "voice") player.stop()
            else library.refWav(v.id).takeIf { it.isFile }?.let { player.play("voice", AudioIO.readWav16(it), 16000) }
        }
        b.renameVoiceButton.setOnClickListener { renameVoice() }
        b.deleteVoiceButton.setOnClickListener { deleteVoice() }
        b.voiceMenu.setOnItemClickListener { _, _, pos, _ -> selectVoice(voices.getOrNull(pos)) }
        b.replayButton.setOnClickListener {
            if (player.playing == "output") player.stop() else lastAudio?.let { player.play("output", it, lastRate) }
        }
        b.speakButton.setOnClickListener { speak(b.textInput.text?.toString().orEmpty(), b.hdSwitch.isChecked) }
        b.stopButton.setOnClickListener {
            SessionLog.i("ui", "stop pressed")
            cancel.set(true)
            player.stop()
        }
        b.saveButton.setOnClickListener { saveOutput() }
        b.hdSwitch.setOnCheckedChangeListener { _, v -> prefs.edit().putBoolean("hd", v).apply(); SessionLog.i("settings", "48 kHz upsampling $v") }
        b.randomSeedSwitch.setOnCheckedChangeListener { _, v -> prefs.edit().putBoolean("random_seed", v).apply() }
        b.accentSwitch.setOnCheckedChangeListener { _, v ->
            prefs.edit().putBoolean("accent", v).apply()
            SessionLog.i("settings", "accent matching $v")
            showVoiceStatus()
        }
        b.transcriptInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (showingVoice) return
                val v = current ?: return
                v.transcript = s?.toString()?.trim().orEmpty()
                library.save(v)
                showVoiceStatus()
            }
        })
        b.threadsSlider.addOnChangeListener { _, _, _ -> updateThreadsLabel() }
        b.threadsSlider.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) {}
            override fun onStopTrackingTouch(slider: Slider) {
                prefs.edit().putInt("threads", slider.value.toInt()).apply()
                SessionLog.i("settings", "threads ${slider.value.toInt()}")
                loadPipeline()
            }
        })
        b.ggufMenu.setOnItemClickListener { parent, _, pos, _ ->
            prefs.edit().putString("gguf", parent.getItemAtPosition(pos) as String).apply()
            loadPipeline()
        }
        b.downloadButton.setOnClickListener { downloadModels() }
        b.updateButton.setOnClickListener { checkForUpdates(silent = false) }

        pendingIntent = intent
        applyIntentSettings(intent)
        refreshModelMenu()
        loadPipeline()
        if (!Engine.updateChecked) {
            Engine.updateChecked = true
            checkForUpdates(silent = true)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        pendingIntent = intent
        applyIntentSettings(intent)
        if (!busy) loadPipeline()
    }

    override fun onResume() {
        super.onResume()
        if (!Engine.updateChecked) {
            Engine.updateChecked = true
            checkForUpdates(silent = true)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        recording.set(false)
        player.stop()
        if (isFinishing) cancel.set(true)
    }

    // ------------------------------------------------------------------ models

    private fun threads() = b.threadsSlider.value.toInt()

    /** The chosen language model, falling back to mira.gguf / the first one present. */
    private fun gguf(): String? {
        val all = ModelFiles.ggufs(this)
        return prefs.getString("gguf", null)?.takeIf { it in all } ?: all.firstOrNull { it == ModelFiles.GGUF } ?: all.firstOrNull()
    }

    private fun configKey() = "${gguf()}/${threads()}"

    private fun refreshModelMenu() {
        val all = ModelFiles.ggufs(this)
        b.ggufMenu.setSimpleItems(all.toTypedArray())
        b.ggufMenu.setText(gguf() ?: "", false)
        b.ggufLayout.visibility = if (all.size > 1) View.VISIBLE else View.GONE
    }

    /** Shows the download button if model files are missing or outdated. Returns true if the app can run. */
    private fun checkModels(): Boolean {
        val outdated = ModelStore.outdated(this)
        val missingRequired = ModelFiles.missingRequired(this)
        if (outdated.isNotEmpty()) {
            val mb = outdated.sumOf { it.size } / 1e6
            b.downloadBox.visibility = View.VISIBLE
            b.downloadProgress.visibility = View.GONE
            b.downloadButton.isEnabled = !busy
            b.downloadButton.text = "Download models (%.0f MB)".format(mb)
            b.modelStatus.text = if (missingRequired.isNotEmpty())
                "The voice models aren't on this phone yet (%.1f GB, downloaded once).".format(mb / 1000)
            else "Updated models are available (%.0f MB): %s".format(mb, outdated.joinToString { it.name })
            SessionLog.i("models", "missing or outdated: ${outdated.joinToString { it.name }}")
        } else {
            b.downloadBox.visibility = View.GONE
        }
        if (missingRequired.isNotEmpty() && ModelStore.manifest(this).isEmpty()) {
            b.modelStatus.text = "Model files missing: ${missingRequired.joinToString()}.\nCopy them to ${ModelFiles.dir(this)}"
        }
        return missingRequired.isEmpty()
    }

    private fun downloadModels() {
        val todo = ModelStore.outdated(this)
        if (todo.isEmpty() || busy) return
        setBusy(true)
        b.downloadButton.isEnabled = false
        b.downloadProgress.visibility = View.VISIBLE
        cancel.set(false)
        SessionLog.i("models", "download started: ${todo.joinToString { "${it.name} ${it.size}" }}")
        lifecycleScope.launch {
            val t0 = SystemClock.elapsedRealtime()
            try {
                withContext(Engine.worker) {
                    Engine.pipeline?.close()  // files may be replaced
                    Engine.pipeline = null
                    Engine.configKey = null
                }
                withContext(Dispatchers.IO) {
                    ModelStore.download(this@MainActivity, todo, { cancel.get() }) { name, done, total ->
                        runOnUiThread {
                            b.downloadProgress.progress = (done * 1000 / total).toInt()
                            b.modelStatus.text = "Downloading %s… %.0f of %.0f MB".format(name, done / 1e6, total / 1e6)
                        }
                    }
                }
                b.modelStatus.text = "Models downloaded"
                SessionLog.i("models", "download finished in %.0f s".format((SystemClock.elapsedRealtime() - t0) / 1000.0))
            } catch (e: Throwable) {
                SessionLog.e("models", "download failed", e)
                b.modelStatus.text = "Download failed: ${e.message}. Tap Download to resume."
            }
            setBusy(false)
            refreshModelMenu()
            loadPipeline()
        }
    }

    /** adb overrides for benchmarking: --es gguf NAME, --ei threads N */
    private fun applyIntentSettings(i: Intent?) {
        i ?: return
        val e = prefs.edit()
        i.getStringExtra("gguf")?.let { e.putString("gguf", it) }
        if (i.hasExtra("threads")) {
            e.putInt("threads", i.getIntExtra("threads", 4))
            b.threadsSlider.value = i.getIntExtra("threads", 4).toFloat().coerceIn(1f, 8f)
        }
        e.apply()
    }

    private fun loadPipeline() {
        if (busy) return  // a load, download or synthesis is running; settings controls are disabled meanwhile
        if (!checkModels()) {
            updateButtons()
            return
        }
        val key = configKey()
        if (Engine.pipeline != null && Engine.configKey == key) {
            if (b.downloadBox.visibility != View.VISIBLE) b.modelStatus.text = "Ready · ${gguf()} · ${threads()} threads"
            updateButtons()
            handleAutomation()
            return
        }
        setBusy(true)
        b.modelStatus.text = getString(R.string.loading)
        SessionLog.i("models", "loading pipeline: $key · ${SessionLog.vitals()}")
        lifecycleScope.launch {
            try {
                val p = withContext(Engine.worker) {
                    Engine.pipeline?.close()
                    Engine.pipeline = null
                    MiraPipeline(ModelFiles.dir(this@MainActivity), gguf()!!, threads(), threads())
                }
                Engine.pipeline = p
                Engine.configKey = key
                b.modelStatus.text = "Ready · %s · loaded in %.1f s · %d threads".format(gguf(), p.loadMs / 1000.0, threads())
                SessionLog.i("models", "pipeline loaded in ${p.loadMs} ms · ${SessionLog.vitals()}")
            } catch (e: Throwable) {
                SessionLog.e("models", "pipeline load failed", e)
                b.modelStatus.text = e.message ?: e.toString()
            }
            refreshModelMenu()
            setBusy(false)
            checkModels()
            handleAutomation()
        }
    }

    // ------------------------------------------------------------------ voices

    /** Reloads the library list and selects [selectId] (or the newest voice). */
    private fun refreshVoices(selectId: String?) {
        voices = library.list()
        b.voiceMenu.setSimpleItems(voices.map { it.name }.toTypedArray())
        selectVoice(voices.firstOrNull { it.id == selectId } ?: voices.lastOrNull())
    }

    private fun selectVoice(v: SavedVoice?) {
        if (v?.id != current?.id && player.playing == "voice") player.stop()
        current = v
        prefs.edit().putString("current_voice", v?.id).apply()
        showingVoice = true
        b.voiceMenu.setText(v?.name ?: "", false)
        b.transcriptInput.setText(v?.transcript ?: "")
        showingVoice = false
        b.voiceLayout.visibility = if (voices.isEmpty()) View.GONE else View.VISIBLE
        b.transcriptLayout.visibility = if (v == null) View.GONE else View.VISIBLE
        b.accentSwitch.visibility = if (v == null) View.GONE else View.VISIBLE
        if (v != null) SessionLog.i("voice", "selected '${v.name}' (${v.id}, %.1f s, ${v.promptCodes?.size ?: 0} prompt codes)".format(v.seconds))
        showVoiceStatus()
        updateButtons()
    }

    private fun showVoiceStatus() {
        val v = current
        if (v == null) {
            b.voiceStatus.text = getString(R.string.no_voice)
            return
        }
        val accent = when {
            !b.accentSwitch.isChecked -> "accent matching off"
            v.promptCodes == null -> "add this voice again to match its accent"
            v.transcript.isBlank() -> "type what's said in the clip to match its accent"
            else -> "accent matching on"
        }
        b.voiceStatus.text = "%s · %.1f s clip · %s".format(v.source.ifBlank { "voice" }, v.seconds, accent)
    }

    private fun renameVoice() {
        val v = current ?: return
        askName("Rename voice", v.name) { name ->
            SessionLog.i("voice", "renamed '${v.name}' to '$name'")
            v.name = name
            library.save(v)
            refreshVoices(v.id)
        }
    }

    private fun deleteVoice() {
        val v = current ?: return
        MaterialAlertDialogBuilder(this)
            .setTitle("Delete '${v.name}'?")
            .setMessage("Its recording is deleted too.")
            .setPositiveButton("Delete") { _, _ ->
                player.stop()
                library.delete(v.id)
                SessionLog.i("voice", "deleted '${v.name}' (${v.id})")
                current = null
                refreshVoices(null)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /** Text-entry dialog; [onOk] gets the trimmed, non-empty name. */
    private fun askName(title: String, initial: String, onCancel: (() -> Unit)? = null, onOk: (String) -> Unit) {
        val input = TextInputEditText(this).apply {
            setText(initial)
            setSelection(initial.length)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
        }
        val box = TextInputLayout(this).apply {
            setPadding(48, 16, 48, 0)
            addView(input)
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(title)
            .setView(box)
            .setPositiveButton("Save") { _, _ -> onOk(input.text?.toString()?.trim()?.ifBlank { initial } ?: initial) }
            .setNegativeButton("Cancel") { _, _ -> onCancel?.invoke() }
            .setOnCancelListener { onCancel?.invoke() }
            .show()
    }

    private suspend fun askNameSuspend(title: String, initial: String): String? = suspendCancellableCoroutine { c ->
        askName(title, initial, onCancel = { if (c.isActive) c.resume(null) }) { if (c.isActive) c.resume(it) }
    }

    private fun startRecording() {
        player.stop()
        recording.set(true)
        b.recordButton.text = getString(R.string.stop_recording)
        b.recordLevel.visibility = View.VISIBLE
        b.script.visibility = View.VISIBLE
        b.script.text = getString(R.string.script_intro) + "\n\n“" + getString(R.string.script_text) + "”"
        SessionLog.i("voice", "recording started")
        lifecycleScope.launch {
            val wav = try {
                withContext(Dispatchers.IO) {
                    AudioIO.record(20f, { recording.get() }) { peak, secs ->
                        runOnUiThread {
                            b.recordLevel.progress = (peak * 100).toInt()
                            b.voiceStatus.text = "Recording… %.1f s".format(secs)
                        }
                    }
                }
            } catch (e: Throwable) {
                SessionLog.e("voice", "recording failed", e)
                FloatArray(0)
            }
            recording.set(false)
            b.recordButton.text = getString(R.string.record)
            b.recordLevel.visibility = View.GONE
            b.script.visibility = View.GONE
            SessionLog.i("voice", "recording stopped: %.1f s".format(wav.size / 16000f))
            addVoice(wav, "Recorded", getString(R.string.script_text), library.nextName(), askName = true)
        }
    }

    private fun loadVoiceFromUri(uri: Uri, name: String, transcript: String?, askName: Boolean, then: (() -> Unit)? = null) {
        lifecycleScope.launch {
            try {
                b.voiceStatus.text = "Reading audio…"
                val t0 = SystemClock.elapsedRealtime()
                val wav = withContext(Dispatchers.IO) { AudioIO.decodeToMono(this@MainActivity, uri, 16000, 30f) }
                SessionLog.i("voice", "decoded '$name': %.1f s in %d ms".format(wav.size / 16000f, SystemClock.elapsedRealtime() - t0))
                if (addVoice(wav, "File", transcript, name, askName)) then?.invoke()
            } catch (e: Throwable) {
                SessionLog.e("voice", "couldn't read '$name'", e)
                b.voiceStatus.text = "Couldn't read that file: ${e.message}"
            }
        }
    }

    /** Analyses a clip (voice tokens from the whole clip, speech codes from its first 12 s) and adds it to the library. */
    private suspend fun addVoice(raw: FloatArray, source: String, transcript: String?, defaultName: String, askName: Boolean): Boolean {
        val wav = AudioIO.trimSilence(raw)
        if (wav.size < 16000 * 1.5) {
            b.voiceStatus.text = "That clip is too short (%.1f s of sound). Aim for 6–12 seconds.".format(wav.size / 16000f)
            SessionLog.w("voice", "clip too short: %.1f s after trimming".format(wav.size / 16000f))
            return false
        }
        val p = Engine.pipeline ?: run { b.voiceStatus.text = "Models aren't loaded yet"; return false }
        setBusy(true)
        b.voiceStatus.text = "Analysing voice…"
        val haveSemantic = File(ModelFiles.dir(this), ModelFiles.SEMANTIC).isFile
        val t0 = SystemClock.elapsedRealtime()
        val (tokens, codes) = try {
            withContext(Engine.worker) {
                val t = p.encodeVoiceTokens(wav)
                val c = if (haveSemantic) p.encodePromptCodes(wav.copyOf(min(wav.size, MiraPipeline.PROMPT_MAX_SAMPLES))) else null
                t to c
            }
        } catch (e: Throwable) {
            SessionLog.e("voice", "voice analysis failed", e)
            b.voiceStatus.text = "Couldn't analyse the voice: ${e.message}"
            setBusy(false)
            return false
        }
        SessionLog.i("voice", "analysed %.1f s clip in %d ms: %d prompt codes%s · %s".format(wav.size / 16000f,
            SystemClock.elapsedRealtime() - t0, codes?.size ?: 0, if (haveSemantic) "" else " (no semantic encoder)", SessionLog.vitals()))
        setBusy(false)
        val name = if (askName) askNameSuspend("Name this voice", defaultName) else defaultName
        if (name == null) {
            SessionLog.i("voice", "new voice discarded")
            showVoiceStatus()
            return false
        }
        val v = library.add(name, source, wav, tokens, codes, transcript ?: "")
        SessionLog.i("voice", "added '${v.name}' (${v.id})")
        refreshVoices(v.id)
        return true
    }

    private fun displayName(uri: Uri): String = runCatching {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0)?.substringBeforeLast('.') else null
        }
    }.getOrNull() ?: library.nextName()

    /** The voice used for speaking: with its clip prompt only when accent matching is on and usable. */
    private fun speakingVoice(): Voice? {
        val v = current?.toVoice() ?: return null
        return if (b.accentSwitch.isChecked && v.accentMode) v else Voice(v.tokens, null, null)
    }

    // ------------------------------------------------------------------ speech

    private fun speak(text: String, hd: Boolean, play: Boolean = true, outName: String? = null, seedOverride: Int? = null) {
        val p = Engine.pipeline ?: return
        val v = speakingVoice() ?: run { toast("Add a voice first"); return }
        if (text.isBlank()) return
        player.stop()
        cancel.set(false)
        setBusy(true)
        b.stopButton.isEnabled = true
        val rate = if (hd) 48000 else 16000
        val seed = seedOverride ?: if (b.randomSeedSwitch.isChecked) (System.nanoTime() and 0x7fffffff).toInt() else 0
        val stats = mutableListOf<ChunkStats>()
        SessionLog.i("speak", "start: ${text.length} chars, voice '${current?.name}', accent ${v.accentMode}, hd $hd, " +
            "seed $seed, ${p.llmThreads} threads · ${SessionLog.vitals()}")
        lifecycleScope.launch {
            val t0 = SystemClock.elapsedRealtime()
            var audio: FloatArray? = null
            try {
                audio = withContext(Engine.worker) {
                    p.synthesize(text, v, hd, GenParams(seed = seed), { cancel.get() },
                        onStatus = { s -> runOnUiThread { b.genStatus.text = s } }, stats = stats)
                }
            } catch (e: Throwable) {
                SessionLog.e("speak", "synthesis failed", e)
                b.genStatus.text = "Failed: ${e.message}"
            }
            val totalMs = SystemClock.elapsedRealtime() - t0
            for ((i, s) in stats.withIndex()) {
                SessionLog.i("speak", "part ${i + 1}: ${s.text.length} chars, prompt ${s.promptTokens} tok in %.0f ms, %d speech tok in %.0f ms (%.1f tok/s), decode %.0f ms, upsample %.0f ms"
                    .format(s.prefillMs, s.speechTokens, s.llmMs, s.speechTokens / (s.llmMs / 1000), s.decodeMs, s.upsampleMs))
            }
            setBusy(false)
            if (cancel.get()) {
                b.genStatus.text = "Stopped"
                SessionLog.i("speak", "cancelled after %.1f s".format(totalMs / 1000.0))
            } else if (audio != null && audio.isNotEmpty()) {
                lastAudio = audio
                lastRate = rate
                showStats(stats, totalMs, hd, seed, v.accentMode)
                if (outName != null) {
                    val f = File(getExternalFilesDir("out"), outName)
                    AudioIO.writeWav(f, audio, rate)
                    SessionLog.i("speak", "wrote ${f.path}")
                }
                if (play) player.play("output", audio, rate)
                updatePlayButtons()
            }
            SessionLog.i("speak", "end · ${SessionLog.vitals()}")
            loadPipeline()  // applies settings changed meanwhile (e.g. over adb), then runs any queued request
        }
    }

    private fun showStats(stats: List<ChunkStats>, totalMs: Long, hd: Boolean, seed: Int, accent: Boolean) {
        if (stats.isEmpty()) return
        val audio = stats.sumOf { it.audioSec }
        val tokens = stats.sumOf { it.speechTokens }
        val llm = stats.sumOf { it.llmMs } / 1000
        val prompt = stats.sumOf { it.promptTokens }
        val prefill = stats.sumOf { it.prefillMs } / 1000
        val dec = stats.sumOf { it.decodeMs } / 1000
        val up = stats.sumOf { it.upsampleMs } / 1000
        val tps = tokens / llm
        b.genStatus.text = "Done: %.1f s of speech in %.1f s".format(audio, totalMs / 1000.0)
        b.stats.text = buildString {
            append("Parts      %d   audio %.1f s%n".format(stats.size, audio))
            append("Prompt     %d tokens in %.2f s%s%n".format(prompt, prefill, if (accent) " (accent)" else ""))
            append("LLM        %d tokens in %.1f s = %.1f tok/s%n".format(tokens, llm, tps))
            append("Decoder    %.2f s  (%.2fx real time)%n".format(dec, dec / audio))
            if (hd) append("Upsampler  %.2f s  (%.2fx real time)%n".format(up, up / audio))
            append("Total      %.1f s  · RTF %.2f".format(totalMs / 1000.0, totalMs / 1000.0 / audio))
        }
        val p = Engine.pipeline
        val json = JSONObject().apply {
            put("parts", stats.size); put("audio_s", audio); put("tokens", tokens); put("llm_s", llm)
            put("tok_per_s", tps); put("prompt_tokens", prompt); put("prefill_s", prefill); put("decoder_s", dec)
            put("upsampler_s", up); put("total_s", totalMs / 1000.0); put("hd", hd); put("accent", accent)
            put("seed", seed); put("threads", p?.llmThreads); put("gguf", p?.gguf)
        }
        SessionLog.i("speak", "MIRA_RESULT $json")
    }

    private fun saveOutput() {
        val a = lastAudio ?: return
        try {
            val name = "mira_" + java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date()) + ".wav"
            AudioIO.saveToMusic(this, name, a, lastRate)
            SessionLog.i("output", "saved Music/MiraVoice/$name")
            toast("Saved Music/MiraVoice/$name")
        } catch (e: Throwable) {
            SessionLog.e("output", "save failed", e)
            toast("Save failed: ${e.message}")
        }
    }

    // ------------------------------------------------------------------ updates

    private fun checkForUpdates(silent: Boolean) {
        lifecycleScope.launch {
            val rel = try {
                withContext(Dispatchers.IO) { Updater.latest() }
            } catch (e: Throwable) {
                SessionLog.w("update", "update check failed: ${e.message}")
                if (!silent) toast("Couldn't check for updates: ${e.message}")
                return@launch
            }
            SessionLog.i("update", "latest release ${rel?.version ?: "none"}, installed ${BuildConfig.VERSION_NAME}")
            if (rel == null || !Updater.isNewer(rel.version, BuildConfig.VERSION_NAME)) {
                if (!silent) toast("You have the latest version (${BuildConfig.VERSION_NAME})")
                return@launch
            }
            MaterialAlertDialogBuilder(this@MainActivity)
                .setTitle("Mira Voice ${rel.version} is available")
                .setMessage("You have ${BuildConfig.VERSION_NAME}.\n\n" + rel.notes.take(1500))
                .setPositiveButton("Install") { _, _ -> installUpdate(rel) }
                .setNegativeButton("Later") { _, _ -> SessionLog.i("update", "update ${rel.version} postponed") }
                .show()
        }
    }

    private fun installUpdate(rel: Updater.Release) {
        if (!packageManager.canRequestPackageInstalls()) {
            SessionLog.i("update", "asking for the install-apps permission")
            MaterialAlertDialogBuilder(this)
                .setTitle("Allow updates")
                .setMessage("To install updates, allow Mira Voice to install apps on the next screen, then come back.")
                .setPositiveButton("Open settings") { _, _ ->
                    Engine.updateChecked = false  // offer the update again on return
                    startActivity(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:$packageName")))
                }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }
        b.modelStatus.text = "Downloading update ${rel.version}…"
        SessionLog.i("update", "downloading ${rel.apkUrl} (${rel.apkSize} bytes)")
        lifecycleScope.launch {
            try {
                val apk = withContext(Dispatchers.IO) {
                    Updater.downloadApk(this@MainActivity, rel) { done, total ->
                        if (total > 0) runOnUiThread {
                            b.modelStatus.text = "Downloading update ${rel.version}… %.0f%%".format(done * 100.0 / total)
                        }
                    }
                }
                withContext(Dispatchers.IO) { Updater.install(this@MainActivity, apk) }
                b.modelStatus.text = "Installing update ${rel.version}…"
                SessionLog.i("update", "handed ${apk.name} to the installer")
            } catch (e: Throwable) {
                SessionLog.e("update", "update failed", e)
                b.modelStatus.text = "Update failed: ${e.message}"
            }
        }
    }

    // ------------------------------------------------------------------ adb automation
    // adb shell am start -n dev.local.miravoice/.MainActivity --es text "Hi" [--es voice_path /path.wav
    //     --es voice_transcript "words in the clip" --es voice_name NAME] [--ez hd true] [--es out result.wav]
    //     [--ei seed 0] [--ez play false]   -> session log / logcat "MIRA_RESULT {...}"

    private fun handleAutomation() {
        val i = pendingIntent ?: return
        if (busy || Engine.pipeline == null) return
        pendingIntent = null
        val text = i.getStringExtra("text") ?: return
        val hd = i.getBooleanExtra("hd", true)
        val out = i.getStringExtra("out")
        val seed = if (i.hasExtra("seed")) i.getIntExtra("seed", 0) else null
        SessionLog.i("automation", "request: hd=$hd out=$out seed=$seed config=${Engine.configKey}")
        val go = { speak(text, hd, play = i.getBooleanExtra("play", true), outName = out, seedOverride = seed) }
        val path = i.getStringExtra("voice_path")
        if (path != null) {
            val name = i.getStringExtra("voice_name") ?: File(path).nameWithoutExtension
            loadVoiceFromUri(Uri.fromFile(File(path)), name, i.getStringExtra("voice_transcript"), askName = false, then = go)
        } else go()
    }

    // ------------------------------------------------------------------ ui helpers

    private fun setBusy(v: Boolean) {
        busy = v
        updateButtons()
        b.progress.visibility = if (v) View.VISIBLE else View.INVISIBLE
        if (!v) b.stopButton.isEnabled = false
    }

    private fun updateButtons() {
        val ready = Engine.pipeline != null && !busy
        b.speakButton.isEnabled = ready && current != null
        b.recordButton.isEnabled = ready || recording.get()
        b.pickButton.isEnabled = ready
        b.renameVoiceButton.isEnabled = current != null && !busy
        b.deleteVoiceButton.isEnabled = current != null && !busy
        b.voiceLayout.isEnabled = !busy
        b.threadsSlider.isEnabled = !busy
        b.ggufLayout.isEnabled = !busy
        b.downloadButton.isEnabled = !busy
        updatePlayButtons()
    }

    private fun updatePlayButtons() {
        b.playVoiceButton.isEnabled = current?.let { library.refWav(it.id).isFile } == true
        b.playVoiceButton.text = getString(if (player.playing == "voice") R.string.stop else R.string.play)
        b.replayButton.text = getString(if (player.playing == "output") R.string.stop else R.string.play)
        b.replayButton.isEnabled = lastAudio != null
        b.saveButton.isEnabled = lastAudio != null
    }

    private fun updateThreadsLabel() {
        b.threadsLabel.text = "CPU threads: ${threads()}"
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()

    /** Counts the fast cores (max clock >= 80% of the fastest), clamped to 2..4: on a Snapdragon 8 Gen 2
     *  llama.cpp ran 63 tok/s with 4 threads but 42 with 5 and 27 with 6. */
    private fun autoThreads(): Int {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { c ->
            runCatching { File("/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrNull()
        }
        if (freqs.isEmpty()) return 4
        val top = freqs.max()
        return freqs.count { it >= top * 0.8 }.coerceIn(2, 4)
    }
}
