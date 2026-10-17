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
import com.ssjrocks.appreports.AppReports
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
    private val player = ClipPlayer { updatePlayButtons() }

    private val pickAudio = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) loadVoiceFromUri(uri, displayName(uri), transcript = null, ask = true)
    }
    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startRecording() else toast("Microphone permission is needed to record a voice")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        // dark status-bar icons (signal, battery) on the light theme, light ones on the dark theme
        val night = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) ==
            android.content.res.Configuration.UI_MODE_NIGHT_YES
        androidx.core.view.WindowCompat.getInsetsController(window, window.decorView).isAppearanceLightStatusBars = !night
        SessionLog.i("ui", "main screen created")

        b.threadsSlider.value = prefs.getInt("threads", autoThreads()).toFloat().coerceIn(1f, 8f)
        b.hdSwitch.isChecked = prefs.getBoolean("hd", true)
        b.randomSeedSwitch.isChecked = prefs.getBoolean("random_seed", true)
        b.accentSwitch.isChecked = prefs.getBoolean("accent", true)
        b.versionLabel.text = "Version ${BuildConfig.VERSION_NAME}"
        b.logsLabel.text = "Logs: last ${SessionLog.KEEP} sessions in Android/data/$packageName/files/logs"
        b.reportButton.setOnClickListener {
            SessionLog.i("ui", "report a problem opened · ${SessionLog.vitals()}")
            AppReports.showReportDialog(this, mapOf("threads" to prefs.getInt("threads", autoThreads()), "hd" to prefs.getBoolean("hd", true)))
        }
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
        b.newTextInput.setText(prefs.getString("new_text", null) ?: getString(R.string.script_text))
        b.newTextInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                prefs.edit().putString("new_text", s?.toString().orEmpty()).apply()
            }
        })
        b.editTranscriptButton.setOnClickListener { editTranscript() }
        b.clipSlider.value = prefs.getInt("clip_s", 12).toFloat().coerceIn(5f, 20f)
        updateClipLabel()
        b.clipSlider.addOnChangeListener { _, _, _ -> updateClipLabel() }
        b.clipSlider.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) {}
            override fun onStopTrackingTouch(slider: Slider) {
                prefs.edit().putInt("clip_s", slider.value.toInt()).apply()
                SessionLog.i("settings", "accent clip length ${slider.value.toInt()} s")
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
            if (pendingIntent == null) checkUncheckedVoice()
            handleAutomation()
        }
    }

    // ------------------------------------------------------------------ voices

    /** Accent clip length (s): recordings stop here, and files contribute their first this-many seconds. */
    private fun clipSeconds() = b.clipSlider.value.toInt()

    private fun updateClipLabel() {
        b.clipLabel.text = "Accent clip length: ${clipSeconds()} s"
    }

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
        b.voiceMenu.setText(v?.name ?: "", false)
        b.voiceLayout.visibility = if (voices.isEmpty()) View.GONE else View.VISIBLE
        for (view in listOf(b.transcriptView, b.editTranscriptButton, b.accentSwitch)) {
            view.visibility = if (v == null) View.GONE else View.VISIBLE
        }
        if (v != null) SessionLog.i("voice", "selected '${v.name}' (${v.id}, %.1f s, ${v.promptCodes?.size ?: 0} prompt codes)".format(v.seconds))
        showVoiceStatus()
        updateButtons()
        checkUncheckedVoice()
    }

    /** Voices made before 0.4 (or whose transcript changed) are checked once, when selected and the models are ready. */
    private fun checkUncheckedVoice() {
        val v = current ?: return
        if (busy || Engine.pipeline == null || v.checkRatio != null || !v.toVoice().accentMode) return
        lifecycleScope.launch {
            checkVoice(v)
            handleAutomation()
        }
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
            v.transcript.isBlank() -> "no transcript, so no accent matching"
            v.transcriptMismatch -> "⚠ transcript doesn't match the clip (tap Edit transcript or record again)"
            else -> "accent matching on"
        }
        b.voiceStatus.text = "%s · %.1f s clip · %s".format(v.source.ifBlank { "Voice" }, v.seconds, accent)
        b.transcriptView.text = if (v.transcript.isBlank()) "What's said in the clip: (not set)"
        else "What's said in the clip: “${v.transcript}”"
    }

    private fun renameVoice() {
        val v = current ?: return
        lifecycleScope.launch {
            val name = askText("Rename voice", v.name, multiLine = false) ?: return@launch
            if (name.isBlank()) return@launch
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

    private fun editTranscript() {
        val v = current ?: return
        lifecycleScope.launch {
            val t = askText("What's said in the clip", v.transcript, multiLine = true,
                message = "Type exactly what's said in the clip, word for word. A transcript that doesn't match makes outputs " +
                    "repeat part of the clip or come out almost silent. Clear it to turn accent matching off for this voice.")
                ?: return@launch
            v.transcript = t
            v.checkRatio = null
            library.save(v)
            SessionLog.i("voice", "transcript of '${v.name}' edited (${t.length} chars)")
            showVoiceStatus()
            checkVoice(v)
        }
    }

    /** A dialog with one text field; returns the trimmed text, or null if cancelled. */
    private suspend fun askText(title: String, initial: String, multiLine: Boolean, message: String? = null): String? =
        suspendCancellableCoroutine { c ->
            val input = TextInputEditText(this).apply {
                setText(initial)
                setSelection(initial.length)
                inputType = if (multiLine) InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                else InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            }
            val box = TextInputLayout(this).apply {
                setPadding(48, 16, 48, 0)
                addView(input)
            }
            MaterialAlertDialogBuilder(this)
                .setTitle(title)
                .apply { if (message != null) setMessage(message) }
                .setView(box)
                .setPositiveButton("Save") { _, _ -> if (c.isActive) c.resume(input.text?.toString()?.trim().orEmpty()) }
                .setNegativeButton("Cancel") { _, _ -> if (c.isActive) c.resume(null) }
                .setOnCancelListener { if (c.isActive) c.resume(null) }
                .show()
        }

    /** Name + transcript for a new voice; null if discarded. */
    private suspend fun askNewVoice(defaultName: String, transcript: String, note: String): Pair<String, String>? =
        suspendCancellableCoroutine { c ->
            fun field(text: String, multi: Boolean) = TextInputEditText(this).apply {
                setText(text)
                inputType = if (multi) InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                else InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            }
            val name = field(defaultName, false)
            val words = field(transcript, true)
            val box = android.widget.LinearLayout(this).apply {
                orientation = android.widget.LinearLayout.VERTICAL
                setPadding(48, 16, 48, 0)
                addView(TextInputLayout(this@MainActivity).apply { hint = "Name"; addView(name) })
                addView(TextInputLayout(this@MainActivity).apply { hint = "What's said in the clip (word for word)"; addView(words) })
            }
            MaterialAlertDialogBuilder(this)
                .setTitle("Save voice")
                .setMessage(note)
                .setView(box)
                .setPositiveButton("Save") { _, _ ->
                    if (c.isActive) c.resume(name.text.toString().trim().ifBlank { defaultName } to words.text.toString().trim())
                }
                .setNegativeButton("Discard") { _, _ -> if (c.isActive) c.resume(null) }
                .setOnCancelListener { if (c.isActive) c.resume(null) }
                .show()
        }

    private fun startRecording() {
        player.stop()
        val text = b.newTextInput.text?.toString()?.trim().orEmpty()
        val limit = clipSeconds()
        recording.set(true)
        b.recordButton.text = getString(R.string.stop_recording)
        b.recordLevel.visibility = View.VISIBLE
        b.script.visibility = View.VISIBLE
        b.script.text = getString(R.string.script_intro) + "\n\n“" + text.ifBlank { "(say anything, then type it in when saving)" } + "”"
        SessionLog.i("voice", "recording started (limit $limit s, text ${text.length} chars)")
        lifecycleScope.launch {
            val wav = try {
                withContext(Dispatchers.IO) {
                    AudioIO.record(limit.toFloat(), { recording.get() }) { peak, secs ->
                        runOnUiThread {
                            b.recordLevel.progress = (peak * 100).toInt()
                            b.voiceStatus.text = "Recording… %.1f of %d s".format(secs, limit)
                        }
                    }
                }
            } catch (e: Throwable) {
                SessionLog.e("voice", "recording failed", e)
                FloatArray(0)
            }
            val hitLimit = wav.size >= (limit - 0.2) * 16000
            recording.set(false)
            b.recordButton.text = getString(R.string.record)
            b.recordLevel.visibility = View.GONE
            b.script.visibility = View.GONE
            SessionLog.i("voice", "recording stopped: %.1f s%s".format(wav.size / 16000f, if (hitLimit) " (hit the limit)" else ""))
            val note = if (hitLimit) "Recording stopped at the $limit s limit (Settings → Accent clip length). If you hadn't " +
                "finished, change the transcript to exactly what you said before the cut, or record again."
            else "Check the transcript matches what you said, word for word."
            addVoice(wav, "Recorded", text, library.nextName(), ask = true, note = note)
        }
    }

    private fun loadVoiceFromUri(uri: Uri, name: String, transcript: String?, ask: Boolean, then: (() -> Unit)? = null) {
        lifecycleScope.launch {
            try {
                b.voiceStatus.text = "Reading audio…"
                val t0 = SystemClock.elapsedRealtime()
                val wav = withContext(Dispatchers.IO) { AudioIO.decodeToMono(this@MainActivity, uri, 16000, 60f) }
                SessionLog.i("voice", "decoded '$name': %.1f s in %d ms".format(wav.size / 16000f, SystemClock.elapsedRealtime() - t0))
                // a typed transcript is used for files too, unless it's still the suggested sentence
                val typed = b.newTextInput.text?.toString()?.trim().orEmpty().takeIf { it != getString(R.string.script_text) }
                val secs = AudioIO.trimSilence(wav).size / 16000f
                val note = if (secs > clipSeconds() + 0.3f) "This file has %.0f s of sound; only the first %d s are used for accent matching. Type just what's said in those seconds, word for word, or leave it empty to copy the voice without its accent."
                    .format(secs, clipSeconds())
                else "Type exactly what's said in the file, word for word, or leave it empty to copy the voice without its accent."
                if (addVoice(wav, "File", transcript ?: typed ?: "", name, ask, note)) then?.invoke()
            } catch (e: Throwable) {
                SessionLog.e("voice", "couldn't read '$name'", e)
                b.voiceStatus.text = "Couldn't read that file: ${e.message}"
            }
        }
    }

    /**
     * Analyses a clip (voice tokens from the whole clip, speech codes from its first [clipSeconds]) and adds it to the
     * library, then checks that its transcript matches the audio.
     */
    private suspend fun addVoice(raw: FloatArray, source: String, transcript: String, defaultName: String, ask: Boolean, note: String): Boolean {
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
        val promptSamples = min(wav.size, clipSeconds() * 16000)
        val t0 = SystemClock.elapsedRealtime()
        val (tokens, codes) = try {
            withContext(Engine.worker) {
                val t = p.encodeVoiceTokens(wav)
                val c = if (haveSemantic) p.encodePromptCodes(wav.copyOf(promptSamples)) else null
                t to c
            }
        } catch (e: Throwable) {
            SessionLog.e("voice", "voice analysis failed", e)
            b.voiceStatus.text = "Couldn't analyse the voice: ${e.message}"
            setBusy(false)
            return false
        }
        SessionLog.i("voice", "analysed %.1f s clip in %d ms: %d prompt codes from the first %.1f s%s · %s".format(wav.size / 16000f,
            SystemClock.elapsedRealtime() - t0, codes?.size ?: 0, promptSamples / 16000f,
            if (haveSemantic) "" else " (no semantic encoder)", SessionLog.vitals()))
        setBusy(false)
        showVoiceStatus()
        val (name, words) = if (ask) askNewVoice(defaultName, transcript, note) ?: run {
            SessionLog.i("voice", "new voice discarded")
            return false
        } else defaultName to transcript
        val v = library.add(name, source, wav, tokens, codes, words)
        SessionLog.i("voice", "added '${v.name}' (${v.id}), transcript ${words.length} chars")
        refreshVoices(v.id)
        checkVoice(v)
        return true
    }

    /** Test-speaks a short sentence to catch a transcript that doesn't match the clip (see MiraPipeline.checkVoice). */
    private suspend fun checkVoice(v: SavedVoice) {
        val voice = v.toVoice()
        val p = Engine.pipeline
        if (!voice.accentMode || p == null) return
        setBusy(true)
        b.voiceStatus.text = "Checking the transcript matches the clip…"
        val ratio = try {
            withContext(Engine.worker) { p.checkVoice(voice) }
        } catch (e: Throwable) {
            SessionLog.e("voice", "voice check failed", e)
            null
        }
        setBusy(false)
        showVoiceStatus()
        ratio ?: return
        SessionLog.i("voice", "check '${v.name}': test sentence came out %.2fx the expected length".format(ratio))
        v.checkRatio = ratio
        library.save(v)
        showVoiceStatus()
        val problem = when {
            ratio < 0.6 -> "The test sentence came out almost silent: the clip seems to contain more speech than its " +
                "transcript, so the model thinks your text has already been said."
            ratio > 2.0 -> "The test sentence came out far too long: the transcript seems to have words that aren't in the " +
                "clip, and they'd be spoken at the start of every output."
            else -> return
        }
        SessionLog.w("voice", "transcript of '${v.name}' doesn't match its clip (ratio %.2f)".format(ratio))
        MaterialAlertDialogBuilder(this)
            .setTitle("Transcript doesn't match '${v.name}'")
            .setMessage("$problem\n\nMake it match what's said in the clip word for word, or record again. Until then, parts " +
                "that come out silent are redone without accent matching.")
            .setPositiveButton("Edit transcript") { _, _ -> editTranscript() }
            .setNegativeButton("Keep as is", null)
            .show()
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
                val fellBack = stats.withIndex().filter { it.value.accentFallback }.map { it.index + 1 }
                if (fellBack.isNotEmpty()) {
                    b.genStatus.append("\nAccent matching was skipped for part ${fellBack.joinToString()}: it came out almost " +
                        "silent, which means the voice's transcript doesn't match its clip (tap Edit transcript).")
                    SessionLog.w("speak", "accent fallback in parts $fellBack (voice '${current?.name}')")
                }
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
            loadVoiceFromUri(Uri.fromFile(File(path)), name, i.getStringExtra("voice_transcript") ?: "", ask = false, then = go)
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
        b.editTranscriptButton.isEnabled = current != null && !busy
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
