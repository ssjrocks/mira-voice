package dev.local.miravoice

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import android.provider.Settings
import android.text.InputType
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import android.widget.LinearLayout
import android.widget.TextView
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.ssjrocks.appreports.AppReports
import dev.local.miravoice.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume

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
    private val pc by lazy { PcServer(this) }
    /** "On my PC" mode: speech is made by the PC server over the LAN (see PcServer). */
    private var usePc = false
    private var pcConnecting = false
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
        if (uri != null) loadVoiceFromUri(uri, displayName(uri), preset = null, ask = true)
    }
    private var micMode: AddMode = AddMode.SPEECH
    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startRecording(micMode) else toast("Microphone permission is needed to record a voice")
    }

    /** Record button handler: stops a running recording, or starts one in [mode] (asking for the mic first). */
    private fun onRecordButton(mode: AddMode) {
        if (recording.get()) {
            recording.set(false)
            return
        }
        micMode = mode
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording(mode)
        else askMic.launch(Manifest.permission.RECORD_AUDIO)
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

        b.exampleButton.setOnClickListener { onRecordButton(AddMode.EXAMPLE) }
        b.recordButton.setOnClickListener { onRecordButton(AddMode.SPEECH) }
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
        b.editTranscriptButton.setOnClickListener { editTranscript() }
        usePc = prefs.getBoolean("use_pc", false)
        b.whereGroup.check(if (usePc) R.id.pcButton else R.id.phoneButton)
        b.whereGroup.addOnButtonCheckedListener { _, id, checked -> if (checked) setWhere(id == R.id.pcButton) }
        b.pcSetupButton.setOnClickListener {
            SessionLog.i("pc", "change PC pressed")
            pc.forget()
            connectPc()
        }
        b.pcEngineMenu.setOnItemClickListener { parent, _, pos, _ ->
            pc.engine = pcEngines.getOrNull(pos)?.first ?: return@setOnItemClickListener
            SessionLog.i("pc", "PC voice model ${pc.engine}")
            warmPc()
        }
        showWhere()
        if (usePc) connectPc(interactive = false)
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
        // PC-mode test hooks, debug builds only: the activity is exported, and other apps mustn't be able to send
        // voices to an address of their choosing
        if (BuildConfig.DEBUG) {
            i.getStringExtra("pc_token")?.let { t -> i.getStringExtra("pc_address")?.let { a -> pc.setPairedForTest(a, t) } }
            i.getStringExtra("pc_engine")?.let { pc.engine = it }
            if (i.hasExtra("use_pc")) {
                val on = i.getBooleanExtra("use_pc", false)
                usePc = !on  // so setWhere sees a change and reconnects
                b.whereGroup.check(if (on) R.id.pcButton else R.id.phoneButton)
                setWhere(on)
            }
        }
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

    /** How a new voice is being added: reading the example text, any recorded speech, or an audio file. */
    private enum class AddMode { EXAMPLE, SPEECH, FILE }

    /** Result of the review / edit dialogs. */
    private sealed class Review {
        class Saved(val name: String, val transcript: String) : Review()
        object Redo : Review()
    }

    private var recordingMode: AddMode? = null
    private var dialogPlayButton: MaterialButton? = null  // Play inside the review dialog; its text follows the player

    private fun haveAsr() = File(ModelFiles.dir(this), ModelFiles.ASR).isFile

    /** Reloads the library list and selects [selectId] (or the newest voice). */
    private fun refreshVoices(selectId: String?, check: Boolean = true) {
        voices = library.list()
        b.voiceMenu.setSimpleItems(voices.map { it.name }.toTypedArray())
        selectVoice(voices.firstOrNull { it.id == selectId } ?: voices.lastOrNull(), check)
    }

    /** [check]: run the one-time transcript check if this voice hasn't had one (callers that check it themselves pass false). */
    private fun selectVoice(v: SavedVoice?, check: Boolean = true) {
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
        if (check) checkUncheckedVoice()
    }

    /** Voices made before 0.4 (or whose transcript changed) are checked once, when selected and the models are ready. */
    private fun checkUncheckedVoice() {
        val v = current ?: return
        if (usePc || busy || Engine.pipeline == null || v.checkRatio != null || !v.toVoice().accentMode) return
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
            v.transcriptMismatch -> "⚠ transcript doesn't match the clip (tap Edit transcript)"
            usePc -> "accent matching on"
            // the clip's codes are re-read for every part of a text: ~0.4 s per clip second on a Snapdragon 8 Gen 2
            else -> "accent matching on (each part starts about %.0f s later)".format(v.promptCodes!!.size / 50.0 * 0.4)
        }
        b.voiceStatus.text = "%s · %.1f s clip · %s".format(v.source.ifBlank { "Voice" }, v.seconds, accent)
        b.transcriptView.text = if (v.transcript.isBlank()) "What's said in the clip: (not set)"
        else "What's said in the clip: “${v.transcript}”"
    }

    private fun renameVoice() {
        val v = current ?: return
        lifecycleScope.launch {
            val name = askText("Rename voice", v.name) ?: return@launch
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
            val clip = library.refWav(v.id).takeIf { it.isFile }?.let { AudioIO.readWav16(it) }
            val r = reviewDialog("What's said in the clip", null, v.transcript,
                "Type exactly what's said in the clip, word for word (play it to check). A transcript that doesn't match " +
                    "makes outputs repeat part of the clip or come out almost silent. Clear it to turn accent matching off " +
                    "for this voice." + if (clip != null && canTranscribe()) " Transcribe again redoes it from the clip." else "",
                clip, offerRedo = clip != null && canTranscribe()) ?: return@launch
            when (r) {
                is Review.Redo -> redoFromClip(v, clip!!)
                is Review.Saved -> {
                    v.transcript = r.transcript
                    v.checkRatio = null
                    library.save(v)
                    SessionLog.i("voice", "transcript of '${v.name}' edited (${r.transcript.length} chars)")
                    showVoiceStatus()
                    checkVoice(v)
                }
            }
        }
    }

    /** Re-derives a voice's speech codes and transcript from its stored clip (fixes voices made before 0.5). */
    private suspend fun redoFromClip(v: SavedVoice, clip: FloatArray) {
        val p = Engine.pipeline ?: return
        val wav = AudioIO.cutAtPause(clip, MiraPipeline.MAX_CLIP_SAMPLES)
        setBusy(true)
        b.voiceStatus.text = "Working out what's said in the clip…"
        val (codes, heard) = try {
            withContext(Engine.worker) {
                if (File(ModelFiles.dir(this@MainActivity), ModelFiles.SEMANTIC).isFile) p.encodePromptCodes(wav) else null
            } to transcribeClip(p, wav)
        } catch (e: Throwable) {
            SessionLog.e("voice", "re-transcribing '${v.name}' failed", e)
            setBusy(false)
            showVoiceStatus()
            toast("Couldn't transcribe the clip: ${e.message}")
            return
        }
        setBusy(false)
        showVoiceStatus()
        SessionLog.i("voice", "re-transcribed '${v.name}': %.1f s, %d codes, ${heard?.length ?: 0} chars".format(wav.size / 16000f, codes?.size ?: 0))
        val r = reviewDialog("What's said in the clip", null, heard.orEmpty(),
            "Speech recognition wrote this down. Play the clip and fix any word it misheard." +
                if (wav.size < clip.size) " The clip is shortened to %.0f s, the longest usable.".format(wav.size / 16000f) else "",
            wav, offerRedo = false) as? Review.Saved ?: return
        if (wav.size < clip.size) library.saveClip(v.id, wav)
        v.seconds = wav.size / 16000f
        v.promptCodes = codes
        v.transcript = r.transcript
        v.checkRatio = null
        library.save(v)
        showVoiceStatus()
        checkVoice(v)
    }

    /** Clip review: an explanation, a Play button, optionally the name, and the transcript to check. */
    private suspend fun reviewDialog(title: String, name: String?, transcript: String, note: String, clip: FloatArray?,
                                     offerRedo: Boolean, cancelLabel: String = "Cancel"): Review? =
        suspendCancellableCoroutine { c ->
            val pad = (16 * resources.displayMetrics.density).toInt()
            val box = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(pad + pad / 2, pad / 2, pad + pad / 2, 0)
            }
            box.addView(TextView(this).apply {
                text = note
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
            })
            if (clip != null) {
                val play = MaterialButton(this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                    text = getString(R.string.play)
                    setOnClickListener { if (player.playing == "review") player.stop() else player.play("review", clip, 16000) }
                }
                dialogPlayButton = play
                box.addView(play)
            }
            val nameField = name?.let {
                TextInputEditText(this).apply {
                    setText(it)
                    inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
                }.also { f -> box.addView(TextInputLayout(this).apply { hint = "Name"; addView(f) }) }
            }
            val words = TextInputEditText(this).apply {
                setText(transcript)
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                minLines = 3
            }
            box.addView(TextInputLayout(this).apply { hint = "What's said in the clip (word for word)"; addView(words) })
            val dialog = MaterialAlertDialogBuilder(this)
                .setTitle(title)
                .setView(android.widget.ScrollView(this).apply { addView(box) })
                .setPositiveButton("Save") { _, _ ->
                    if (c.isActive) c.resume(Review.Saved(nameField?.text?.toString()?.trim()?.ifBlank { name } ?: "",
                        words.text.toString().trim()))
                }
                .setNegativeButton(cancelLabel) { _, _ -> if (c.isActive) c.resume(null) }
                .apply { if (offerRedo) setNeutralButton("Transcribe again") { _, _ -> if (c.isActive) c.resume(Review.Redo) } }
                .setOnCancelListener { if (c.isActive) c.resume(null) }
                .setOnDismissListener {
                    if (player.playing == "review") player.stop()
                    dialogPlayButton = null
                }
                .show()
            c.invokeOnCancellation { dialog.dismiss() }
        }

    /** A dialog with one text field (and an optional [message] above it); returns the trimmed text, or null if cancelled. */
    private suspend fun askText(title: String, initial: String, message: String? = null, digits: Boolean = false): String? =
        suspendCancellableCoroutine { c ->
            val input = TextInputEditText(this).apply {
                setText(initial)
                setSelection(initial.length)
                inputType = if (digits) InputType.TYPE_CLASS_NUMBER else InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            }
            val box = TextInputLayout(this).apply {
                setPadding(48, 16, 48, 0)
                addView(input)
            }
            MaterialAlertDialogBuilder(this)
                .setTitle(title)
                .apply { if (message != null) setMessage(message) }
                .setView(box)
                .setPositiveButton(if (digits) "Pair" else "Save") { _, _ -> if (c.isActive) c.resume(input.text?.toString()?.trim().orEmpty()) }
                .setNegativeButton("Cancel") { _, _ -> if (c.isActive) c.resume(null) }
                .setOnCancelListener { if (c.isActive) c.resume(null) }
                .show()
        }

    private fun startRecording(mode: AddMode) {
        player.stop()
        recording.set(true)
        recordingMode = mode
        updateButtons()
        b.recordLevel.visibility = View.VISIBLE
        b.script.visibility = View.VISIBLE
        b.script.text = if (mode == AddMode.EXAMPLE) getString(R.string.script_intro) + "\n\n“" + getString(R.string.script_text) + "”"
        else "Recording. Speak, or play the speech you want to copy (a video, a voice message) near the phone, then tap " +
            "Stop recording. Up to 25 s is used."
        SessionLog.i("voice", "recording started ($mode)")
        lifecycleScope.launch {
            val wav = try {
                withContext(Dispatchers.IO) {
                    AudioIO.record(40f, { recording.get() }) { peak, secs ->
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
            recordingMode = null
            b.recordLevel.visibility = View.GONE
            b.script.visibility = View.GONE
            updateButtons()
            SessionLog.i("voice", "recording stopped: %.1f s".format(wav.size / 16000f))
            if (mode == AddMode.EXAMPLE) addVoice(wav, "Read example", getString(R.string.script_text), library.nextName(), ask = true)
            else addVoice(wav, "Recorded", null, library.nextName(), ask = true)
        }
    }

    private fun loadVoiceFromUri(uri: Uri, name: String, preset: String?, ask: Boolean, then: (() -> Unit)? = null,
                                 expected: String? = null) {
        lifecycleScope.launch {
            try {
                b.voiceStatus.text = "Reading audio…"
                val t0 = SystemClock.elapsedRealtime()
                val wav = withContext(Dispatchers.IO) { AudioIO.decodeToMono(this@MainActivity, uri, 16000, 60f) }
                SessionLog.i("voice", "decoded '$name': %.1f s in %d ms".format(wav.size / 16000f, SystemClock.elapsedRealtime() - t0))
                if (addVoice(wav, if (expected != null) "Read example" else "File", expected, name, ask, preset)) then?.invoke()
            } catch (e: Throwable) {
                SessionLog.e("voice", "couldn't read '$name'", e)
                b.voiceStatus.text = "Couldn't read that file: ${e.message}"
            }
        }
    }

    /**
     * Turns a clip into a voice: trims silence, keeps at most 25 s (cut at a pause), derives voice tokens and speech
     * codes, works out what was said (speech recognition, or the example text when that's what was read), asks the
     * user to check it, saves the voice and checks that transcript and clip match. [preset]: a known transcript (adb).
     */
    private suspend fun addVoice(raw: FloatArray, source: String, expected: String?, defaultName: String, ask: Boolean,
                                 preset: String? = null): Boolean {
        val trimmed = AudioIO.trimSilence(raw)
        if (trimmed.size < 16000 * 1.5) {
            b.voiceStatus.text = "That clip is too short (%.1f s of sound). Aim for 8 seconds or more.".format(trimmed.size / 16000f)
            SessionLog.w("voice", "clip too short: %.1f s after trimming".format(trimmed.size / 16000f))
            return false
        }
        val wav = AudioIO.cutAtPause(trimmed, MiraPipeline.MAX_CLIP_SAMPLES)
        val cut = wav.size < trimmed.size
        val p = Engine.pipeline ?: run { b.voiceStatus.text = "Models aren't loaded yet"; return false }
        setBusy(true)
        b.voiceStatus.text = "Analysing voice…"
        val haveSemantic = File(ModelFiles.dir(this), ModelFiles.SEMANTIC).isFile
        val t0 = SystemClock.elapsedRealtime()
        var heard: String? = null
        val (tokens, codes) = try {
            val tc = withContext(Engine.worker) {
                p.encodeVoiceTokens(wav) to if (haveSemantic) p.encodePromptCodes(wav) else null
            }
            if (preset == null && canTranscribe()) {
                b.voiceStatus.text = "Working out what was said…"
                val t1 = SystemClock.elapsedRealtime()
                heard = transcribeClip(p, wav)
                SessionLog.i("voice", "speech recognition: %.1f s clip in %d ms, ${heard?.length ?: 0} chars"
                    .format(wav.size / 16000f, SystemClock.elapsedRealtime() - t1))
            }
            tc
        } catch (e: Throwable) {
            SessionLog.e("voice", "voice analysis failed", e)
            b.voiceStatus.text = "Couldn't analyse the voice: ${e.message}"
            setBusy(false)
            return false
        }
        SessionLog.i("voice", "analysed %.1f s clip (%.1f s before trimming%s) in %d ms: %d prompt codes · %s".format(
            wav.size / 16000f, raw.size / 16000f, if (cut) ", cut at a pause" else "", SystemClock.elapsedRealtime() - t0,
            codes?.size ?: 0, SessionLog.vitals()))
        setBusy(false)
        showVoiceStatus()
        val werToExample = if (expected != null && heard != null) wordErrorRate(expected, heard) else null
        if (werToExample != null) SessionLog.i("voice", "example read: %.0f%% of words differ from what was heard".format(werToExample * 100))
        val transcript: String
        var note: String
        when {
            preset != null -> { transcript = preset; note = "" }
            expected != null && !cut && (heard == null || (werToExample!! <= 0.25 && sameEnding(expected, heard!!))) -> {
                transcript = expected
                note = "You read the example text. Check it matches what you actually said, word for word."
                if (werToExample != null && werToExample > 0.03) note += "\n\nSpeech recognition heard:\n“$heard”"
            }
            heard != null -> {
                transcript = heard!!
                note = (if (expected != null) "What was said differs from the example text, so this is what speech recognition heard. "
                else "Speech recognition wrote down what was said. ") + "Play the clip and fix any word it misheard."
            }
            else -> {
                transcript = expected.orEmpty()
                note = "Type exactly what's said in the clip, word for word, or leave it empty to copy the voice without its accent."
            }
        }
        if (cut) note += "\n\nOnly the first %.0f s are kept (the longest a voice clip can be), cut at a pause.".format(wav.size / 16000f)
        SessionLog.i("voice", "transcript chosen: ${if (transcript == expected) "the example text" else if (transcript == heard) "speech recognition" else "preset/typed"}")
        val (name, words) = if (ask) {
            val r = reviewDialog("Save voice", defaultName, transcript, note, wav, offerRedo = false, cancelLabel = "Discard")
                as? Review.Saved ?: run {
                    SessionLog.i("voice", "new voice discarded")
                    return false
                }
            r.name to r.transcript
        } else defaultName to transcript
        val v = library.add(name, source, wav, tokens, codes, words)
        SessionLog.i("voice", "added '${v.name}' (${v.id}), transcript ${words.length} chars" +
            if (heard != null && words != heard) " (edited after speech recognition)" else "")
        refreshVoices(v.id, check = false)
        val h = heard
        if (expected != null && words == expected && h != null && h != words) {
            // The example text is what was meant; the clip may still end differently (a dropped or cut last word),
            // which makes outputs come out silent. If it fails the check, what was heard usually passes.
            val ratio = checkVoice(v, warn = false)
            if (ratio != null && (ratio < 0.6 || ratio > 2.0)) {
                SessionLog.i("voice", "example text failed the check (%.2f): trying what was heard".format(ratio))
                v.transcript = h
                v.checkRatio = null
                library.save(v)
                showVoiceStatus()
                val r2 = checkVoice(v)
                if (r2 != null && r2 >= 0.6 && r2 <= 2.0) toast("The clip didn't quite match the example text, so the voice uses what was heard instead.")
            }
        } else checkVoice(v)
        return true
    }

    /** True if the last three words agree: the end of the clip has to match its transcript exactly. */
    private fun sameEnding(a: String, b: String): Boolean {
        fun words(s: String) = s.lowercase().replace(Regex("[^a-z0-9' ]"), " ").split(" ").filter { it.isNotBlank() }
        return words(a).takeLast(3) == words(b).takeLast(3)
    }

    /** Word error rate of [hyp] against [ref], ignoring case and punctuation. */
    private fun wordErrorRate(ref: String, hyp: String): Double {
        fun words(s: String) = s.lowercase().replace(Regex("[^a-z0-9' ]"), " ").split(" ").filter { it.isNotBlank() }
        val r = words(ref)
        val h = words(hyp)
        if (r.isEmpty()) return if (h.isEmpty()) 0.0 else 1.0
        var prev = IntArray(h.size + 1) { it }
        for (i in 1..r.size) {
            val cur = IntArray(h.size + 1)
            cur[0] = i
            for (j in 1..h.size) cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (r[i - 1] == h[j - 1]) 0 else 1)
            prev = cur
        }
        return prev[h.size].toDouble() / r.size
    }

    /** Test-speaks a short sentence to catch a transcript that doesn't match the clip (see MiraPipeline.checkVoice). */
    private suspend fun checkVoice(v: SavedVoice, warn: Boolean = true): Double? {
        val voice = v.toVoice()
        val p = Engine.pipeline
        if (!voice.accentMode || p == null) return null
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
        ratio ?: return null
        SessionLog.i("voice", "check '${v.name}': test sentence came out %.2fx the expected length".format(ratio))
        v.checkRatio = ratio
        library.save(v)
        showVoiceStatus()
        val problem = when {
            ratio < 0.6 -> "The test sentence came out almost silent: the clip seems to contain more speech than its " +
                "transcript, so the model thinks your text has already been said."
            ratio > 2.0 -> "The test sentence came out far too long: the transcript seems to have words that aren't in the " +
                "clip, and they'd be spoken at the start of every output."
            else -> return ratio
        }
        if (!warn) return ratio
        SessionLog.w("voice", "transcript of '${v.name}' doesn't match its clip (ratio %.2f)".format(ratio))
        MaterialAlertDialogBuilder(this)
            .setTitle("Transcript doesn't match '${v.name}'")
            .setMessage("$problem\n\nMake it match what's said in the clip word for word, or record again. Until then, parts " +
                "that come out silent are redone without accent matching.")
            .setPositiveButton("Edit transcript") { _, _ -> editTranscript() }
            .setNegativeButton("Keep as is", null)
            .show()
        return ratio
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
        if (usePc) return speakOnPc(text, play, outName, seedOverride)
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

    // ------------------------------------------------------------------ on my PC

    private var pcEngines: List<Pair<String, String>> = emptyList()

    private fun setWhere(onPc: Boolean) {
        if (onPc == usePc) return
        usePc = onPc
        prefs.edit().putBoolean("use_pc", onPc).apply()
        SessionLog.i("pc", if (onPc) "switched to the PC" else "switched to the phone")
        showWhere()
        showVoiceStatus()
        updateButtons()
        if (onPc) connectPc() else checkUncheckedVoice()
    }

    /** Settings that only apply on the phone are hidden in PC mode. */
    private fun showWhere() {
        val pcViews = listOf(b.pcStatus, b.pcSetupButton)
        for (v in pcViews) v.visibility = if (usePc) View.VISIBLE else View.GONE
        b.pcEngineLayout.visibility = if (usePc && pcEngines.size > 1) View.VISIBLE else View.GONE
        b.hdSwitch.visibility = if (usePc) View.GONE else View.VISIBLE
    }

    /**
     * Finds the PC server (saved address, mDNS, or typed in), pairs with it if needed and starts loading its model.
     * Not [interactive] (app start): no dialogs, so a PC that's switched off only shows a status line.
     */
    private fun connectPc(interactive: Boolean = true) {
        if (pcConnecting) return
        pcConnecting = true
        updateButtons()
        lifecycleScope.launch {
            try {
                b.pcStatus.text = "Looking for your PC…"
                var addr = pc.address
                var info = addr?.let { a -> runCatching { withContext(Dispatchers.IO) { pc.info(a) } }.getOrNull() }
                if (info == null) {
                    addr = pc.discover()
                    SessionLog.i("pc", "mDNS: ${addr ?: "nothing found"}")
                    if (addr == null && !interactive) {
                        b.pcStatus.text = "Your PC isn't answering. Is it on, with the Mira Voice server running? Tap Change PC to look again."
                        return@launch
                    }
                    if (addr == null) {
                        addr = askText("PC address", pc.address ?: "192.168.1.", "Your PC wasn't found automatically. Is the Mira " +
                            "Voice server running on it? Type its address (port 8790 unless you changed it).")
                            ?.takeIf { it.isNotBlank() }?.let { if (':' in it) it else "$it:8790" }
                    }
                    if (addr == null) {
                        b.pcStatus.text = "No PC set. Tap Change PC to try again, or switch back to the phone."
                        return@launch
                    }
                    info = try {
                        withContext(Dispatchers.IO) { pc.info(addr) }
                    } catch (e: Throwable) {
                        SessionLog.w("pc", "no server at $addr: ${e.message}")
                        b.pcStatus.text = "Couldn't reach the Mira Voice server at $addr (${e.message}). Tap Change PC to try again."
                        return@launch
                    }
                }
                // a paired PC found at a new address (DHCP) keeps its token; a different PC answers 401 and re-pairs
                if (pc.paired && pc.address != addr) pc.address = addr
                if (!pc.paired) {
                    if (!interactive) {
                        b.pcStatus.text = "Found ${info.host}, but this phone isn't paired with it. Tap Change PC to pair."
                        return@launch
                    }
                    val req = withContext(Dispatchers.IO) { pc.startPairing(addr!!) }
                    val code = askText("Pair with ${info.host}", "", "Enter the 6-digit code shown on your PC's screen.", digits = true)
                    if (code.isNullOrBlank()) {
                        b.pcStatus.text = "Not paired. Tap Change PC to pair."
                        return@launch
                    }
                    withContext(Dispatchers.IO) { pc.confirmPairing(addr!!, req, code) }
                    SessionLog.i("pc", "paired with ${info.host} at $addr")
                }
                pcEngines = info.engines
                if (pcEngines.none { it.first == pc.engine }) pcEngines.firstOrNull()?.let { pc.engine = it.first }
                b.pcEngineMenu.setSimpleItems(pcEngines.map { it.second }.toTypedArray())
                b.pcEngineMenu.setText(pcEngines.firstOrNull { it.first == pc.engine }?.second ?: "", false)
                b.pcStatus.text = "Using ${info.host} · ${info.gpu}. Speech is made there; voices and text are sent over your Wi-Fi."
                SessionLog.i("pc", "connected: ${info.host} · ${info.gpu} · server ${info.version} at $addr")
                warmPc()
            } catch (e: Throwable) {
                SessionLog.e("pc", "connecting to the PC failed", e)
                b.pcStatus.text = "Couldn't connect to the PC: ${e.message}. Tap Change PC to try again."
            } finally {
                pcConnecting = false
                showWhere()
                updateButtons()
                handleAutomation()
            }
        }
    }

    /** The PC loads its voice model on first use (up to a minute); start that now rather than when Speak is tapped. */
    private fun warmPc() {
        lifecycleScope.launch {
            runCatching { withContext(Dispatchers.IO) { pc.warmup() } }
                .onFailure { SessionLog.w("pc", "warm-up failed: ${it.message}") }
        }
    }

    private fun canTranscribe() = haveAsr() || (usePc && pc.paired)

    /** What's said in a clip: on the PC in PC mode (Whisper large, better with accents), else on the phone. */
    private suspend fun transcribeClip(p: MiraPipeline, wav: FloatArray): String? {
        if (usePc && pc.paired) {
            try {
                return withContext(Dispatchers.IO) { pc.transcribe(wav) }.also { SessionLog.i("voice", "transcribed on the PC") }
            } catch (e: Throwable) {
                SessionLog.w("voice", "PC speech recognition failed (${e.message}); using the phone's")
            }
        }
        return if (haveAsr()) withContext(Engine.worker) { p.transcribe(wav) } else null
    }

    private fun speakOnPc(text: String, play: Boolean, outName: String?, seedOverride: Int?) {
        val v = current ?: run { toast("Add a voice first"); return }
        val ref = library.refWav(v.id).takeIf { it.isFile } ?: run { toast("This voice has no clip to send to the PC"); return }
        if (text.isBlank()) return
        if (!pc.paired) { connectPc(); return }
        player.stop()
        cancel.set(false)
        setBusy(true)
        b.stopButton.isEnabled = true
        val seed = seedOverride ?: if (b.randomSeedSwitch.isChecked) (System.nanoTime() and 0x7fffffff).toInt() else 0
        // the PC clones from clip + transcript ("accent matching"), or from the clip alone when that's off
        val transcript = if (b.accentSwitch.isChecked && !v.transcriptMismatch) v.transcript else ""
        SessionLog.i("speak", "start on the PC (${pc.engine}): ${text.length} chars, voice '${v.name}', transcript ${transcript.isNotEmpty()}, seed $seed")
        lifecycleScope.launch {
            val t0 = SystemClock.elapsedRealtime()
            val ticker = launch {
                while (isActive) {
                    val s = (SystemClock.elapsedRealtime() - t0) / 1000
                    b.genStatus.text = "Speaking on your PC… $s s" + if (s >= 8) " (the first time, the PC loads its model: up to a minute)" else ""
                    delay(1000)
                }
            }
            val r = try {
                withContext(Dispatchers.IO) { pc.speak(ref, transcript, text, seed) { cancel.get() } }
            } catch (e: Throwable) {
                if (!cancel.get()) {
                    SessionLog.e("speak", "PC synthesis failed", e)
                    b.genStatus.text = "The PC couldn't speak it: ${e.message}" +
                        if (!pc.paired) "\nThe PC doesn't know this phone any more: tap Change PC to pair again." else ""
                }
                null
            }
            ticker.cancel()
            setBusy(false)
            val totalMs = SystemClock.elapsedRealtime() - t0
            if (cancel.get()) {
                b.genStatus.text = "Stopped"
                SessionLog.i("speak", "PC synthesis cancelled after %.1f s".format(totalMs / 1000.0))
            } else if (r != null && r.audio.isNotEmpty()) {
                lastAudio = r.audio
                lastRate = r.rate
                val secs = r.audio.size / r.rate.toDouble()
                b.genStatus.text = "Done on your PC: %.1f s of speech in %.1f s".format(secs, totalMs / 1000.0)
                b.stats.text = buildString {
                    append("PC model   %s%n".format(pcEngines.firstOrNull { it.first == pc.engine }?.second ?: pc.engine))
                    append("Parts      %d   audio %.1f s at %d kHz%n".format(r.parts, secs, r.rate / 1000))
                    if (r.loadSeconds >= 1) append("Loading    %.1f s (first use)%n".format(r.loadSeconds))
                    append("Generating %.1f s  (%.2fx real time)%n".format(r.genSeconds, r.genSeconds / secs))
                    append("Total      %.1f s incl. Wi-Fi".format(totalMs / 1000.0))
                }
                SessionLog.i("speak", "MIRA_RESULT " + JSONObject().apply {
                    put("where", "pc"); put("engine", pc.engine); put("audio_s", secs); put("gen_s", r.genSeconds)
                    put("load_s", r.loadSeconds); put("total_s", totalMs / 1000.0); put("parts", r.parts); put("rate", r.rate)
                })
                if (outName != null) {
                    val f = File(getExternalFilesDir("out"), outName)
                    AudioIO.writeWav(f, r.audio, r.rate)
                    SessionLog.i("speak", "wrote ${f.path}")
                }
                if (play) player.play("output", r.audio, r.rate)
                updatePlayButtons()
            }
            handleAutomation()
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
        if (busy || (Engine.pipeline == null && !usePc)) return
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
            // voice_expected: test the "read the example" path with a file (the text the speaker was meant to read)
            loadVoiceFromUri(Uri.fromFile(File(path)), name, i.getStringExtra("voice_transcript"), ask = false, then = go,
                expected = i.getStringExtra("voice_expected"))
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
        b.speakButton.isEnabled = (if (usePc) !busy && !pcConnecting else ready) && current != null
        val rec = recordingMode
        b.exampleButton.isEnabled = if (recording.get()) rec == AddMode.EXAMPLE else ready
        b.recordButton.isEnabled = if (recording.get()) rec == AddMode.SPEECH else ready
        b.exampleButton.text = getString(if (rec == AddMode.EXAMPLE) R.string.stop_recording else R.string.read_example)
        b.recordButton.text = getString(if (rec == AddMode.SPEECH) R.string.stop_recording else R.string.record_speech)
        b.pickButton.isEnabled = ready && !recording.get()
        b.renameVoiceButton.isEnabled = current != null && !busy
        b.editTranscriptButton.isEnabled = current != null && !busy
        b.deleteVoiceButton.isEnabled = current != null && !busy
        b.voiceLayout.isEnabled = !busy
        b.threadsSlider.isEnabled = !busy
        b.ggufLayout.isEnabled = !busy
        b.downloadButton.isEnabled = !busy
        b.whereGroup.isEnabled = !busy
        b.pcSetupButton.isEnabled = !busy && !pcConnecting
        updatePlayButtons()
    }

    private fun updatePlayButtons() {
        b.playVoiceButton.isEnabled = current?.let { library.refWav(it.id).isFile } == true
        b.playVoiceButton.text = getString(if (player.playing == "voice") R.string.stop else R.string.play)
        b.replayButton.text = getString(if (player.playing == "output") R.string.stop else R.string.play)
        dialogPlayButton?.text = getString(if (player.playing == "review") R.string.stop else R.string.play)
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
