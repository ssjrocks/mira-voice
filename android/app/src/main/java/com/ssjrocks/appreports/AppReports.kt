// Shared crash and problem reporter for my Android apps.
//
// Source of truth: ~/dev/app-reports/android/AppReports.kt (github.com/ssjrocks/app-reports).
// Copy it into an app with tools/add-to-app.sh; don't edit the per-app copies.
//
// What it does, with no setup beyond the manifest <provider> entry (see README):
//  - Java/Kotlin crashes: saved to disk as the process dies (with this process's logcat), sent
//    on the next launch.
//  - ANRs and native crashes (API 30+): read from ApplicationExitInfo on the next launch.
//  - AppReports.log(tag, msg): a small on-device log (also goes to logcat) attached to reports.
//  - AppReports.reportError(t): send a caught, non-fatal exception.
//  - AppReports.attach(name) { text }: include the app's own log (or any text) in every report.
//  - AppReports.sendReport(note): the "Report a problem" button; sends the recent log now.
//    AppReports.showReportDialog(activity) is the ready-made dialog for that button.
//
// Reports go to the URL in the manifest's com.ssjrocks.appreports.URL <meta-data>, which the build
// fills in from the appReportsUrl Gradle property (in ~/.gradle/gradle.properties, so it isn't in
// the app's source). Without it the app works but reports stay on the device. They're write-only;
// read them with the `app-reports` command on the desktop. No dependencies beyond the Android SDK.
package com.ssjrocks.appreports

import android.app.Activity
import android.app.ActivityManager
import android.app.AlertDialog
import android.app.Application
import android.app.ApplicationExitInfo
import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.content.ContextWrapper
import android.content.SharedPreferences
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.text.InputType
import android.util.Base64
import android.util.Log
import android.view.Gravity
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.Toast
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.PrintWriter
import java.io.StringWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPOutputStream
import kotlin.system.exitProcess

object AppReports {
    const val LIB_VERSION = 1
    private const val META_URL = "com.ssjrocks.appreports.URL"
    private const val TAG = "AppReports"
    private const val LOG_MAX_BYTES = 256 * 1024
    private const val LOGCAT_LINES = 1500
    private const val MAX_TEXT = 512 * 1024
    private const val MAX_PENDING = 20
    private const val PENDING_MAX_AGE_MS = 7L * 24 * 3600 * 1000
    private const val MAX_ERRORS_PER_RUN = 10

    @Volatile
    private var app: Context? = null
    @Volatile
    private var endpoint = ""
    private val logLock = Any()
    private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "app-reports").apply { isDaemon = true } }
    private val stamp = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)
    private val sentErrors = HashSet<String>()
    private val attachments = ConcurrentHashMap<String, () -> String?>()

    /** Called by [AppReportsInit] at process start; safe to call again from anywhere. */
    fun init(context: Context) {
        if (app != null) return
        val c = context.applicationContext ?: context
        synchronized(this) {
            if (app != null) return
            app = c
        }
        endpoint = readEndpoint(c)
        if (endpoint.isEmpty()) Log.w(TAG, "no report URL in this build (appReportsUrl); reports stay on the device")
        installCrashHandler()
        io.execute {
            runCatching { checkExitReasons(c) }.onFailure { Log.w(TAG, "exit reasons", it) }
            flush(c)
        }
    }

    /** Add a line to the on-device log that goes with every report (and to logcat). */
    fun log(tag: String, message: String) {
        Log.i(tag, message)
        append("I", tag, message)
    }

    /** Like [log] but at error level, with the stack trace if given. Doesn't send anything. */
    fun error(tag: String, message: String, t: Throwable? = null) {
        if (t != null) Log.e(tag, message, t) else Log.e(tag, message)
        append("E", tag, if (t != null) "$message\n${stackOf(t)}" else message)
    }

    /**
     * Attach app-specific text (e.g. the app's own log file) to every report under [name].
     * [provider] runs when a report is built, including inside the crash handler, so keep it to
     * a quick file read. Only the last 512 KB of what it returns is kept.
     */
    fun attach(name: String, provider: () -> String?) {
        attachments[name] = provider
    }

    /** Send a caught (non-fatal) exception. Repeats of the same one are sent once per run. */
    fun reportError(t: Throwable, message: String? = null) {
        val c = app ?: return
        error("AppReports", message ?: "reported error", t)
        val key = "${t.javaClass.name}@${t.stackTrace.firstOrNull()}"
        synchronized(sentErrors) {
            if (sentErrors.size >= MAX_ERRORS_PER_RUN || !sentErrors.add(key)) return
        }
        io.execute {
            runCatching {
                val r = baseReport(c, "error")
                r.put("thread", Thread.currentThread().name)
                r.put("exception", exceptionJson(t))
                if (message != null) r.put("note", message)
                r.put("log", recentLog(c))
                r.put("logcat", logcat("-t", "$LOGCAT_LINES", "--pid=${Process.myPid()}"))
                save(c, r)
            }
            flush(c)
        }
    }

    /**
     * "Report a problem": send the recent log, this process's logcat and [note] now. [onDone] runs
     * on the main thread with true once the server has it (false: queued, retried next launch).
     */
    fun sendReport(note: String? = null, extra: Map<String, Any?> = emptyMap(), onDone: ((Boolean) -> Unit)? = null) {
        val c = app
        if (c == null) {
            onDone?.let { cb -> Handler(Looper.getMainLooper()).post { cb(false) } }
            return
        }
        append("I", "AppReports", "problem report: ${note ?: ""}")
        io.execute {
            val ok = runCatching {
                val r = baseReport(c, "report")
                if (!note.isNullOrBlank()) r.put("note", note)
                if (extra.isNotEmpty()) r.put("extra", JSONObject(extra))
                r.put("log", recentLog(c))
                r.put("logcat", logcat("-t", "$LOGCAT_LINES", "--pid=${Process.myPid()}"))
                val f = save(c, r)
                flush(c)
                !f.exists()
            }.getOrDefault(false)
            onDone?.let { cb -> Handler(Looper.getMainLooper()).post { cb(ok) } }
        }
    }

    /**
     * The standard "Report a problem" dialog: an optional note, then [sendReport] with a toast
     * saying whether it went. Takes an Activity or anything wrapping one (e.g. Compose's
     * LocalContext.current).
     */
    fun showReportDialog(context: Context, extra: Map<String, Any?> = emptyMap()) {
        var c: Context? = context
        while (c !is Activity && c is ContextWrapper) c = c.baseContext
        val activity = c as? Activity ?: run {
            sendReport(null, extra)
            return
        }
        val pad = (20 * activity.resources.displayMetrics.density).toInt()
        val input = EditText(activity).apply {
            hint = "What went wrong? (optional)"
            minLines = 3
            gravity = Gravity.TOP or Gravity.START
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        }
        val box = FrameLayout(activity).apply {
            setPadding(pad, pad / 2, pad, 0)
            addView(input)
        }
        AlertDialog.Builder(activity)
            .setTitle("Report a problem")
            .setMessage("Sends this app's recent log and device details (no personal data) so the problem can be looked into.")
            .setView(box)
            .setPositiveButton("Send") { _, _ ->
                val app = activity.applicationContext
                sendReport(input.text.toString().trim().ifBlank { null }, extra) { ok ->
                    Toast.makeText(
                        app,
                        if (ok) "Report sent. Thanks!" else "Couldn't send now; it will be sent automatically later.",
                        Toast.LENGTH_LONG,
                    ).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // --- crash capture ---------------------------------------------------------------------------

    private class CrashHandler(val prev: Thread.UncaughtExceptionHandler?) : Thread.UncaughtExceptionHandler {
        override fun uncaughtException(t: Thread, e: Throwable) {
            runCatching {
                val c = app ?: return@runCatching
                append("E", "AppReports", "uncaught on thread ${t.name}: $e") // full stack is in "exception"
                val r = baseReport(c, "crash")
                r.put("thread", t.name)
                r.put("exception", exceptionJson(e))
                r.put("log", recentLog(c))
                r.put("logcat", logcat("-t", "$LOGCAT_LINES", "--pid=${Process.myPid()}"))
                save(c, r)
            }
            if (prev != null) {
                prev.uncaughtException(t, e)
            } else {
                Process.killProcess(Process.myPid())
                exitProcess(10)
            }
        }
    }

    private fun installCrashHandler() {
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        if (prev is CrashHandler) return
        Thread.setDefaultUncaughtExceptionHandler(CrashHandler(prev))
    }

    /** ANRs and native crashes can't be caught in-process; Android records them for the next run. */
    private fun checkExitReasons(c: Context) {
        if (Build.VERSION.SDK_INT < 30) return
        val am = c.getSystemService(ActivityManager::class.java) ?: return
        val p = prefs(c)
        val since = p.getLong("exitCheckedAt", 0L)
        val infos = am.getHistoricalProcessExitReasons(null, 0, 16)
        val newest = maxOf(since, infos.maxOfOrNull { it.timestamp } ?: 0L, if (since == 0L) System.currentTimeMillis() else 0L)
        p.edit().putLong("exitCheckedAt", newest).apply()
        if (since == 0L) return // first run of this install: older history isn't ours to report
        for (info in infos.filter { it.timestamp > since }.sortedBy { it.timestamp }) {
            val kind = when (info.reason) {
                ApplicationExitInfo.REASON_ANR -> "anr"
                ApplicationExitInfo.REASON_CRASH_NATIVE -> "native"
                else -> continue // Java crashes are saved by CrashHandler; the rest aren't faults
            }
            val r = baseReport(c, kind)
            r.put("happenedAt", iso(info.timestamp))
            r.put("process", info.processName)
            r.put(
                "exit",
                JSONObject()
                    .put("reason", info.reason)
                    .put("description", info.description)
                    .put("status", info.status)
                    .put("importance", info.importance)
                    .put("pssKb", info.pss)
                    .put("rssKb", info.rss)
                    .put("pid", info.pid),
            )
            val trace = runCatching { info.traceInputStream?.use { readCapped(it, 2 * MAX_TEXT) } }.getOrNull()
            if (trace != null) {
                if (kind == "anr") {
                    r.put("trace", String(trace, Charsets.UTF_8))
                } else {
                    r.put("tombstoneB64", Base64.encodeToString(trace, Base64.NO_WRAP))
                }
            }
            r.put("logcat", logcat("-t", "$LOGCAT_LINES", "--pid=${info.pid}"))
            if (kind == "native") r.put("logcatCrash", logcat("-b", "crash", "-t", "500"))
            r.put("log", recentLog(c))
            save(c, r)
        }
    }

    // --- report building -------------------------------------------------------------------------

    private fun baseReport(c: Context, kind: String): JSONObject {
        val pm = c.packageManager
        val pi = runCatching { pm.getPackageInfo(c.packageName, 0) }.getOrNull()
        val rt = Runtime.getRuntime()
        return JSONObject().apply {
            put("schema", 1)
            put("lib", LIB_VERSION)
            put("app", c.packageName)
            put("appName", runCatching { c.applicationInfo.loadLabel(pm).toString() }.getOrNull())
            put("version", pi?.versionName)
            put("versionCode", pi?.longVersionCode)
            put("debuggable", (c.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0)
            put("kind", kind)
            put("happenedAt", iso(System.currentTimeMillis()))
            put("installId", installId(c))
            put("process", if (Build.VERSION.SDK_INT >= 28) Application.getProcessName() else c.packageName)
            put("uptimeSec", (SystemClock.elapsedRealtime() - Process.getStartElapsedRealtime()) / 1000)
            put(
                "device",
                JSONObject()
                    .put("manufacturer", Build.MANUFACTURER)
                    .put("model", Build.MODEL)
                    .put("device", Build.DEVICE)
                    .put("sdk", Build.VERSION.SDK_INT)
                    .put("release", Build.VERSION.RELEASE)
                    .put("abi", Build.SUPPORTED_ABIS.firstOrNull())
                    .put("build", Build.FINGERPRINT),
            )
            put(
                "memory",
                JSONObject()
                    .put("javaUsedMb", (rt.totalMemory() - rt.freeMemory()) / 1048576)
                    .put("javaMaxMb", rt.maxMemory() / 1048576),
            )
        }
    }

    private fun exceptionJson(t: Throwable) = JSONObject()
        .put("type", t.javaClass.name)
        .put("message", t.message)
        .put("stack", stackOf(t).take(MAX_TEXT))

    private fun stackOf(t: Throwable): String = StringWriter().also { t.printStackTrace(PrintWriter(it)) }.toString()

    /** Own-process logcat needs no permission; other apps' lines are filtered out by Android. */
    private fun logcat(vararg args: String): String? = runCatching {
        val p = ProcessBuilder(listOf("logcat", "-d", "-v", "threadtime") + args).redirectErrorStream(true).start()
        val out = p.inputStream.use { readCapped(it, MAX_TEXT) }
        if (!p.waitFor(3, TimeUnit.SECONDS)) p.destroy()
        String(out, Charsets.UTF_8)
    }.getOrNull()

    private fun readCapped(input: InputStream, max: Int): ByteArray {
        val out = ByteArrayOutputStream()
        val buf = ByteArray(16 * 1024)
        while (out.size() < max) {
            val n = input.read(buf, 0, minOf(buf.size, max - out.size()))
            if (n < 0) break
            out.write(buf, 0, n)
        }
        return out.toByteArray()
    }

    // --- on-device log ---------------------------------------------------------------------------

    private fun dir(c: Context) = File(c.filesDir, "app-reports").also { it.mkdirs() }

    private fun append(level: String, tag: String, message: String) {
        val c = app ?: return
        val line = "${synchronized(stamp) { stamp.format(Date()) }} $level/$tag: $message\n"
        synchronized(logLock) {
            runCatching {
                val f = File(dir(c), "log.txt")
                if (f.length() > LOG_MAX_BYTES) {
                    val keep = f.readText().takeLast(LOG_MAX_BYTES / 2)
                    f.writeText(keep.substringAfter('\n'))
                }
                f.appendText(line)
            }
        }
    }

    private fun recentLog(c: Context): String? = synchronized(logLock) {
        runCatching { File(dir(c), "log.txt").takeIf { it.exists() }?.readText() }.getOrNull()
    }

    // --- queue and upload ------------------------------------------------------------------------

    private fun pendingDir(c: Context) = File(dir(c), "pending").also { it.mkdirs() }

    private fun save(c: Context, report: JSONObject): File {
        if (attachments.isNotEmpty()) {
            val a = JSONObject()
            for ((name, provider) in attachments) {
                runCatching { provider() }.getOrNull()?.let { a.put(name, it.takeLast(MAX_TEXT)) }
            }
            if (a.length() > 0) report.put("attachments", a)
        }
        val d = pendingDir(c)
        val f = File(d, "${System.currentTimeMillis()}-${report.optString("kind")}-${UUID.randomUUID().toString().take(8)}.json")
        val tmp = File(d, f.name + ".tmp")
        tmp.writeText(report.toString())
        tmp.renameTo(f)
        d.listFiles { x -> x.name.endsWith(".json") }?.sortedBy { it.name }?.dropLast(MAX_PENDING)?.forEach { it.delete() }
        return f
    }

    /** Send queued reports, oldest first; stop at the first network/server failure and retry later. */
    private fun flush(c: Context) {
        val files = pendingDir(c).listFiles { x -> x.name.endsWith(".json") }?.sortedBy { it.name } ?: return
        for (f in files) {
            val age = System.currentTimeMillis() - (f.name.substringBefore('-').toLongOrNull() ?: 0L)
            if (age > PENDING_MAX_AGE_MS) {
                f.delete()
                continue
            }
            val code = runCatching { post(f.readBytes()) }.getOrElse {
                Log.w(TAG, "upload failed, will retry: $it")
                return
            }
            when {
                code in 200..299 -> f.delete()
                code == 408 || code == 429 || code >= 500 -> return
                else -> { // the server will never take this one
                    Log.w(TAG, "report rejected ($code), dropping ${f.name}")
                    f.delete()
                }
            }
        }
    }

    private fun post(json: ByteArray): Int {
        val body = ByteArrayOutputStream().also { b -> GZIPOutputStream(b).use { it.write(json) } }.toByteArray()
        val url = endpoint.ifEmpty { throw IllegalStateException("no report URL in this build") }
        val conn = URL(url).openConnection() as HttpURLConnection
        try {
            conn.requestMethod = "POST"
            conn.connectTimeout = 15_000
            conn.readTimeout = 30_000
            conn.doOutput = true
            conn.setFixedLengthStreamingMode(body.size)
            conn.setRequestProperty("Content-Type", "application/json")
            conn.setRequestProperty("Content-Encoding", "gzip")
            conn.setRequestProperty("User-Agent", "AppReports/$LIB_VERSION ${app?.packageName}")
            conn.outputStream.use { it.write(body) }
            return conn.responseCode
        } finally {
            conn.disconnect()
        }
    }

    // --- misc ------------------------------------------------------------------------------------

    private fun readEndpoint(c: Context): String = runCatching {
        val info = if (Build.VERSION.SDK_INT >= 33) {
            c.packageManager.getApplicationInfo(c.packageName, PackageManager.ApplicationInfoFlags.of(PackageManager.GET_META_DATA.toLong()))
        } else {
            @Suppress("DEPRECATION")
            c.packageManager.getApplicationInfo(c.packageName, PackageManager.GET_META_DATA)
        }
        info.metaData?.getString(META_URL)
    }.getOrNull()?.trim().orEmpty()

    private fun prefs(c: Context): SharedPreferences = c.getSharedPreferences("app_reports", Context.MODE_PRIVATE)

    private fun installId(c: Context): String {
        val p = prefs(c)
        return p.getString("installId", null) ?: UUID.randomUUID().toString().also {
            p.edit().putString("installId", it).apply()
        }
    }

    private fun iso(ms: Long): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
        .apply { timeZone = TimeZone.getTimeZone("UTC") }
        .format(Date(ms))
}

/**
 * Starts [AppReports] when the app's process starts, before Application.onCreate, so crashes
 * anywhere are caught without any code in the app. Declared in AndroidManifest.xml.
 */
class AppReportsInit : ContentProvider() {
    override fun onCreate(): Boolean {
        context?.let { AppReports.init(it) }
        return true
    }

    override fun query(uri: Uri, p: Array<out String>?, s: String?, a: Array<out String>?, o: String?): Cursor? = null
    override fun getType(uri: Uri): String? = null
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, s: String?, a: Array<out String>?): Int = 0
    override fun update(uri: Uri, v: ContentValues?, s: String?, a: Array<out String>?): Int = 0
}
