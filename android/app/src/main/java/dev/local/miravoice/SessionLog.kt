package dev.local.miravoice

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * One plain-text log file per app session (process start), in <external files>/logs, newest [KEEP] kept.
 * Lines: "yyyy-MM-dd HH:mm:ss.SSS L/tag: message". Native (llama.cpp) warnings are appended to the same file.
 * Everything is mirrored to logcat. Pull with: adb pull /sdcard/Android/data/dev.local.miravoice/files/logs
 */
object SessionLog {
    const val KEEP = 25
    private const val TAG = "MiraVoice"
    private val lock = Any()
    private val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private var file: File? = null
    private lateinit var app: Context

    val path: String? get() = file?.path

    /** Opens this session's log (once per process), prunes old ones and writes the device/app header. */
    fun start(context: Context) {
        synchronized(lock) {
            if (file != null) return
            app = context.applicationContext
            val dir = File(app.getExternalFilesDir(null), "logs").apply { mkdirs() }
            val name = "session-" + SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date()) + ".log"
            file = File(dir, name)
            prune(dir)
        }
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            e("crash", "uncaught exception in thread ${t.name}", e)
            prev?.uncaughtException(t, e)
        }
        header()
    }

    /** Keeps the newest KEEP files (names sort by time), the one just created included. */
    private fun prune(dir: File) {
        val logs = dir.listFiles { f -> f.isFile && f.name.startsWith("session-") && f.name.endsWith(".log") }
            ?.sortedBy { it.name } ?: return
        val excess = logs.size + 1 - KEEP  // +1: this session's file is created on first write
        logs.take(maxOf(0, excess)).forEach { it.delete() }
    }

    fun i(tag: String, msg: String) = write('I', tag, msg)
    fun w(tag: String, msg: String, t: Throwable? = null) = write('W', tag, msg, t)
    fun e(tag: String, msg: String, t: Throwable? = null) = write('E', tag, msg, t)

    private fun write(level: Char, tag: String, msg: String, t: Throwable? = null) {
        val full = if (t == null) msg else msg + "\n" + StringWriter().also { t.printStackTrace(PrintWriter(it)) }
        when (level) {
            'E' -> Log.e(TAG, "$tag: $full")
            'W' -> Log.w(TAG, "$tag: $full")
            else -> Log.i(TAG, "$tag: $full")
        }
        val f = file ?: return
        synchronized(lock) {
            runCatching {
                f.appendText(full.lines().joinToString("\n") { "${stamp.format(Date())} $level/$tag: $it" } + "\n")
            }
        }
    }

    /** Memory and temperature, for logging around heavy work. */
    fun vitals(): String {
        val am = app.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val rt = Runtime.getRuntime()
        val pm = app.getSystemService(Context.POWER_SERVICE) as PowerManager
        val thermal = when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "?"
        }
        return "ram avail %.2f/%.2f GB%s · java heap %d/%d MB · native heap %d MB · thermal %s".format(
            mi.availMem / 1e9, mi.totalMem / 1e9, if (mi.lowMemory) " (LOW)" else "",
            (rt.totalMemory() - rt.freeMemory()) / 1_000_000, rt.maxMemory() / 1_000_000,
            android.os.Debug.getNativeHeapAllocatedSize() / 1_000_000, thermal)
    }

    private fun header() {
        val pi = app.packageManager.getPackageInfo(app.packageName, 0)
        i("session", "Mira Voice ${pi.versionName} (code ${pi.longVersionCode}, ${BuildConfig.BUILD_TYPE})")
        val soc = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
        i("session", "device ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE}), Android ${Build.VERSION.RELEASE} " +
            "(SDK ${Build.VERSION.SDK_INT}), SoC $soc, ABI ${Build.SUPPORTED_ABIS.joinToString()}")
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).map { c ->
            runCatching { File("/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq").readText().trim().toLong() / 1000 }
                .getOrDefault(0L)
        }
        i("session", "cpu ${freqs.size} cores, max MHz ${freqs.joinToString(",")}")
        i("session", vitals())
        val dir = ModelFiles.dir(app)
        val st = StatFs(dir.path)
        i("session", "storage free %.1f GB; models: %s".format(st.availableBytes / 1e9,
            dir.listFiles()?.sortedBy { it.name }?.joinToString { "${it.name} ${it.length()}" } ?: "none"))
        i("session", "log $path (keeping the last $KEEP sessions)")
    }
}
