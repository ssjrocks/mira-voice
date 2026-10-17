package dev.local.miravoice

import android.app.Application
import com.ssjrocks.appreports.AppReports
import java.io.File

/** Starts the session log before anything else runs, so startup problems are recorded too. */
class MiraApp : Application() {
    override fun onCreate() {
        super.onCreate()
        SessionLog.start(this)
        MiraNative.init(applicationInfo.nativeLibraryDir)
        SessionLog.path?.let { MiraNative.setLogFile(it) }
        SessionLog.i("session", "native: ${MiraNative.systemInfo().trim()}")
        // Crash/problem reports (shared AppReports) carry this session's log, and the previous one:
        // a native crash or ANR is only reported on the next launch, after the session that died.
        AppReports.attach("session log") { SessionLog.path?.let { File(it) }?.takeIf { it.exists() }?.readText() }
        AppReports.attach("previous session log") {
            val current = SessionLog.path?.let { File(it) }
            current?.parentFile?.listFiles { f -> f.name.startsWith("session-") && f.name.endsWith(".log") }
                ?.filter { it.name < current.name }?.maxByOrNull { it.name }
                ?.let { "${it.name}\n${it.readText()}" }
        }
    }
}
