package dev.local.miravoice

import android.app.Application

/** Starts the session log before anything else runs, so startup problems are recorded too. */
class MiraApp : Application() {
    override fun onCreate() {
        super.onCreate()
        SessionLog.start(this)
        MiraNative.init(applicationInfo.nativeLibraryDir)
        SessionLog.path?.let { MiraNative.setLogFile(it) }
        SessionLog.i("session", "native: ${MiraNative.systemInfo().trim()}")
    }
}
