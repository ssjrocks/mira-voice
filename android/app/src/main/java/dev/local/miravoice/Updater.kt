package dev.local.miravoice

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.widget.Toast
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** Checks the GitHub releases page for a newer version and installs it (the user confirms the install). */
object Updater {
    const val REPO = "ssjrocks/mira-voice"

    data class Release(val version: String, val notes: String, val apkUrl: String, val apkSize: Long, val pageUrl: String)

    /** The latest published release, or null if there is none with an APK. Network call: not on the UI thread. */
    fun latest(): Release? {
        val conn = (URL("https://api.github.com/repos/$REPO/releases/latest").openConnection() as HttpURLConnection).apply {
            setRequestProperty("Accept", "application/vnd.github+json")
            connectTimeout = 15_000
            readTimeout = 15_000
        }
        if (conn.responseCode == 404) return null
        if (conn.responseCode != 200) throw IOException("GitHub answered HTTP ${conn.responseCode}")
        val j = JSONObject(conn.inputStream.bufferedReader().readText())
        val assets = j.getJSONArray("assets")
        val apk = (0 until assets.length()).map { assets.getJSONObject(it) }
            .firstOrNull { it.getString("name").endsWith(".apk") } ?: return null
        return Release(j.getString("tag_name").removePrefix("v"), j.optString("body"),
            apk.getString("browser_download_url"), apk.getLong("size"), j.optString("html_url"))
    }

    /** True if dotted version [remote] is newer than [local] (e.g. 0.3.0 > 0.2.1). */
    fun isNewer(remote: String, local: String): Boolean {
        val r = remote.split(".", "-").map { it.toIntOrNull() ?: 0 }
        val l = local.split(".", "-").map { it.toIntOrNull() ?: 0 }
        for (i in 0 until maxOf(r.size, l.size)) {
            val a = r.getOrElse(i) { 0 }
            val b = l.getOrElse(i) { 0 }
            if (a != b) return a > b
        }
        return false
    }

    fun downloadApk(context: Context, rel: Release, onProgress: (Long, Long) -> Unit): File {
        val f = File(context.cacheDir, "update-${rel.version}.apk")
        val conn = (URL(rel.apkUrl).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = 20_000
            readTimeout = 60_000
        }
        if (conn.responseCode != 200) throw IOException("APK download failed: HTTP ${conn.responseCode}")
        var done = 0L
        conn.inputStream.use { inp ->
            f.outputStream().use { out ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = inp.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    done += n
                    onProgress(done, rel.apkSize)
                }
            }
        }
        if (rel.apkSize > 0 && f.length() != rel.apkSize) throw IOException("APK download incomplete")
        return f
    }

    /** Hands the APK to the system installer; Android then asks the user to confirm. */
    fun install(context: Context, apk: File) {
        val pi = context.packageManager.packageInstaller
        val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
            setAppPackageName(context.packageName)
        }
        val id = pi.createSession(params)
        pi.openSession(id).use { s ->
            s.openWrite("base.apk", 0, apk.length()).use { out ->
                apk.inputStream().use { it.copyTo(out, 1 shl 16) }
                s.fsync(out)
            }
            val intent = Intent(context, InstallResultReceiver::class.java)
            val pending = PendingIntent.getBroadcast(context, id, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE)
            s.commit(pending.intentSender)
        }
        SessionLog.i("update", "install session $id committed")
    }
}

/** Receives the installer's status: shows the confirmation screen, or reports a failure. */
class InstallResultReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (val status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE)) {
            PackageInstaller.STATUS_PENDING_USER_ACTION -> {
                SessionLog.i("update", "asking the user to confirm the install")
                @Suppress("DEPRECATION")
                val confirm = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT) ?: return
                context.startActivity(confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            }
            PackageInstaller.STATUS_SUCCESS -> SessionLog.i("update", "update installed")
            else -> {
                val msg = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE) ?: "status $status"
                SessionLog.w("update", "install failed (status $status): $msg")
                Toast.makeText(context, "Update not installed: $msg", Toast.LENGTH_LONG).show()
            }
        }
    }
}
