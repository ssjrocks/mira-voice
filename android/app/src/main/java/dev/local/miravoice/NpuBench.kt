package dev.local.miravoice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max

/**
 * Experimental (npu branch): times a fixed-shape audio model on the CPU or on Qualcomm's NPU (QNN HTP, fp16) and
 * checks it against the PC's output. A graph compiled on the PC (export/qnn_context.py: <model>_ctx.onnx and its
 * _qnn.bin in the models folder) is used when present; otherwise the phone compiles it once and caches it.
 * adb: --es bench_model vocoder_npu.onnx --es bench_input features --es bench_shape 1,1024,412 --es bench_accel npu|cpu
 * (test vectors <model>_in.f32 / <model>_out.f32 in the models folder)
 */
object NpuBench {
    fun qnnOptions(ctxPath: String, threads: Int, generate: Boolean) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        if (generate) {
            addConfigEntry("ep.context_enable", "1")
            addConfigEntry("ep.context_file_path", ctxPath)
        }
        addQnn(mapOf(
            "backend_type" to "htp",
            "htp_performance_mode" to "burst",
            "enable_htp_fp16_precision" to "1",
            "htp_graph_finalization_optimization_mode" to "3",
        ))
    }

    private fun readF32(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }

    fun run(context: Context, model: String, input: String, shape: LongArray, accel: String, threads: Int, iters: Int): JSONObject {
        val dir = ModelFiles.dir(context)
        val env = OrtEnvironment.getEnvironment()
        val stem = model.removeSuffix(".onnx")
        val x = readF32(File(dir, "${stem}_in.f32"))
        val expected = readF32(File(dir, "${stem}_out.f32"))
        val res = JSONObject().put("model", model).put("accel", accel).put("threads", threads)
        val pcCtx = File(dir, "${stem}_ctx.onnx")
        val ctx = if (pcCtx.isFile) pcCtx else File(context.cacheDir, "${stem}_ctx.onnx")
        val t0 = SystemClock.elapsedRealtime()
        val session = try {
            when (accel) {
                "cpu" -> env.createSession(File(dir, model).path, OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) })
                "npu" -> if (ctx.isFile) {
                    res.put("cached", true).put("compiled_on", if (ctx == pcCtx) "pc" else "phone")
                    env.createSession(ctx.path, qnnOptions(ctx.path, threads, generate = false))
                } else {
                    res.put("cached", false)
                    env.createSession(File(dir, model).path, qnnOptions(ctx.path, threads, generate = true))
                }
                else -> throw IllegalArgumentException("accel must be cpu or npu")
            }
        } catch (e: Throwable) {
            SessionLog.e("npu", "session for $model on $accel failed", e)
            return res.put("error", "create: ${e.message?.take(300)}")
        }
        res.put("create_ms", SystemClock.elapsedRealtime() - t0)
        session.use { s ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(x), shape).use { t ->
                var out = FloatArray(0)
                val times = ArrayList<Double>()
                repeat(iters + 1) { i ->
                    val a = SystemClock.elapsedRealtimeNanos()
                    s.run(mapOf(input to t)).use { r ->
                        val b = (r[0] as OnnxTensor).floatBuffer
                        out = FloatArray(b.remaining()).also { b.get(it) }
                    }
                    val ms = (SystemClock.elapsedRealtimeNanos() - a) / 1e6
                    if (i == 0) res.put("first_ms", ms) else times.add(ms)
                }
                var err = 0.0
                var sig = 0.0
                var maxDiff = 0f
                for (k in expected.indices) {
                    val d = out[k] - expected[k]
                    err += d * d
                    sig += expected[k] * expected[k]
                    maxDiff = max(maxDiff, abs(d))
                }
                res.put("mean_ms", times.average()).put("min_ms", times.min())
                    .put("snr_db", 10 * log10(sig / max(err, 1e-30))).put("max_diff", maxDiff.toDouble())
            }
        }
        SessionLog.i("npu", "bench $res")
        return res
    }
}
