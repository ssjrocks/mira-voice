package dev.local.miravoice

/** JNI bridge to the llama.cpp-based MiraTTS language model (native/mira_engine.cpp). */
object MiraNative {
    fun interface Progress {
        /** Called with every speech code as it is generated; return false to cancel. */
        fun onCode(code: Int, codesSoFar: Int): Boolean
    }

    init {
        System.loadLibrary("miravoice")
    }

    external fun init(nativeLibDir: String)
    external fun systemInfo(): String

    /** Returns a handle; throws RuntimeException if the model can't be loaded. */
    external fun load(ggufPath: String, headPath: String, threads: Int): Long
    external fun free(handle: Long)

    /**
     * Speech codes (0..8191). stats receives [promptTokens, generatedTokens, prefillMs, generateMs].
     * promptCodes: speech codes of the voice clip whose transcript starts [text] (accent mode), or null.
     */
    external fun generate(
        handle: Long, text: String, globalTokens: IntArray,
        temperature: Float, topK: Int, topP: Float, minP: Float, repeatPenalty: Float,
        maxTokens: Int, seed: Int, promptCodes: IntArray?, progress: Progress?, stats: DoubleArray,
    ): IntArray
}
