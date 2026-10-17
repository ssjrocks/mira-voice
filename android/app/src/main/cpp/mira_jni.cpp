// JNI bridge: dev.local.miravoice.MiraNative <-> MiraEngine (llama.cpp).
#include <android/log.h>
#include <jni.h>

#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mira_engine.h"

#define TAG "MiraNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

namespace {

void log_callback(ggml_log_level level, const char *text, void *) {
    // GGML_LOG_LEVEL_CONT continues the previous line (e.g. model-load progress dots): skip it
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR)
        __android_log_print(ANDROID_LOG_WARN, TAG, "%s", text);
}

std::string to_string(JNIEnv *env, jstring s) {
    if (!s) return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

void throw_runtime(JNIEnv *env, const std::string &msg) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), msg.c_str());
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_dev_local_miravoice_MiraNative_init(JNIEnv *env, jobject, jstring native_lib_dir) {
    llama_log_set(log_callback, nullptr);
    std::string dir = to_string(env, native_lib_dir);
    LOGI("loading ggml backends from %s", dir.c_str());
    ggml_backend_load_all_from_path(dir.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_local_miravoice_MiraNative_systemInfo(JNIEnv *env, jobject) {
    return env->NewStringUTF(MiraEngine::system_info().c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_dev_local_miravoice_MiraNative_load(JNIEnv *env, jobject, jstring gguf, jstring head, jint threads) {
    std::string err;
    MiraEngine *e = MiraEngine::load(to_string(env, gguf), to_string(env, head), threads, err);
    if (!e) {
        throw_runtime(env, err);
        return 0;
    }
    return reinterpret_cast<jlong>(e);
}

extern "C" JNIEXPORT void JNICALL
Java_dev_local_miravoice_MiraNative_free(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<MiraEngine *>(handle);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_dev_local_miravoice_MiraNative_generate(JNIEnv *env, jobject, jlong handle, jstring text, jintArray global,
                                             jfloat temperature, jint top_k, jfloat top_p, jfloat min_p,
                                             jfloat repeat_penalty, jint max_tokens, jint seed, jintArray prompt_codes,
                                             jobject progress, jdoubleArray stats_out) {
    auto *e = reinterpret_cast<MiraEngine *>(handle);
    std::vector<int> g(env->GetArrayLength(global));
    env->GetIntArrayRegion(global, 0, (jsize)g.size(), g.data());

    MiraParams p;
    p.temperature = temperature;
    p.top_k = top_k;
    p.top_p = top_p;
    p.min_p = min_p;
    p.repeat_penalty = repeat_penalty;
    p.max_new_tokens = max_tokens;
    p.seed = (uint32_t)seed;

    jmethodID on_code = nullptr;
    if (progress) on_code = env->GetMethodID(env->GetObjectClass(progress), "onCode", "(II)Z");

    std::vector<int> prompt;
    if (prompt_codes) {
        prompt.resize(env->GetArrayLength(prompt_codes));
        env->GetIntArrayRegion(prompt_codes, 0, (jsize)prompt.size(), prompt.data());
    }

    MiraStats st;
    std::vector<int> codes = e->generate(to_string(env, text), g, p, st, [&](int code, int n) {
        if (!on_code) return true;
        jboolean keep = env->CallBooleanMethod(progress, on_code, (jint)code, (jint)n);
        if (env->ExceptionCheck()) return false;
        return keep == JNI_TRUE;
    }, prompt);

    if (stats_out && env->GetArrayLength(stats_out) >= 4) {
        jdouble s[4] = {(double)st.prompt_tokens, (double)st.generated_tokens, st.prefill_ms, st.generate_ms};
        env->SetDoubleArrayRegion(stats_out, 0, 4, s);
    }
    jintArray out = env->NewIntArray((jsize)codes.size());
    env->SetIntArrayRegion(out, 0, (jsize)codes.size(), codes.data());
    return out;
}
