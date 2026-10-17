#include "mira_asr.h"

#include <chrono>
#include <cstring>

#include "whisper.h"

MiraAsr *MiraAsr::load(const std::string &model_path, std::string &err) {
    auto cp = whisper_context_default_params();
    cp.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(model_path.c_str(), cp);
    if (!ctx) {
        err = "failed to load speech recognition model: " + model_path;
        return nullptr;
    }
    auto *a = new MiraAsr();
    a->ctx_ = ctx;
    return a;
}

MiraAsr::~MiraAsr() {
    if (ctx_) whisper_free(ctx_);
}

std::string MiraAsr::transcribe(const float *pcm, int n_samples, int n_threads, double *ms) {
    auto t0 = std::chrono::steady_clock::now();
    auto p = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    p.beam_search.beam_size = 3;
    p.n_threads = n_threads;
    p.language = "en";
    p.translate = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_special = false;
    p.print_timestamps = false;
    p.suppress_nst = true;  // no "[Music]" and similar
    p.no_context = true;
    std::string text;
    if (whisper_full(ctx_, p, pcm, n_samples) == 0) {
        for (int i = 0; i < whisper_full_n_segments(ctx_); ++i) {
            const char *s = whisper_full_get_segment_text(ctx_, i);
            if (!s) continue;
            while (*s == ' ') ++s;
            if (!*s) continue;
            if (!text.empty()) text += ' ';
            text += s;
        }
    }
    while (!text.empty() && text.back() == ' ') text.pop_back();
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return text;
}
