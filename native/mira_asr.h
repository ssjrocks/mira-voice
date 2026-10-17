// Speech recognition for voice setup (whisper.cpp, sharing llama.cpp's ggml): 16 kHz audio -> transcript.
#pragma once

#include <string>

struct whisper_context;

class MiraAsr {
public:
    // Returns nullptr and sets err on failure.
    static MiraAsr *load(const std::string &model_path, std::string &err);
    ~MiraAsr();

    // English transcript of 16 kHz mono float PCM, with punctuation. ms receives the processing time.
    std::string transcribe(const float *pcm, int n_samples, int n_threads, double *ms = nullptr);

private:
    MiraAsr() = default;
    whisper_context *ctx_ = nullptr;
};
