// MiraTTS language-model stage on llama.cpp: text + 32 speaker tokens -> speech codes (0..8191).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

struct MiraParams {
    // Defaults match upstream MiraTTS (mira/model.py GenerationConfig).
    float temperature = 0.8f;
    int top_k = 50;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.2f;
    int max_new_tokens = 1024;
    uint32_t seed = 0;
};

struct MiraStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double prefill_ms = 0;
    double generate_ms = 0;
};

class MiraEngine {
public:
    // Returns nullptr and sets err on failure. head_path: mira_head.bin (fp16 [8193, 896] rows of the
    // LM head for speech tokens + eos). If given, llama.cpp skips its full 166k-entry LM head and the
    // engine scores only those rows; if empty, the full logits are used.
    static MiraEngine *load(const std::string &gguf_path, const std::string &head_path, int n_threads,
                            std::string &err);
    ~MiraEngine();

    // on_code(code, codes_so_far) is called for every speech code as it is generated (for streaming);
    // return false to cancel. prompt_codes: optional speech codes of a reference clip whose transcript
    // starts `text`; the model then continues in that speaker's style (accent, pacing).
    std::vector<int> generate(const std::string &text, const std::vector<int> &global_tokens,
                              const MiraParams &p, MiraStats &stats,
                              const std::function<bool(int, int)> &on_code = nullptr,
                              const std::vector<int> &prompt_codes = {});

    static std::string system_info();

private:
    MiraEngine() = default;
    std::vector<int32_t> build_prompt(const std::string &text, const std::vector<int> &global_tokens,
                                      const std::vector<int> &prompt_codes);

    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    int n_ctx_ = 0;
    int n_embd_ = 0;
    std::vector<float> head_;  // [kNumCandidates, n_embd_], empty = use llama.cpp logits
    std::vector<float> logits_;
    const float *candidate_logits();
};
