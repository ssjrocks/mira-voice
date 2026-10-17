#include "mira_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <cstring>

#include "llama.h"

namespace {

// Token id layout of the MiraTTS vocabulary (added_tokens.json): contiguous blocks.
constexpr int32_t kContextTokenBase = 151665;  // <|context_token_0|> .. 4095
constexpr int32_t kSpeechTokenBase = 155761;   // <|speech_token_0|> .. 8191
constexpr int32_t kNumSpeechTokens = 8192;
constexpr int32_t kImEnd = 151645;             // eos
constexpr int32_t kTaskTts = 165137;
constexpr int32_t kStartText = 165146;
constexpr int32_t kEndText = 165152;
constexpr int32_t kContextAudioStart = 165150;
constexpr int32_t kContextAudioEnd = 165156;
constexpr int32_t kPromptSpeechStart = 165151;

// Candidate index i < kNumSpeechTokens is speech code i; index kNumSpeechTokens is eos.
constexpr int kNumCandidates = kNumSpeechTokens + 1;

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// HF-style sampling restricted to speech tokens + eos:
// repetition penalty -> temperature -> top-k -> top-p -> min-p -> sample.
class Sampler {
public:
    explicit Sampler(const MiraParams &p) : p_(p), rng_(p.seed), seen_(kNumCandidates, 0) {}

    // tokens already in the context count for the repetition penalty (as in HF transformers)
    void mark_seen(int idx) { seen_[idx] = 1; }

    // logits: kNumCandidates entries (speech codes, then eos)
    int sample(const float *logits) {
        cand_.clear();
        for (int i = 0; i < kNumCandidates; ++i) {
            float l = logits[i];
            if (seen_[i]) l = l > 0 ? l / p_.repeat_penalty : l * p_.repeat_penalty;
            cand_.push_back({i, l / p_.temperature});
        }
        size_t k = std::min<size_t>(std::max(p_.top_k, 1), cand_.size());
        std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(),
                          [](const Cand &a, const Cand &b) { return a.logit > b.logit; });
        cand_.resize(k);
        // softmax over the top-k
        float mx = cand_[0].logit, sum = 0;
        for (auto &c : cand_) sum += (c.p = std::exp(c.logit - mx));
        for (auto &c : cand_) c.p /= sum;
        // top-p: smallest prefix with cumulative prob >= top_p
        float cum = 0;
        size_t n = cand_.size();
        for (size_t i = 0; i < cand_.size(); ++i) {
            cum += cand_[i].p;
            if (cum >= p_.top_p) { n = i + 1; break; }
        }
        cand_.resize(n);
        // min-p relative to the most likely token
        float thresh = p_.min_p * cand_[0].p;
        while (cand_.size() > 1 && cand_.back().p < thresh) cand_.pop_back();
        // sample from the renormalised remainder
        float total = 0;
        for (auto &c : cand_) total += c.p;
        float r = std::uniform_real_distribution<float>(0.f, total)(rng_);
        int chosen = cand_.back().idx;
        for (auto &c : cand_) {
            if ((r -= c.p) <= 0) { chosen = c.idx; break; }
        }
        seen_[chosen] = 1;
        return chosen;
    }

private:
    struct Cand { int idx; float logit; float p; };
    MiraParams p_;
    std::mt19937 rng_;
    std::vector<uint8_t> seen_;
    std::vector<Cand> cand_;
};

}  // namespace

std::string MiraEngine::system_info() { return llama_print_system_info(); }

// fp16 -> fp32 without relying on compiler half-float support
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

MiraEngine *MiraEngine::load(const std::string &gguf_path, const std::string &head_path, int n_threads,
                             std::string &err) {
    static bool backend_ready = false;
    if (!backend_ready) {
        llama_backend_init();
        backend_ready = true;
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
#if defined(__aarch64__)
    // On ARM, llama.cpp's repacked / KleidiAI weight layouts give wrong hidden states for this model
    // (checked on a Snapdragon 8 Gen 2: first-token logits diverge from x86 and speech degrades into noise),
    // while the plain kernels match x86. Plain Q8_0 with i8mm is also the fastest option measured there.
    mp.use_extra_bufts = false;
#endif
    if (const char *v = std::getenv("MIRA_EXTRA_BUFTS")) mp.use_extra_bufts = v[0] == '1';
    llama_model *model = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!model) {
        err = "failed to load model: " + gguf_path;
        return nullptr;
    }
    std::vector<float> head;
    const int n_embd = llama_model_n_embd(model);
    if (!head_path.empty()) {
        FILE *f = std::fopen(head_path.c_str(), "rb");
        std::vector<uint16_t> h16((size_t)kNumCandidates * n_embd);
        size_t got = f ? std::fread(h16.data(), 2, h16.size(), f) : 0;
        if (f) std::fclose(f);
        if (got != h16.size()) {
            llama_model_free(model);
            err = "failed to read LM head file: " + head_path;
            return nullptr;
        }
        head.resize(h16.size());
        for (size_t i = 0; i < h16.size(); ++i) head[i] = half_to_float(h16[i]);
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 3072;  // prompt (~40 + text + up to 1000 voice-clip codes = 20 s in accent mode) + up to 1024 speech tokens
    cp.n_batch = 1536;
    cp.n_ubatch = 1536;  // embeddings mode needs the whole prompt in one ubatch
    if (!head.empty()) {
        // return the final (normed) hidden state of the last token; this drops the LM head from the graph
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
    }
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    if (const char *v = std::getenv("MIRA_THREADS_BATCH")) cp.n_threads_batch = std::atoi(v);
    cp.no_perf = true;
    // Qwen2's attention activations overflow fp16: CPUs with native fp16 arithmetic (most phones) produce
    // NaN with an f16 KV cache or flash attention, so keep attention in f32 (the cache is only ~50 MB).
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        err = "failed to create context";
        return nullptr;
    }
    auto *e = new MiraEngine();
    e->model_ = model;
    e->ctx_ = ctx;
    e->vocab_ = llama_model_get_vocab(model);
    e->n_ctx_ = (int)llama_n_ctx(ctx);
    e->n_embd_ = n_embd;
    e->head_ = std::move(head);
    e->logits_.resize(kNumCandidates);
    if (llama_vocab_n_tokens(e->vocab_) <= kTaskTts) {
        err = "model vocabulary is not MiraTTS's";
        delete e;
        return nullptr;
    }
    return e;
}

MiraEngine::~MiraEngine() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

const float *MiraEngine::candidate_logits() {
    if (head_.empty()) {
        const float *all = llama_get_logits_ith(ctx_, -1);
        for (int i = 0; i < kNumSpeechTokens; ++i) logits_[i] = all[kSpeechTokenBase + i];
        logits_[kNumSpeechTokens] = all[kImEnd];
    } else {
        const float *h = llama_get_embeddings_seq(ctx_, 0);
        const int n = n_embd_;
        for (int r = 0; r < kNumCandidates; ++r) {
            const float *w = &head_[(size_t)r * n];
            float acc = 0;
            for (int j = 0; j < n; ++j) acc += w[j] * h[j];
            logits_[r] = acc;
        }
    }
    return logits_.data();
}

std::vector<int32_t> MiraEngine::build_prompt(const std::string &text, const std::vector<int> &global_tokens,
                                             const std::vector<int> &prompt_codes) {
    std::vector<int32_t> ids = {kTaskTts, kStartText};
    // Text is tokenized as plain text (no special-token parsing), like the HF tokenizer does between tags.
    std::vector<llama_token> t(text.size() + 16);
    int n = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(), t.data(), (int32_t)t.size(), false, false);
    if (n < 0) {
        t.resize(-n);
        n = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(), t.data(), (int32_t)t.size(), false, false);
    }
    ids.insert(ids.end(), t.begin(), t.begin() + std::max(n, 0));
    ids.push_back(kEndText);
    ids.push_back(kContextAudioStart);
    for (int g : global_tokens) ids.push_back(kContextTokenBase + g);
    ids.push_back(kContextAudioEnd);
    ids.push_back(kPromptSpeechStart);
    for (int c : prompt_codes) ids.push_back(kSpeechTokenBase + c);
    return ids;
}

std::vector<int> MiraEngine::generate(const std::string &text, const std::vector<int> &global_tokens,
                                      const MiraParams &p, MiraStats &stats,
                                      const std::function<bool(int, int)> &on_code,
                                      const std::vector<int> &prompt_codes) {
    std::vector<int> codes;
    stats = MiraStats();
    llama_memory_clear(llama_get_memory(ctx_), true);

    std::vector<int32_t> prompt = build_prompt(text, global_tokens, prompt_codes);
    stats.prompt_tokens = (int)prompt.size();
    int max_new = std::min(p.max_new_tokens, n_ctx_ - (int)prompt.size() - 1);
    if (max_new <= 0) return codes;

    auto t0 = std::chrono::steady_clock::now();
    if (llama_decode(ctx_, llama_batch_get_one(prompt.data(), (int32_t)prompt.size())) != 0) return codes;
    stats.prefill_ms = ms_since(t0);

    Sampler sampler(p);
    // HF transformers would also penalise the voice clip's codes (they are in the input), but that measured worse:
    // word errors 6% -> 1% without it, same speaker similarity. MIRA_PROMPT_PENALTY=1 restores it for comparison.
    if (const char *pp = std::getenv("MIRA_PROMPT_PENALTY"); pp && pp[0] == '1')
        for (int c : prompt_codes) sampler.mark_seen(c);
    const char *dbg_env = std::getenv("MIRA_DEBUG");
    const int debug_steps = dbg_env ? std::atoi(dbg_env) : 0;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new; ++i) {
        const float *lg = candidate_logits();
        if (i < debug_steps) {  // per-step diagnostics on stderr
            int am = 0;
            int nan = 0;
            for (int r = 0; r < kNumCandidates; ++r) {
                if (std::isnan(lg[r])) nan++;
                else if (lg[r] > lg[am]) am = r;
            }
            double hn = 0;
            if (!head_.empty()) {
                const float *h = llama_get_embeddings_seq(ctx_, 0);
                for (int j = 0; j < n_embd_; ++j) hn += (double)h[j] * h[j];
            }
            fprintf(stderr, "step %3d argmax %4d max %8.3f eos %8.3f nan %d |h| %.3f\n", i, am, lg[am],
                    lg[kNumSpeechTokens], nan, std::sqrt(hn));
        }
        int c = sampler.sample(lg);
        stats.generated_tokens++;
        if (c == kNumSpeechTokens) break;  // eos
        codes.push_back(c);
        if (on_code && !on_code(c, (int)codes.size())) break;
        llama_token tok = kSpeechTokenBase + c;
        if (llama_decode(ctx_, llama_batch_get_one(&tok, 1)) != 0) break;
    }
    stats.generate_ms = ms_since(t1);
    return codes;
}
