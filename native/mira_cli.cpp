// Desktop test for the engine: prints speech codes as JSON.
// usage: mira_cli MODEL.gguf HEAD.bin|- THREADS SEED "text" "g0,g1,...,g31"
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "ggml-backend.h"
#include "mira_engine.h"

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL.gguf HEAD.bin|- THREADS SEED TEXT GLOBAL_TOKENS_CSV\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();  // no-op for static builds; loads CPU variants when built with GGML_BACKEND_DL
    std::vector<int> global;
    std::stringstream ss(argv[6]);
    for (std::string s; std::getline(ss, s, ',');) global.push_back(std::atoi(s.c_str()));

    std::string err;
    std::string head = argv[2];
    MiraEngine *e = MiraEngine::load(argv[1], head == "-" ? "" : head, std::atoi(argv[3]), err);
    if (!e) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    MiraParams p;
    p.seed = (uint32_t)std::atoi(argv[4]);
    if (const char *v = std::getenv("MIRA_GREEDY"); v && v[0] == '1') p.top_k = 1;
    if (const char *v = std::getenv("MIRA_MAX")) p.max_new_tokens = std::atoi(v);
    std::vector<int> prompt_codes;  // MIRA_PROMPT_CODES=file with comma-separated reference speech codes
    if (const char *f = std::getenv("MIRA_PROMPT_CODES")) {
        if (FILE *fp = std::fopen(f, "r")) {
            for (int c; std::fscanf(fp, "%d,", &c) == 1;) prompt_codes.push_back(c);
            std::fclose(fp);
        }
    }
    MiraStats st;
    std::vector<int> codes = e->generate(argv[5], global, p, st, nullptr, prompt_codes);
    printf("{\"prompt_tokens\": %d, \"generated_tokens\": %d, \"prefill_ms\": %.1f, \"generate_ms\": %.1f, \"codes\": [",
           st.prompt_tokens, st.generated_tokens, st.prefill_ms, st.generate_ms);
    for (size_t i = 0; i < codes.size(); ++i) printf(i ? ",%d" : "%d", codes[i]);
    printf("]}\n");
    delete e;
    return 0;
}
