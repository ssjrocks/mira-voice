// Desktop test for the speech recognition: prints the transcript of a 16 kHz mono 16-bit WAV.
// usage: mira_asr_cli MODEL.bin THREADS FILE.wav
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "mira_asr.h"

static std::vector<float> read_wav16(const char *path) {
    std::vector<float> out;
    FILE *f = std::fopen(path, "rb");
    if (!f) return out;
    char hdr[12];
    if (std::fread(hdr, 1, 12, f) != 12 || std::memcmp(hdr, "RIFF", 4) || std::memcmp(hdr + 8, "WAVE", 4)) {
        std::fclose(f);
        return out;
    }
    char id[4];
    uint32_t size;
    while (std::fread(id, 1, 4, f) == 4 && std::fread(&size, 4, 1, f) == 1) {
        if (std::memcmp(id, "data", 4) == 0) {
            std::vector<int16_t> s(size / 2);
            size_t n = std::fread(s.data(), 2, s.size(), f);
            out.resize(n);
            for (size_t i = 0; i < n; ++i) out[i] = s[i] / 32768.0f;
            break;
        }
        std::fseek(f, size, SEEK_CUR);
    }
    std::fclose(f);
    return out;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL.bin THREADS FILE.wav (16 kHz mono 16-bit)\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    std::vector<float> pcm = read_wav16(argv[3]);
    if (pcm.empty()) {
        fprintf(stderr, "could not read %s\n", argv[3]);
        return 1;
    }
    std::string err;
    MiraAsr *a = MiraAsr::load(argv[1], err);
    if (!a) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    double ms = 0;
    std::string text = a->transcribe(pcm.data(), (int)pcm.size(), std::atoi(argv[2]), &ms);
    printf("%.2f s audio in %.0f ms: %s\n", pcm.size() / 16000.0, ms, text.c_str());
    delete a;
    return 0;
}
