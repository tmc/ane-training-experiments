// test_ffn_bench.m
// Bench ANE FFN model latency across sequence lengths via bridge.
#import <Foundation/Foundation.h>
#include "../bridge/ane_bridge.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static double now_ms(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000.0;
}

static void fill_input(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        x[i] = (float)((i % 97) - 48) * 0.01f;
    }
}

static const char *try_open_key(const char *model_path, size_t bytes, ANEClientHandle **out) {
    static const char *keys[] = {"main", "s", "default"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        ANEClientHandle *h = ane_bridge_client_open(model_path, keys[i], bytes, bytes);
        if (h) {
            *out = h;
            return keys[i];
        }
    }
    return NULL;
}

static int run_seq_once(const char *model_path, uint32_t hidden, uint32_t seq, int iters, int warmup) {
    @autoreleasepool {
        if (ane_bridge_init() != 0) {
            printf("FAIL seq=%u init\n", seq);
            return 1;
        }

        uint64_t elems64 = (uint64_t)hidden * (uint64_t)seq;
        if (elems64 > UINT32_MAX) {
            printf("FAIL seq=%u element count overflow\n", seq);
            return 1;
        }
        uint32_t elems = (uint32_t)elems64;
        size_t bytes = (size_t)elems * sizeof(float);

        ANEClientHandle *h = NULL;
        const char *key = try_open_key(model_path, bytes, &h);
        if (!h) {
            printf("FAIL seq=%u open model key probe failed\n", seq);
            return 1;
        }

        float *in = (float *)calloc((size_t)elems, sizeof(float));
        float *out = (float *)calloc((size_t)elems, sizeof(float));
        if (!in || !out) {
            printf("FAIL seq=%u alloc\n", seq);
            return 1;
        }
        fill_input(in, elems);

        ane_bridge_client_write_input(h, in, (int)elems);
        for (int i = 0; i < warmup; i++) {
            if (!ane_bridge_client_eval(h)) {
                printf("FAIL seq=%u warmup_eval=%d\n", seq, i);
                return 1;
            }
        }

        double *samples = (double *)calloc((size_t)iters, sizeof(double));
        if (!samples) {
            printf("FAIL seq=%u alloc samples\n", seq);
            return 1;
        }

        for (int i = 0; i < iters; i++) {
            double t0 = now_ms();
            bool ok = ane_bridge_client_eval(h);
            double t1 = now_ms();
            if (!ok) {
                printf("FAIL seq=%u eval_iter=%d\n", seq, i);
                return 1;
            }
            samples[i] = t1 - t0;
        }

        ane_bridge_client_read_output(h, out, (int)elems);

        double sum = 0.0;
        double mn = 1e30;
        double mx = 0.0;
        for (int i = 0; i < iters; i++) {
            double v = samples[i];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double avg = sum / (double)iters;

        // crude median via partial sort enough for 5 points reporting.
        for (int i = 0; i < iters; i++) {
            for (int j = i + 1; j < iters; j++) {
                if (samples[j] < samples[i]) {
                    double tmp = samples[i];
                    samples[i] = samples[j];
                    samples[j] = tmp;
                }
            }
        }
        double p50 = samples[iters / 2];

        printf("OK model=%s hidden=%u seq=%u key=%s iters=%d avg_ms=%.3f p50_ms=%.3f min_ms=%.3f max_ms=%.3f out0=%.6f\n",
               model_path, hidden, seq, key, iters, avg, p50, mn, mx, out[0]);

        // shared-event paths are fragile on cleanup in this runtime; hard-exit child.
        _exit(0);
    }
}

static int run_model(const char *model_path, uint32_t hidden) {
    const uint32_t seqs[] = {1, 32, 128, 512, 1024};
    const int nseq = (int)(sizeof(seqs) / sizeof(seqs[0]));

    for (int i = 0; i < nseq; i++) {
        uint32_t seq = seqs[i];
        int iters = (seq <= 128) ? 20 : 8;
        int warmup = 3;

        pid_t pid = fork();
        if (pid == 0) {
            int rc = run_seq_once(model_path, hidden, seq, iters, warmup);
            _exit(rc);
        }
        if (pid < 0) {
            printf("FAIL seq=%u fork\n", seq);
            return 1;
        }

        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            printf("FAIL seq=%u waitpid\n", seq);
            return 1;
        }
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            if (WIFSIGNALED(status)) {
                printf("FAIL seq=%u child signal=%d\n", seq, WTERMSIG(status));
            } else {
                printf("FAIL seq=%u child exit=%d\n", seq, WEXITSTATUS(status));
            }
            return 1;
        }
    }

    return 0;
}

int main(void) {
    setbuf(stdout, NULL);

    const char *llama = getenv("ANE_FFN_MODEL_PATH");
    if (!llama || !llama[0]) {
        llama = "/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/llama3b_ffn.mlmodelc";
    }
    const char *draft = getenv("ANE_FFN_DRAFT_MODEL_PATH");
    if (!draft || !draft[0]) {
        draft = "/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_ffn.mlmodelc";
    }

    printf("# Benchmark Llama-3B FFN hidden=3072\n");
    if (run_model(llama, 3072) != 0) {
        return 1;
    }

    printf("# Benchmark Draft FFN hidden=768\n");
    if (run_model(draft, 768) != 0) {
        return 1;
    }

    return 0;
}
