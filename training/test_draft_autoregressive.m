// test_draft_autoregressive.m
// Bench complete draft-core model and run 8-token loopback on ANE.
#import <Foundation/Foundation.h>
#include "../bridge/ane_bridge.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K_HIDDEN 768u
#define K_VOCAB 32000u

static double now_ms(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000.0;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static uint32_t argmax_logits(const float *logits, uint32_t n) {
    uint32_t bestIdx = 0;
    float best = -INFINITY;
    for (uint32_t i = 0; i < n; i++) {
        float v = logits[i];
        if (!isfinite((double)v)) {
            continue;
        }
        if (v > best) {
            best = v;
            bestIdx = i;
        }
    }
    return bestIdx;
}

static ANEClientHandle *open_client_with_key_probe(const char *modelPath, size_t inBytes, size_t outBytes, const char **outKey) {
    static const char *keys[] = {"main", "s", "default"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        ANEClientHandle *h = ane_bridge_client_open(modelPath, keys[i], inBytes, outBytes);
        if (h) {
            if (outKey) {
                *outKey = keys[i];
            }
            return h;
        }
    }
    return NULL;
}

static float *load_embedding_table(const char *path, uint32_t vocab, uint32_t hidden) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    size_t count = (size_t)vocab * (size_t)hidden;
    float *tbl = (float *)malloc(count * sizeof(float));
    if (!tbl) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(tbl, sizeof(float), count, f);
    fclose(f);
    if (n != count) {
        free(tbl);
        return NULL;
    }
    return tbl;
}

typedef struct {
    const float *emb;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t *tokens;
    int token_cap;
    int token_n;
} LoopCtx;

static void loop_token_callback(const float *logits, uint32_t logit_count,
                                float *next_input, uint32_t input_count,
                                void *ctxp) {
    LoopCtx *ctx = (LoopCtx *)ctxp;
    uint32_t tok = argmax_logits(logits, logit_count);
    if (ctx && ctx->tokens && ctx->token_n < ctx->token_cap) {
        ctx->tokens[ctx->token_n++] = tok;
    }
    if (!ctx || !ctx->emb || input_count != ctx->hidden || tok >= ctx->vocab) {
        memset(next_input, 0, (size_t)input_count * sizeof(float));
        return;
    }
    const float *row = ctx->emb + ((size_t)tok * (size_t)ctx->hidden);
    memcpy(next_input, row, (size_t)input_count * sizeof(float));
}

static int bench_single_token_latency(ANEClientHandle *h, const float *emb) {
    const int iters = 100;
    const int warmup = 8;

    float in[K_HIDDEN];
    float logits[K_VOCAB];
    memcpy(in, emb, sizeof(in));

    for (int i = 0; i < warmup; i++) {
        ane_bridge_client_write_input(h, in, (int)K_HIDDEN);
        if (!ane_bridge_client_eval(h)) {
            printf("FAIL: warmup eval at iter=%d\n", i);
            return 1;
        }
        ane_bridge_client_read_output(h, logits, (int)K_VOCAB);
        uint32_t tok = argmax_logits(logits, K_VOCAB);
        memcpy(in, emb + ((size_t)tok * (size_t)K_HIDDEN), sizeof(in));
    }

    double *samples = (double *)calloc((size_t)iters, sizeof(double));
    if (!samples) {
        printf("FAIL: alloc samples\n");
        return 1;
    }

    memcpy(in, emb, sizeof(in));
    for (int i = 0; i < iters; i++) {
        ane_bridge_client_write_input(h, in, (int)K_HIDDEN);
        double t0 = now_ms();
        bool ok = ane_bridge_client_eval(h);
        double t1 = now_ms();
        if (!ok) {
            printf("FAIL: eval at iter=%d\n", i);
            free(samples);
            return 1;
        }
        samples[i] = t1 - t0;
        ane_bridge_client_read_output(h, logits, (int)K_VOCAB);
        uint32_t tok = argmax_logits(logits, K_VOCAB);
        memcpy(in, emb + ((size_t)tok * (size_t)K_HIDDEN), sizeof(in));
    }

    qsort(samples, (size_t)iters, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int i = 0; i < iters; i++) {
        sum += samples[i];
    }
    double avg = sum / (double)iters;
    double p50 = samples[(int)(0.50 * (iters - 1))];
    double p95 = samples[(int)(0.95 * (iters - 1))];
    double p99 = samples[(int)(0.99 * (iters - 1))];

    printf("latency_single_token iters=%d avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f\n",
           iters, avg, p50, p95, p99);
    free(samples);
    return 0;
}

static int run_loopback_test(ANEClientHandle *h, const float *emb) {
    float init[K_HIDDEN];
    float logits[K_VOCAB];
    memcpy(init, emb, sizeof(init));

    uint32_t toks[8] = {0};
    LoopCtx ctx = {
        .emb = emb,
        .vocab = K_VOCAB,
        .hidden = K_HIDDEN,
        .tokens = toks,
        .token_cap = 8,
        .token_n = 0,
    };

    double t0 = now_ms();
    int rc = ane_bridge_eval_loopback(h, init, K_HIDDEN, logits, K_VOCAB, 8, loop_token_callback, &ctx);
    double t1 = now_ms();
    if (rc != 0) {
        printf("FAIL: ane_bridge_eval_loopback rc=%d\n", rc);
        return 1;
    }

    printf("loopback tokens=8 total_ms=%.3f per_token_ms=%.3f seq=[", t1 - t0, (t1 - t0) / 8.0);
    for (int i = 0; i < ctx.token_n; i++) {
        printf("%u%s", toks[i], (i + 1 < ctx.token_n) ? "," : "");
    }
    printf("]\n");
    return 0;
}

int main(void) {
    setbuf(stdout, NULL);

    if (ane_bridge_init() != 0) {
        printf("FAIL: ane_bridge_init\n");
        return 1;
    }

    const char *modelPath = getenv("ANE_DRAFT_MODEL_PATH");
    if (!modelPath || !modelPath[0]) {
        modelPath = "/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_full_core.mlmodelc";
    }
    const char *embPath = getenv("ANE_DRAFT_EMB_PATH");
    if (!embPath || !embPath[0]) {
        embPath = "/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_full_embed_f32.bin";
    }

    float *emb = load_embedding_table(embPath, K_VOCAB, K_HIDDEN);
    if (!emb) {
        printf("FAIL: load embedding table %s\n", embPath);
        return 1;
    }

    const char *key = NULL;
    ANEClientHandle *h = open_client_with_key_probe(modelPath, (size_t)K_HIDDEN * sizeof(float), (size_t)K_VOCAB * sizeof(float), &key);
    if (!h) {
        printf("FAIL: open client %s\n", modelPath);
        free(emb);
        return 1;
    }
    printf("model=%s key=%s\n", modelPath, key ? key : "?");

    if (bench_single_token_latency(h, emb) != 0) {
        free(emb);
        ane_bridge_client_close(h);
        return 1;
    }
    if (run_loopback_test(h, emb) != 0) {
        free(emb);
        ane_bridge_client_close(h);
        return 1;
    }

    free(emb);
    ane_bridge_client_close(h);
    printf("PASS: draft autoregressive bench\n");
    return 0;
}
