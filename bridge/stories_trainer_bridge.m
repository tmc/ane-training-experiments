#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ane_bridge.h"
#include "stories_trainer_bridge.h"

struct ANEStoriesTrainerHandle {
    ANEClientHandle *client;
    uint16_t *tokens;
    size_t n_tokens;
    size_t token_pos;

    uint32_t step;
    uint32_t total_steps;
    uint32_t compile_budget;
    bool ane_extras;
    float lr;
    float last_loss;

    size_t input_bytes;
    size_t output_bytes;
    uint32_t input_count;
    uint32_t output_count;
    float *input_buf;
    float *output_buf;
};

typedef struct StoriesCkptV1 {
    uint32_t magic;
    uint32_t version;
    uint32_t step;
    uint32_t total_steps;
    uint64_t token_pos;
    uint32_t compile_budget;
    uint32_t ane_extras;
    float lr;
    float last_loss;
} StoriesCkptV1;

static const uint32_t kStoriesMagic = 0x53545231; // "STR1"
static const uint32_t kStoriesVersion = 1;

static char g_stories_last_error[512];

static void stories_set_error(const char *msg) {
    if (!msg) {
        g_stories_last_error[0] = '\0';
        return;
    }
    snprintf(g_stories_last_error, sizeof(g_stories_last_error), "%s", msg);
}

static int stories_load_tokens(const char *path, uint16_t **out_tokens, size_t *out_count) {
    *out_tokens = NULL;
    *out_count = 0;
    if (!path || path[0] == '\0') {
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long size = ftell(f);
    if (size <= 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    size_t n_tokens = (size_t)size / sizeof(uint16_t);
    if (n_tokens == 0) {
        fclose(f);
        return -1;
    }
    uint16_t *tokens = (uint16_t *)calloc(n_tokens, sizeof(uint16_t));
    if (!tokens) {
        fclose(f);
        return -1;
    }
    size_t n = fread(tokens, sizeof(uint16_t), n_tokens, f);
    fclose(f);
    if (n != n_tokens) {
        free(tokens);
        return -1;
    }
    *out_tokens = tokens;
    *out_count = n_tokens;
    return 0;
}

static void stories_fill_input(struct ANEStoriesTrainerHandle *h) {
    if (!h || !h->input_buf || h->input_count == 0) {
        return;
    }
    if (!h->tokens || h->n_tokens == 0) {
        for (uint32_t i = 0; i < h->input_count; i++) {
            h->input_buf[i] = (float)((i + h->step) % 1024) / 1024.0f;
        }
        return;
    }
    size_t pos = h->token_pos;
    for (uint32_t i = 0; i < h->input_count; i++) {
        uint16_t tok = h->tokens[pos % h->n_tokens];
        h->input_buf[i] = (float)(tok % 32000u) / 32000.0f;
        pos++;
    }
    h->token_pos = pos % h->n_tokens;
}

ANEStoriesTrainerHandle *ane_bridge_stories_open(const char *model_path,
                                                 const char *model_key,
                                                 const char *data_path,
                                                 size_t input_bytes,
                                                 size_t output_bytes,
                                                 uint32_t total_steps,
                                                 float lr,
                                                 bool ane_extras,
                                                 uint32_t compile_budget) {
    @autoreleasepool {
        stories_set_error(NULL);
        if (!model_path || model_path[0] == '\0') {
            stories_set_error("stories open: model path is empty");
            return NULL;
        }
        if (input_bytes == 0 || output_bytes == 0 || (input_bytes % sizeof(float)) != 0 || (output_bytes % sizeof(float)) != 0) {
            stories_set_error("stories open: invalid input/output byte sizes");
            return NULL;
        }
        ANEClientHandle *client = ane_bridge_client_open(model_path, model_key, input_bytes, output_bytes);
        if (!client) {
            stories_set_error("stories open: ane_bridge_client_open failed");
            return NULL;
        }

        struct ANEStoriesTrainerHandle *h = (struct ANEStoriesTrainerHandle *)calloc(1, sizeof(*h));
        if (!h) {
            ane_bridge_client_close(client);
            stories_set_error("stories open: allocation failed");
            return NULL;
        }
        h->client = client;
        h->input_bytes = input_bytes;
        h->output_bytes = output_bytes;
        h->input_count = (uint32_t)(input_bytes / sizeof(float));
        h->output_count = (uint32_t)(output_bytes / sizeof(float));
        h->total_steps = total_steps;
        h->lr = lr;
        h->ane_extras = ane_extras;
        h->compile_budget = compile_budget;
        h->input_buf = (float *)calloc(h->input_count, sizeof(float));
        h->output_buf = (float *)calloc(h->output_count, sizeof(float));
        if (!h->input_buf || !h->output_buf) {
            ane_bridge_stories_close(h);
            stories_set_error("stories open: buffer allocation failed");
            return NULL;
        }
        (void)stories_load_tokens(data_path, &h->tokens, &h->n_tokens);
        ane_bridge_reset_compile_count();
        return h;
    }
}

int ane_bridge_stories_step(ANEStoriesTrainerHandle *h, ANEStoriesStepStats *stats) {
    @autoreleasepool {
        stories_set_error(NULL);
        if (!h || !h->client) {
            stories_set_error("stories step: invalid handle");
            return -1;
        }
        if (h->total_steps > 0 && h->step >= h->total_steps) {
            return 1;
        }

        double t0 = (double)CFAbsoluteTimeGetCurrent() * 1000.0;
        stories_fill_input(h);
        ane_bridge_client_write_input(h->client, h->input_buf, (int)h->input_count);
        if (!ane_bridge_client_eval(h->client)) {
            stories_set_error("stories step: ane eval failed");
            return -2;
        }
        ane_bridge_client_read_output(h->client, h->output_buf, (int)h->output_count);

        uint32_t n = h->output_count < 1024 ? h->output_count : 1024;
        double loss = 0.0;
        if (n == 0) {
            loss = 0.0;
        } else {
            for (uint32_t i = 0; i < n; i++) {
                double v = (double)h->output_buf[i];
                loss += fabs(v);
            }
            loss /= (double)n;
        }
        h->last_loss = (float)loss;
        h->step++;
        double t1 = (double)CFAbsoluteTimeGetCurrent() * 1000.0;

        if (stats) {
            stats->step = h->step;
            stats->loss = h->last_loss;
            stats->step_ms = t1 - t0;
            stats->compiles = (uint32_t)ane_bridge_get_compile_count();
            stats->restart_required = (h->compile_budget > 0 && stats->compiles >= h->compile_budget) ? 1u : 0u;
        }
        return 0;
    }
}

int ane_bridge_stories_save_checkpoint(ANEStoriesTrainerHandle *h, const char *path) {
    stories_set_error(NULL);
    if (!h || !path || path[0] == '\0') {
        stories_set_error("stories save checkpoint: invalid args");
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        stories_set_error("stories save checkpoint: fopen failed");
        return -1;
    }
    StoriesCkptV1 ckpt = {
        .magic = kStoriesMagic,
        .version = kStoriesVersion,
        .step = h->step,
        .total_steps = h->total_steps,
        .token_pos = (uint64_t)h->token_pos,
        .compile_budget = h->compile_budget,
        .ane_extras = h->ane_extras ? 1u : 0u,
        .lr = h->lr,
        .last_loss = h->last_loss,
    };
    size_t n = fwrite(&ckpt, sizeof(ckpt), 1, f);
    fclose(f);
    if (n != 1) {
        stories_set_error("stories save checkpoint: fwrite failed");
        return -1;
    }
    return 0;
}

int ane_bridge_stories_load_checkpoint(ANEStoriesTrainerHandle *h, const char *path) {
    stories_set_error(NULL);
    if (!h || !path || path[0] == '\0') {
        stories_set_error("stories load checkpoint: invalid args");
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        stories_set_error("stories load checkpoint: fopen failed");
        return -1;
    }
    StoriesCkptV1 ckpt;
    size_t n = fread(&ckpt, sizeof(ckpt), 1, f);
    fclose(f);
    if (n != 1) {
        stories_set_error("stories load checkpoint: fread failed");
        return -1;
    }
    if (ckpt.magic != kStoriesMagic || ckpt.version != kStoriesVersion) {
        stories_set_error("stories load checkpoint: bad checkpoint header");
        return -1;
    }
    h->step = ckpt.step;
    h->total_steps = ckpt.total_steps;
    h->token_pos = (size_t)ckpt.token_pos;
    h->compile_budget = ckpt.compile_budget;
    h->ane_extras = ckpt.ane_extras ? true : false;
    h->lr = ckpt.lr;
    h->last_loss = ckpt.last_loss;
    return 0;
}

void ane_bridge_stories_close(ANEStoriesTrainerHandle *h) {
    if (!h) {
        return;
    }
    if (h->client) {
        ane_bridge_client_close(h->client);
    }
    free(h->tokens);
    free(h->input_buf);
    free(h->output_buf);
    free(h);
}

int ane_bridge_stories_last_error(char *buf, size_t n) {
    if (!buf || n == 0) {
        return 0;
    }
    snprintf(buf, n, "%s", g_stories_last_error);
    return (int)strlen(buf);
}
