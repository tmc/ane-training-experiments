// test_full_draft_transformer.m
// Build and benchmark a hand-written Espresso 6-layer draft transformer (no coremltools).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const unsigned int kQoS = 21;
static const int kWarmup = 10;
static const int kIters = 100;

static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIO;
static Class CNSURL;

typedef struct {
    int vocab;
    int hidden;
    int intermediate;
    int layers;
    int heads;
} Arch;

typedef struct {
    uint64_t *sizes;
    uint64_t *offsets;
    size_t n;
    size_t cap;
} BlobPlan;

typedef struct {
    BOOL ok;
    NSString *error;
} ProbeResult;

typedef struct {
    BOOL ok;
    const char *key;
    double avg_us;
    double p50_us;
    double min_us;
    double max_us;
    float out0;
    float out1;
    float out2;
    NSString *error;
} BenchResult;

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
}

static uint64_t align256(uint64_t v) {
    return (v + 255ULL) & ~255ULL;
}

static double now_us(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000000.0;
}

static int env_int(const char *name, int fallback) {
    const char *raw = getenv(name);
    if (!raw || !raw[0]) {
        return fallback;
    }
    char *end = NULL;
    long v = strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || v <= 0 || v > INT32_MAX) {
        return fallback;
    }
    return (int)v;
}

static int env_nonneg_int(const char *name, int fallback) {
    const char *raw = getenv(name);
    if (!raw || !raw[0]) {
        return fallback;
    }
    char *end = NULL;
    long v = strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || v < 0 || v > INT32_MAX) {
        return fallback;
    }
    return (int)v;
}

static BOOL setup_ane_classes(void) {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "dlopen AppleNeuralEngine failed: %s\n", dlerror());
        return NO;
    }

    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");
    CNSURL = NSClassFromString(@"NSURL");

    if (!CClient || !CModel || !CReq || !CAIO || !CNSURL) {
        fprintf(stderr, "resolve ANE classes failed\n");
        return NO;
    }
    return YES;
}

static BOOL copy_if_exists(NSString *src, NSString *dst, NSError **err) {
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:src]) {
        return YES;
    }
    NSString *dir = [dst stringByDeletingLastPathComponent];
    if (![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:err]) {
        return NO;
    }
    [fm removeItemAtPath:dst error:nil];
    return [fm copyItemAtPath:src toPath:dst error:err];
}

static BOOL stage_template_scaffold(NSString *dstDir, NSError **err) {
    const char *raw = getenv("ANE_ESPRESSO_TEMPLATE_MODEL");
    NSString *srcDir = raw && raw[0]
        ? [NSString stringWithUTF8String:raw]
        : @"/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_full_core.mlmodelc";

    return copy_if_exists([srcDir stringByAppendingPathComponent:@"coremldata.bin"], [dstDir stringByAppendingPathComponent:@"coremldata.bin"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"metadata.json"], [dstDir stringByAppendingPathComponent:@"metadata.json"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"model.mil"], [dstDir stringByAppendingPathComponent:@"model.mil"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"analytics/coremldata.bin"], [dstDir stringByAppendingPathComponent:@"analytics/coremldata.bin"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"model/coremldata.bin"], [dstDir stringByAppendingPathComponent:@"model/coremldata.bin"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"neural_network_optionals/coremldata.bin"], [dstDir stringByAppendingPathComponent:@"neural_network_optionals/coremldata.bin"], err);
}

static void blob_plan_init(BlobPlan *p) {
    memset(p, 0, sizeof(*p));
}

static void blob_plan_free(BlobPlan *p) {
    free(p->sizes);
    free(p->offsets);
    memset(p, 0, sizeof(*p));
}

static uint64_t blob_plan_add(BlobPlan *p, uint64_t size_bytes) {
    if (p->n == p->cap) {
        size_t ncap = p->cap ? (p->cap * 2) : 32;
        uint64_t *ns = (uint64_t *)realloc(p->sizes, ncap * sizeof(uint64_t));
        uint64_t *no = (uint64_t *)realloc(p->offsets, ncap * sizeof(uint64_t));
        if (!ns || !no) {
            free(ns);
            free(no);
            return 0;
        }
        p->sizes = ns;
        p->offsets = no;
        p->cap = ncap;
    }
    p->sizes[p->n] = size_bytes;
    p->offsets[p->n] = 0;
    uint64_t id = (uint64_t)(1 + (p->n * 2)); // odd ids: 1,3,5,...
    p->n++;
    return id;
}

static NSDictionary *shape_entry_ex(int width, int height, int rank) {
    return @{
        @"k": @1,
        @"w": @(width),
        @"n": @1,
        @"_rank": @(rank),
        @"h": @(height),
    };
}

static void add_shape_ex(NSMutableDictionary *shapes, NSString *name, int width, int height, int rank) {
    if (!name || !width || !height || !rank) {
        return;
    }
    shapes[name] = shape_entry_ex(width, height, rank);
}

static void add_shape(NSMutableDictionary *shapes, NSString *name, int width) {
    add_shape_ex(shapes, name, width, 1, 3);
}

static NSDictionary *make_inner_product_layer(
    NSString *name,
    NSString *bottom,
    NSString *top,
    int nB,
    int nC,
    BOOL hasBias,
    BOOL isOutput,
    BOOL isLookup,
    BlobPlan *blobs)
{
    uint64_t biasId = 0;
    if (hasBias) {
        uint64_t biasBytes = (uint64_t)nC * 16ULL;
        biasId = blob_plan_add(blobs, biasBytes);
        if (!biasId) {
            return nil;
        }
    }

    uint64_t weightBytes = (uint64_t)nB * (uint64_t)nC * 4ULL;
    uint64_t weightId = blob_plan_add(blobs, weightBytes);
    if (!weightId) {
        return nil;
    }

    NSMutableDictionary *layer = [@{
        @"name": name,
        @"type": @"inner_product",
        @"bottom": bottom,
        @"top": top,
        @"debug_info": name,
        @"weights": @{},
        @"has_relu": @0,
        @"has_tanh": @0,
        @"has_prelu": @0,
        @"has_biases": hasBias ? @1 : @0,
        @"nB": @(nB),
        @"nC": @(nC),
        @"blob_weights": @(weightId),
    } mutableCopy];

    if (hasBias) {
        layer[@"blob_biases"] = @(biasId);
    }

    if (isLookup || isOutput) {
        NSMutableDictionary *attrs = [NSMutableDictionary dictionary];
        if (isLookup) {
            attrs[@"is_lookup"] = @1;
        }
        if (isOutput) {
            attrs[@"is_output"] = @1;
        }
        layer[@"attributes"] = attrs;
    }

    return layer;
}

static NSDictionary *make_elementwise_layer(NSString *name, NSString *bottom, NSString *top, int operation) {
    return @{
        @"name": name,
        @"type": @"elementwise",
        @"bottom": bottom,
        @"top": top,
        @"debug_info": name,
        @"weights": @{},
        @"operation": @(operation),
        @"alpha": @1,
        @"beta": @0,
        @"fused_relu": @0,
    };
}

static NSDictionary *make_activation_sigmoid_layer(NSString *name, NSString *bottom, NSString *top) {
    return @{
        @"name": name,
        @"type": @"activation",
        @"bottom": bottom,
        @"top": top,
        @"debug_info": name,
        @"weights": @{},
        @"mode": @3,
    };
}

static NSDictionary *make_reshape_layer(NSString *name, NSString *bottom, NSString *top, int dstW, int dstH, int dstRank) {
    return @{
        @"name": name,
        @"weights": @{},
        @"dst_w": @(dstW),
        @"version": @1,
        @"dst_n": @1,
        @"dst_nd_rank": @(dstRank),
        @"type": @"reshape",
        @"dst_h": @(dstH),
        @"mode": @0,
        @"dynamic_shape": @NO,
        @"bottom": bottom,
        @"debug_info": name,
        @"dst_seq": @1,
        @"dst_k": @1,
        @"top": top,
    };
}

static NSDictionary *make_elementwise_layer_alpha(NSString *name, NSString *bottom, NSString *top, int operation, double alpha) {
    return @{
        @"name": name,
        @"type": @"elementwise",
        @"bottom": bottom,
        @"top": top,
        @"debug_info": name,
        @"weights": @{},
        @"operation": @(operation),
        @"alpha": @(alpha),
        @"beta": @0,
        @"fused_relu": @0,
    };
}

static NSDictionary *make_sdpa_candidate_layer(NSString *name, NSString *bottomQKV, NSString *top, NSString *typeName) {
    return @{
        @"name": name,
        @"type": typeName,
        @"bottom": bottomQKV,
        @"top": top,
        @"debug_info": name,
        @"weights": @{},
    };
}

static NSData *build_weights_data(BlobPlan *blobs, NSString **errOut) {
    if (!blobs || blobs->n == 0) {
        if (errOut) *errOut = @"blob plan is empty";
        return nil;
    }

    // Espresso format observed in model.espresso.weights:
    // [2*N,0, offset(id1),id1, size(id1),next, offset(id3),id3, size(id3),next, ...]
    const uint64_t firstOffset = 56ULL;
    blobs->offsets[0] = firstOffset;
    for (size_t i = 1; i < blobs->n; i++) {
        blobs->offsets[i] = align256(blobs->offsets[i - 1] + blobs->sizes[i - 1]);
    }
    uint64_t totalBytes = blobs->offsets[blobs->n - 1] + blobs->sizes[blobs->n - 1];

    NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)totalBytes];
    if (!data) {
        if (errOut) *errOut = @"weights allocation failed";
        return nil;
    }

    uint8_t *buf = (uint8_t *)data.mutableBytes;
    if (!buf) {
        if (errOut) *errOut = @"weights mutable bytes unavailable";
        return nil;
    }

    size_t nWords = 2 + (blobs->n * 4);
    uint64_t *words = (uint64_t *)calloc(nWords, sizeof(uint64_t));
    if (!words) {
        if (errOut) *errOut = @"header allocation failed";
        return nil;
    }

    size_t w = 0;
    words[w++] = (uint64_t)(blobs->n * 2); // max id + 1
    words[w++] = 0;
    for (size_t i = 0; i < blobs->n; i++) {
        uint64_t oddId = 1 + (uint64_t)(i * 2);
        words[w++] = (i == 0) ? blobs->offsets[0] : 0;
        words[w++] = oddId;
        words[w++] = blobs->sizes[i];
        words[w++] = (i + 1 < blobs->n) ? (oddId + 1) : 0;
    }

    memcpy(buf, words, nWords * sizeof(uint64_t));
    free(words);

    uint64_t state = 0x123456789abcdef0ULL;
    for (size_t i = 1; i < blobs->n; i++) {
        uint64_t off = blobs->offsets[i];
        uint64_t sz = blobs->sizes[i];
        float *dst = (float *)(buf + off);
        uint64_t n = sz / 4ULL;
        for (uint64_t j = 0; j < n; j++) {
            state = state * 6364136223846793005ULL + 1ULL;
            uint32_t r = (uint32_t)(state >> 32);
            int32_t x = (int32_t)(r % 20001U) - 10000;
            dst[j] = ((float)x) * 0.0001f;
        }
    }

    return data;
}

static BOOL write_json_file(NSString *path, id object, NSError **err) {
    NSData *d = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted error:err];
    if (!d) {
        return NO;
    }
    return [d writeToFile:path options:NSDataWritingAtomic error:err];
}

static BOOL write_model(
    NSString *modelDir,
    const Arch *arch,
    BOOL useLookup,
    NSString *sdpaType,
    BOOL withScaffold,
    NSString **detailOut)
{
    BlobPlan blobs;
    blob_plan_init(&blobs);

    NSMutableArray *layers = [NSMutableArray array];
    NSMutableDictionary *shapes = [NSMutableDictionary dictionary];

    NSString *inputName = useLookup ? @"token_ids" : @"onehot";
    add_shape(shapes, inputName, useLookup ? 1 : arch->vocab);

    NSString *h = @"h0";
    NSDictionary *embed = make_inner_product_layer(
        useLookup ? @"embed_lookup" : @"embed_proj",
        inputName,
        h,
        arch->vocab,
        arch->hidden,
        YES,
        NO,
        useLookup,
        &blobs);
    if (!embed) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = @"failed to create embedding layer";
        return NO;
    }
    [layers addObject:embed];
    add_shape(shapes, h, arch->hidden);

    for (int l = 0; l < arch->layers; l++) {
        NSString *lp = [NSString stringWithFormat:@"L%d", l];

        NSString *qLin = [NSString stringWithFormat:@"%@_q_lin", lp];
        NSString *kLin = [NSString stringWithFormat:@"%@_k_lin", lp];
        NSString *vLin = [NSString stringWithFormat:@"%@_v_lin", lp];
        NSString *q = [NSString stringWithFormat:@"%@_q", lp];
        NSString *k = [NSString stringWithFormat:@"%@_k", lp];
        NSString *v = [NSString stringWithFormat:@"%@_v", lp];
        NSString *a = [NSString stringWithFormat:@"%@_attn", lp];

        NSDictionary *qL = make_inner_product_layer([NSString stringWithFormat:@"%@_q_proj", lp], h, qLin, arch->hidden, arch->hidden, YES, NO, NO, &blobs);
        NSDictionary *kL = make_inner_product_layer([NSString stringWithFormat:@"%@_k_proj", lp], h, kLin, arch->hidden, arch->hidden, YES, NO, NO, &blobs);
        NSDictionary *vL = make_inner_product_layer([NSString stringWithFormat:@"%@_v_proj", lp], h, vLin, arch->hidden, arch->hidden, YES, NO, NO, &blobs);
        if (!qL || !kL || !vL) {
            blob_plan_free(&blobs);
            if (detailOut) *detailOut = [NSString stringWithFormat:@"failed to create qkv for layer %d", l];
            return NO;
        }
        [layers addObject:qL];
        [layers addObject:kL];
        [layers addObject:vL];
        add_shape(shapes, qLin, arch->hidden);
        add_shape(shapes, kLin, arch->hidden);
        add_shape(shapes, vLin, arch->hidden);

        NSString *attnOut = nil;
        if (sdpaType) {
            NSString *qkv = [NSString stringWithFormat:@"%@,%@,%@", qLin, kLin, vLin];
            NSDictionary *sdpa = make_sdpa_candidate_layer([NSString stringWithFormat:@"%@_sdpa", lp], qkv, a, sdpaType);
            [layers addObject:sdpa];
            attnOut = a;
            add_shape(shapes, attnOut, arch->hidden);
        } else {
            int headDim = arch->hidden / arch->heads;
            double scale = 1.0 / sqrt((double)headDim);
            NSString *qk = [NSString stringWithFormat:@"%@_qk", lp];
            NSString *qkv = [NSString stringWithFormat:@"%@_qkv", lp];
            NSString *scaled = [NSString stringWithFormat:@"%@_scaled", lp];
            NSString *packed = [NSString stringWithFormat:@"%@_packed", lp];

            [layers addObject:make_reshape_layer([NSString stringWithFormat:@"%@_q_reshape", lp], qLin, q, headDim, arch->heads, 4)];
            [layers addObject:make_reshape_layer([NSString stringWithFormat:@"%@_k_reshape", lp], kLin, k, headDim, arch->heads, 4)];
            [layers addObject:make_reshape_layer([NSString stringWithFormat:@"%@_v_reshape", lp], vLin, v, headDim, arch->heads, 4)];

            NSDictionary *qkL = make_elementwise_layer([NSString stringWithFormat:@"%@_qk_mul", lp], [NSString stringWithFormat:@"%@,%@", q, k], qk, 0);
            NSDictionary *qvL = make_elementwise_layer([NSString stringWithFormat:@"%@_qkv_mul", lp], [NSString stringWithFormat:@"%@,%@", qk, v], qkv, 0);
            NSDictionary *scaleL = make_elementwise_layer_alpha([NSString stringWithFormat:@"%@_scale", lp], qkv, scaled, 1, scale);
            NSDictionary *packL = make_reshape_layer([NSString stringWithFormat:@"%@_pack", lp], scaled, packed, arch->hidden, 1, 3);
            [layers addObject:qkL];
            [layers addObject:qvL];
            [layers addObject:scaleL];
            [layers addObject:packL];

            add_shape_ex(shapes, q, headDim, arch->heads, 4);
            add_shape_ex(shapes, k, headDim, arch->heads, 4);
            add_shape_ex(shapes, v, headDim, arch->heads, 4);
            add_shape_ex(shapes, qk, headDim, arch->heads, 4);
            add_shape_ex(shapes, qkv, headDim, arch->heads, 4);
            add_shape_ex(shapes, scaled, headDim, arch->heads, 4);
            add_shape(shapes, packed, arch->hidden);
            attnOut = packed;
        }

        NSString *o = [NSString stringWithFormat:@"%@_o", lp];
        NSDictionary *oL = make_inner_product_layer([NSString stringWithFormat:@"%@_o_proj", lp], attnOut, o, arch->hidden, arch->hidden, YES, NO, NO, &blobs);
        if (!oL) {
            blob_plan_free(&blobs);
            if (detailOut) *detailOut = [NSString stringWithFormat:@"failed to create o projection for layer %d", l];
            return NO;
        }
        [layers addObject:oL];
        add_shape(shapes, o, arch->hidden);

        NSString *res1 = [NSString stringWithFormat:@"%@_res1", lp];
        [layers addObject:make_elementwise_layer([NSString stringWithFormat:@"%@_res1_add", lp], [NSString stringWithFormat:@"%@,%@", h, o], res1, 0)];
        add_shape(shapes, res1, arch->hidden);

        NSString *gate = [NSString stringWithFormat:@"%@_gate", lp];
        NSString *sig = [NSString stringWithFormat:@"%@_gate_sig", lp];
        NSString *silu = [NSString stringWithFormat:@"%@_silu", lp];
        NSString *up = [NSString stringWithFormat:@"%@_up", lp];
        NSString *mul = [NSString stringWithFormat:@"%@_mul", lp];
        NSString *down = [NSString stringWithFormat:@"%@_down", lp];
        NSString *res2 = [NSString stringWithFormat:@"%@_res2", lp];

        NSDictionary *gateL = make_inner_product_layer([NSString stringWithFormat:@"%@_gate_proj", lp], res1, gate, arch->hidden, arch->intermediate, YES, NO, NO, &blobs);
        NSDictionary *upL = make_inner_product_layer([NSString stringWithFormat:@"%@_up_proj", lp], res1, up, arch->hidden, arch->intermediate, YES, NO, NO, &blobs);
        NSDictionary *downL = make_inner_product_layer([NSString stringWithFormat:@"%@_down_proj", lp], mul, down, arch->intermediate, arch->hidden, YES, NO, NO, &blobs);
        if (!gateL || !upL || !downL) {
            blob_plan_free(&blobs);
            if (detailOut) *detailOut = [NSString stringWithFormat:@"failed to create ffn for layer %d", l];
            return NO;
        }

        [layers addObject:gateL];
        [layers addObject:make_activation_sigmoid_layer([NSString stringWithFormat:@"%@_sigmoid", lp], gate, sig)];
        [layers addObject:make_elementwise_layer([NSString stringWithFormat:@"%@_silu_mul", lp], [NSString stringWithFormat:@"%@,%@", gate, sig], silu, 1)];
        [layers addObject:upL];
        [layers addObject:make_elementwise_layer([NSString stringWithFormat:@"%@_ffn_mul", lp], [NSString stringWithFormat:@"%@,%@", silu, up], mul, 1)];
        [layers addObject:downL];
        [layers addObject:make_elementwise_layer([NSString stringWithFormat:@"%@_res2_add", lp], [NSString stringWithFormat:@"%@,%@", res1, down], res2, 0)];

        add_shape(shapes, gate, arch->intermediate);
        add_shape(shapes, sig, arch->intermediate);
        add_shape(shapes, silu, arch->intermediate);
        add_shape(shapes, up, arch->intermediate);
        add_shape(shapes, mul, arch->intermediate);
        add_shape(shapes, down, arch->hidden);
        add_shape(shapes, res2, arch->hidden);

        h = res2;
    }

    NSDictionary *lm = make_inner_product_layer(@"lm_head", h, @"logits", arch->hidden, arch->vocab, YES, YES, NO, &blobs);
    if (!lm) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = @"failed to create lm_head";
        return NO;
    }
    [layers addObject:lm];
    add_shape(shapes, @"logits", arch->vocab);

    NSDictionary *net = @{
        @"storage": @"model.espresso.weights",
        @"analyses": @{},
        @"properties": @{},
        @"format_version": @200,
        @"metadata_in_weights": @[],
        @"layers": layers,
    };
    NSDictionary *shape = @{@"layer_shapes": shapes};

    NSString *weightsErr = nil;
    NSData *weights = build_weights_data(&blobs, &weightsErr);
    if (!weights) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = [NSString stringWithFormat:@"weights build failed: %@", weightsErr ?: @"unknown"];;
        return NO;
    }

    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:modelDir error:nil];
    NSError *err = nil;
    if (![fm createDirectoryAtPath:modelDir withIntermediateDirectories:YES attributes:nil error:&err]) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = [NSString stringWithFormat:@"create dir failed: %s", err_desc(err)];
        return NO;
    }

    NSString *netPath = [modelDir stringByAppendingPathComponent:@"model.espresso.net"];
    NSString *shapePath = [modelDir stringByAppendingPathComponent:@"model.espresso.shape"];
    NSString *weightsPath = [modelDir stringByAppendingPathComponent:@"model.espresso.weights"];

    if (!write_json_file(netPath, net, &err)) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = [NSString stringWithFormat:@"write net failed: %s", err_desc(err)];
        return NO;
    }
    if (!write_json_file(shapePath, shape, &err)) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = [NSString stringWithFormat:@"write shape failed: %s", err_desc(err)];
        return NO;
    }
    if (![weights writeToFile:weightsPath options:NSDataWritingAtomic error:&err]) {
        blob_plan_free(&blobs);
        if (detailOut) *detailOut = [NSString stringWithFormat:@"write weights failed: %s", err_desc(err)];
        return NO;
    }

    if (withScaffold) {
        if (!stage_template_scaffold(modelDir, &err)) {
            blob_plan_free(&blobs);
            if (detailOut) *detailOut = [NSString stringWithFormat:@"stage scaffold failed: %s", err_desc(err)];
            return NO;
        }
    }

    printf("generated model path=%s layers=%lu blobs=%lu weights_bytes=%llu\n",
           modelDir.UTF8String,
           (unsigned long)layers.count,
           (unsigned long)blobs.n,
           (unsigned long long)weights.length);

    blob_plan_free(&blobs);
    return YES;
}

static id model_with_key_probe(NSString *modelPath, const char **outKey) {
    static const char *keys[] = {"main", "s", "default"};

    id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), modelPath);
    if (!url) {
        return nil;
    }

    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        NSString *k = [NSString stringWithUTF8String:keys[i]];
        id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, k);
        if (model) {
            if (outKey) {
                *outKey = keys[i];
            }
            return model;
        }
    }

    return nil;
}

static BOOL compile_and_load_model(NSString *modelPath, id *outClient, id *outModel, NSString **detail) {
    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        if (detail) *detail = @"_ANEClient sharedConnection=nil";
        return NO;
    }

    const char *key = NULL;
    id model = model_with_key_probe(modelPath, &key);
    if (!model) {
        if (detail) *detail = [NSString stringWithFormat:@"modelAtURL:key: failed for all keys path=%@", modelPath];
        return NO;
    }

    NSError *err = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(compileModel:options:qos:error:), model, @{}, kQoS, &err);
    if (!ok && err && [err.domain isEqualToString:NSCocoaErrorDomain] && err.code == 4097) {
        usleep(200000);
        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(compileModel:options:qos:error:), model, @{}, kQoS, &err);
    }
    if (!ok) {
        if (detail) *detail = [NSString stringWithFormat:@"compileModel failed: %s", err_desc(err)];
        return NO;
    }

    err = nil;
    ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(loadModel:options:qos:error:), model, @{}, kQoS, &err);
    if (!ok) {
        usleep(100000);
        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(loadModel:options:qos:error:), model, @{}, kQoS, &err);
    }
    if (!ok) {
        if (detail) *detail = [NSString stringWithFormat:@"loadModel failed: %s", err_desc(err)];
        return NO;
    }

    if (outClient) *outClient = client;
    if (outModel) *outModel = model;
    return YES;
}

static IOSurfaceRef make_surface(size_t bytes) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static void write_surface_f32(IOSurfaceRef surf, const float *vals, int n) {
    IOSurfaceLock(surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(surf), vals, (size_t)n * sizeof(float));
    IOSurfaceUnlock(surf, 0, NULL);
}

static void read_surface_f32(IOSurfaceRef surf, float *vals, int n) {
    IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(vals, IOSurfaceGetBaseAddress(surf), (size_t)n * sizeof(float));
    IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da < db) ? -1 : (da > db ? 1 : 0);
}

static BenchResult benchmark_model(NSString *modelPath, BOOL useLookup, const Arch *arch) {
    BenchResult r = {0};

    id client = nil;
    id model = nil;
    NSString *detail = nil;
    if (!compile_and_load_model(modelPath, &client, &model, &detail)) {
        r.error = detail;
        return r;
    }

    const int inputCount = useLookup ? 1 : arch->vocab;
    const int outputCount = arch->vocab;
    size_t inBytes = (size_t)inputCount * sizeof(float);
    size_t outBytes = (size_t)outputCount * sizeof(float);

    IOSurfaceRef inSurf = make_surface(inBytes);
    IOSurfaceRef outSurf = make_surface(outBytes);
    if (!inSurf || !outSurf) {
        r.error = @"surface allocation failed";
        if (inSurf) CFRelease(inSurf);
        if (outSurf) CFRelease(outSurf);
        return r;
    }

    float *inVals = (float *)calloc((size_t)inputCount, sizeof(float));
    float *outVals = (float *)calloc((size_t)outputCount, sizeof(float));
    if (!inVals || !outVals) {
        r.error = @"buffer allocation failed";
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }

    if (useLookup) {
        inVals[0] = 123.0f; // token id
    } else {
        inVals[123 % inputCount] = 1.0f; // one-hot
    }
    write_surface_f32(inSurf, inVals, inputCount);

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0);
    if (!req) {
        r.error = @"request creation failed";
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }

    NSError *err = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        client,
        @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
        model,
        req,
        YES,
        &err);
    if (!ok) {
        r.error = [NSString stringWithFormat:@"mapIOSurfaces failed: %s", err_desc(err)];
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }

    SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    if (![client respondsToSelector:evalSel]) {
        evalSel = @selector(evaluateWithModel:options:request:qos:error:);
    }

    for (int i = 0; i < kWarmup; i++) {
        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client, evalSel, model, @{}, req, kQoS, &err);
        if (!ok) {
            r.error = [NSString stringWithFormat:@"warmup eval failed iter=%d err=%s", i, err_desc(err)];
            free(inVals);
            free(outVals);
            CFRelease(inSurf);
            CFRelease(outSurf);
            return r;
        }
    }

    int iters = env_int("ANE_TRANSFORMER_ITERS", kIters);
    double *samples = (double *)calloc((size_t)iters, sizeof(double));
    if (!samples) {
        r.error = @"sample allocation failed";
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }

    for (int i = 0; i < iters; i++) {
        double t0 = now_us();
        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client, evalSel, model, @{}, req, kQoS, &err);
        double t1 = now_us();
        if (!ok) {
            r.error = [NSString stringWithFormat:@"eval failed iter=%d err=%s", i, err_desc(err)];
            free(samples);
            free(inVals);
            free(outVals);
            CFRelease(inSurf);
            CFRelease(outSurf);
            return r;
        }
        samples[i] = t1 - t0;
    }

    read_surface_f32(outSurf, outVals, outputCount);

    double sum = 0.0;
    double mn = DBL_MAX;
    double mx = 0.0;
    for (int i = 0; i < iters; i++) {
        double v = samples[i];
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    qsort(samples, (size_t)iters, sizeof(samples[0]), cmp_double);

    r.ok = YES;
    r.avg_us = sum / (double)iters;
    r.p50_us = samples[iters / 2];
    r.min_us = mn;
    r.max_us = mx;
    r.out0 = outVals[0];
    r.out1 = outVals[1];
    r.out2 = outVals[2];

    NSError *uerr = nil;
    ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &uerr);

    free(samples);
    free(inVals);
    free(outVals);
    CFRelease(inSurf);
    CFRelease(outSurf);
    return r;
}

static ProbeResult probe_lookup_support(void) {
    ProbeResult r = {0};
    Arch a = {.vocab = 256, .hidden = 64, .intermediate = 128, .layers = 0, .heads = 1};
    NSString *path = @"/tmp/ane_lookup_probe.mlmodelc";
    NSString *detail = nil;
    if (!write_model(path, &a, YES, nil, NO, &detail)) {
        r.error = [NSString stringWithFormat:@"lookup probe model write failed: %@", detail ?: @"unknown"];
        return r;
    }

    BenchResult b = benchmark_model(path, YES, &a);
    if (!b.ok) {
        r.error = [NSString stringWithFormat:@"lookup probe compile/eval failed: %@", b.error ?: @"unknown"];
        return r;
    }

    r.ok = YES;
    return r;
}

static ProbeResult probe_sdpa_type(NSString *typeName) {
    ProbeResult r = {0};
    NSString *src = @"/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_full_core.mlmodelc";
    NSString *dst = [NSString stringWithFormat:@"/tmp/ane_sdpa_probe_%@.mlmodelc", typeName];

    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:dst error:nil];
    NSError *err = nil;
    if (![fm copyItemAtPath:src toPath:dst error:&err]) {
        r.error = [NSString stringWithFormat:@"copy probe template failed (%@): %s", typeName, err_desc(err)];
        return r;
    }

    NSString *netPath = [dst stringByAppendingPathComponent:@"model.espresso.net"];
    NSData *netData = [NSData dataWithContentsOfFile:netPath options:0 error:&err];
    if (!netData) {
        r.error = [NSString stringWithFormat:@"read probe net failed (%@): %s", typeName, err_desc(err)];
        return r;
    }

    id obj = [NSJSONSerialization JSONObjectWithData:netData options:NSJSONReadingMutableContainers error:&err];
    if (!obj || ![obj isKindOfClass:[NSDictionary class]]) {
        r.error = [NSString stringWithFormat:@"parse probe net failed (%@): %s", typeName, err_desc(err)];
        return r;
    }

    NSMutableDictionary *net = [(NSDictionary *)obj mutableCopy];
    NSMutableArray *layers = [net[@"layers"] mutableCopy];
    if (!layers || layers.count <= 11 || ![layers[11] isKindOfClass:[NSDictionary class]]) {
        r.error = [NSString stringWithFormat:@"probe net missing expected layer index (%@)", typeName];
        return r;
    }

    NSMutableDictionary *layer = [layers[11] mutableCopy];
    layer[@"type"] = typeName;
    layer[@"name"] = [NSString stringWithFormat:@"probe_%@", typeName];
    layer[@"debug_info"] = [NSString stringWithFormat:@"probe_%@", typeName];
    layer[@"bottom"] = @"q.1,k.1,v.1";
    [layer removeObjectForKey:@"operation"];
    [layer removeObjectForKey:@"alpha"];
    [layer removeObjectForKey:@"beta"];
    [layer removeObjectForKey:@"fused_relu"];
    layers[11] = layer;
    net[@"layers"] = layers;

    NSData *patched = [NSJSONSerialization dataWithJSONObject:net options:0 error:&err];
    if (!patched || ![patched writeToFile:netPath options:NSDataWritingAtomic error:&err]) {
        r.error = [NSString stringWithFormat:@"write patched probe net failed (%@): %s", typeName, err_desc(err)];
        return r;
    }

    NSString *detail = nil;
    id client = nil;
    id model = nil;
    if (!compile_and_load_model(dst, &client, &model, &detail)) {
        r.error = [NSString stringWithFormat:@"compile failed (%@): %@", typeName, detail ?: @"unknown"];
        return r;
    }

    NSError *uerr = nil;
    ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &uerr);

    r.ok = YES;
    return r;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (!setup_ane_classes()) {
            return 2;
        }

        Arch arch = {
            .vocab = env_int("ANE_DRAFT_VOCAB", 32000),
            .hidden = env_int("ANE_DRAFT_HIDDEN", 768),
            .intermediate = env_int("ANE_DRAFT_INTERMEDIATE", 3072),
            .layers = env_nonneg_int("ANE_DRAFT_LAYERS", 6),
            .heads = env_int("ANE_DRAFT_HEADS", 12),
        };

        printf("# Full Draft Transformer Espresso Build\n");
        printf("arch vocab=%d hidden=%d intermediate=%d layers=%d heads=%d seq=1\n",
               arch.vocab, arch.hidden, arch.intermediate, arch.layers, arch.heads);

        ProbeResult lookup = probe_lookup_support();
        printf("lookup_probe ok=%d detail=%s\n", lookup.ok ? 1 : 0, lookup.error ? lookup.error.UTF8String : "ok");

        NSArray<NSString *> *sdpaCandidates = @[@"sdpa", @"scaled_dot_product_attention", @"attention"];
        NSString *acceptedSDPA = nil;
        for (NSString *cand in sdpaCandidates) {
            ProbeResult p = probe_sdpa_type(cand);
            printf("sdpa_probe type=%s ok=%d detail=%s\n",
                   cand.UTF8String,
                   p.ok ? 1 : 0,
                   p.error ? p.error.UTF8String : "ok");
            if (p.ok && !acceptedSDPA) {
                acceptedSDPA = cand;
            }
        }

        BOOL useLookup = lookup.ok;
        NSString *sdpaType = acceptedSDPA; // nil means fallback to elementwise attention chain.

        NSString *modelPath = @"/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft6l_transformer_espresso.mlmodelc";
        const char *rawPath = getenv("ANE_DRAFT_MODEL_OUT");
        if (rawPath && rawPath[0]) {
            modelPath = [NSString stringWithUTF8String:rawPath];
        }

        NSString *detail = nil;
        printf("build_full_model lookup=%d sdpa_type=%s\n", useLookup ? 1 : 0, sdpaType ? sdpaType.UTF8String : "fallback_elementwise");
        if (!write_model(modelPath, &arch, useLookup, sdpaType, NO, &detail)) {
            printf("build_full_model attempt=no_scaffold failed detail=%s\n", detail ? detail.UTF8String : "unknown");
            detail = nil;
            if (!write_model(modelPath, &arch, useLookup, sdpaType, YES, &detail)) {
                printf("build_full_model attempt=with_scaffold failed detail=%s\n", detail ? detail.UTF8String : "unknown");
                return 2;
            }
        }

        BenchResult full = benchmark_model(modelPath, useLookup, &arch);
        if (!full.ok) {
            printf("benchmark_full ok=0 err=%s\n", full.error ? full.error.UTF8String : "unknown");
            return 2;
        }

        int iters = env_int("ANE_TRANSFORMER_ITERS", kIters);
        printf("benchmark_full ok=1 path=%s iters=%d avg_us=%.3f p50_us=%.3f min_us=%.3f max_us=%.3f out=[%.6f, %.6f, %.6f]\n",
               modelPath.UTF8String,
               iters,
               full.avg_us,
               full.p50_us,
               full.min_us,
               full.max_us,
               full.out0,
               full.out1,
               full.out2);

        printf("\n# Summary\n");
        printf("- is_lookup support: %s\n", lookup.ok ? "YES" : "NO");
        printf("- accepted sdpa type: %s\n", sdpaType ? sdpaType.UTF8String : "none (using elementwise fallback)");
        printf("- full model layers: %d (single .espresso.net)\n", 2 + (arch.layers * (sdpaType ? 13 : 19)));
        printf("- eval latency (seq=1): %.3f us\n", full.avg_us);

        return 0;
    }
}
