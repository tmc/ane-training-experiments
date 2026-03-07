// test_espresso_gen.m
// Hand-craft a tiny Espresso FFN model and benchmark ANE execution.
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
static const int kDefaultDim = 64;
static const int kDefaultHidden = 256;
static const int kWarmup = 10;
static const int kIters = 100;

static int gDim = 64;
static int gHidden = 256;

static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIO;
static Class CNSURL;

typedef struct {
    uint64_t bias0_bytes;
    uint64_t weight0_bytes;
    uint64_t bias1_bytes;
    uint64_t weight1_bytes;
    uint64_t bias2_bytes;
    uint64_t weight2_bytes;
    uint64_t bias0_off;
    uint64_t weight0_off;
    uint64_t bias1_off;
    uint64_t weight1_off;
    uint64_t bias2_off;
    uint64_t weight2_off;
    uint64_t total_bytes;
} EspressoLayout;

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
} BenchResult;

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
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

static double now_us(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000000.0;
}

static uint64_t align256(uint64_t v) {
    return (v + 255ULL) & ~255ULL;
}

static void compute_layout(EspressoLayout *out) {
    memset(out, 0, sizeof(*out));

    out->bias0_bytes = (uint64_t)gHidden * 16ULL;
    out->weight0_bytes = (uint64_t)gDim * (uint64_t)gHidden * 4ULL;
    out->bias1_bytes = (uint64_t)gHidden * 16ULL;
    out->weight1_bytes = (uint64_t)gDim * (uint64_t)gHidden * 4ULL;
    out->bias2_bytes = (uint64_t)gDim * 16ULL;
    out->weight2_bytes = (uint64_t)gHidden * (uint64_t)gDim * 4ULL;

    out->bias0_off = 56;
    out->weight0_off = align256(out->bias0_off + out->bias0_bytes);
    out->bias1_off = out->weight0_off + out->weight0_bytes;
    out->weight1_off = align256(out->bias1_off + out->bias1_bytes);
    out->bias2_off = out->weight1_off + out->weight1_bytes;
    out->weight2_off = align256(out->bias2_off + out->bias2_bytes);
    out->total_bytes = out->weight2_off + out->weight2_bytes;
}

static uint32_t rng_u32(uint64_t *state) {
    *state = (*state * 6364136223846793005ULL) + 1ULL;
    return (uint32_t)(*state >> 32);
}

static float rand_fp16_like(uint64_t *state) {
    int32_t x = (int32_t)(rng_u32(state) % 20001U) - 10000;
    float v = (float)x / 10000.0f;
    _Float16 h = (_Float16)v;
    return (float)h;
}

static BOOL write_text_file(NSString *path, NSString *text, NSError **err) {
    return [text writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:err];
}

static NSString *build_net_json(void) {
    return [NSString stringWithFormat:
        @"{\n"
         "  \"storage\" : \"model.espresso.weights\",\n"
         "  \"analyses\" : {\n\n  },\n"
         "  \"properties\" : {\n\n  },\n"
         "  \"format_version\" : 200,\n"
         "  \"metadata_in_weights\" : [\n\n  ],\n"
         "  \"layers\" : [\n"
         "    {\n"
         "      \"nB\" : %d,\n"
         "      \"top\" : \"linear_0\",\n"
         "      \"has_biases\" : 1,\n"
         "      \"weights\" : {\n\n      },\n"
         "      \"nC\" : %d,\n"
         "      \"blob_weights\" : 3,\n"
         "      \"type\" : \"inner_product\",\n"
         "      \"has_relu\" : 0,\n"
         "      \"bottom\" : \"x\",\n"
         "      \"blob_biases\" : 1,\n"
         "      \"has_tanh\" : 0,\n"
         "      \"debug_info\" : \"linear_0\",\n"
         "      \"name\" : \"linear_0\",\n"
         "      \"has_prelu\" : 0\n"
         "    },\n"
         "    {\n"
         "      \"bottom\" : \"linear_0\",\n"
         "      \"weights\" : {\n\n      },\n"
         "      \"mode\" : 3,\n"
         "      \"debug_info\" : \"8__silu_sigmoid__\",\n"
         "      \"top\" : \"8__silu_sigmoid__\",\n"
         "      \"type\" : \"activation\",\n"
         "      \"name\" : \"8__silu_sigmoid__\"\n"
         "    },\n"
         "    {\n"
         "      \"bottom\" : \"linear_0,8__silu_sigmoid__\",\n"
         "      \"alpha\" : 1,\n"
         "      \"operation\" : 1,\n"
         "      \"weights\" : {\n\n      },\n"
         "      \"fused_relu\" : 0,\n"
         "      \"debug_info\" : \"8\",\n"
         "      \"top\" : \"8\",\n"
         "      \"type\" : \"elementwise\",\n"
         "      \"name\" : \"8\",\n"
         "      \"beta\" : 0\n"
         "    },\n"
         "    {\n"
         "      \"nB\" : %d,\n"
         "      \"top\" : \"linear_1\",\n"
         "      \"has_biases\" : 1,\n"
         "      \"weights\" : {\n\n      },\n"
         "      \"nC\" : %d,\n"
         "      \"blob_weights\" : 7,\n"
         "      \"type\" : \"inner_product\",\n"
         "      \"has_relu\" : 0,\n"
         "      \"bottom\" : \"x\",\n"
         "      \"blob_biases\" : 5,\n"
         "      \"has_tanh\" : 0,\n"
         "      \"debug_info\" : \"linear_1\",\n"
         "      \"name\" : \"linear_1\",\n"
         "      \"has_prelu\" : 0\n"
         "    },\n"
         "    {\n"
         "      \"bottom\" : \"8,linear_1\",\n"
         "      \"alpha\" : 1,\n"
         "      \"operation\" : 1,\n"
         "      \"weights\" : {\n\n      },\n"
         "      \"fused_relu\" : 0,\n"
         "      \"debug_info\" : \"input\",\n"
         "      \"top\" : \"input\",\n"
         "      \"type\" : \"elementwise\",\n"
         "      \"name\" : \"input\",\n"
         "      \"beta\" : 0\n"
         "    },\n"
         "    {\n"
         "      \"has_prelu\" : 0,\n"
         "      \"top\" : \"linear_2\",\n"
         "      \"has_biases\" : 1,\n"
         "      \"weights\" : {\n\n      },\n"
         "      \"nC\" : %d,\n"
         "      \"blob_weights\" : 11,\n"
         "      \"type\" : \"inner_product\",\n"
         "      \"has_relu\" : 0,\n"
         "      \"attributes\" : {\n"
         "        \"is_output\" : 1\n"
         "      },\n"
         "      \"bottom\" : \"input\",\n"
         "      \"debug_info\" : \"linear_2\",\n"
         "      \"has_tanh\" : 0,\n"
         "      \"blob_biases\" : 9,\n"
         "      \"name\" : \"linear_2\",\n"
         "      \"nB\" : %d\n"
         "    }\n"
         "  ]\n"
         "}\n",
         gDim, gHidden, gDim, gHidden, gDim, gHidden
    ];
}

static NSString *build_shape_json(void) {
    return [NSString stringWithFormat:
        @"{\n"
         "  \"layer_shapes\" : {\n"
         "    \"linear_1\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    },\n"
         "    \"input\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    },\n"
         "    \"x\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    },\n"
         "    \"8\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    },\n"
         "    \"linear_0\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    },\n"
         "    \"8__silu_sigmoid__\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    },\n"
         "    \"linear_2\" : {\n"
         "      \"k\" : 1,\n"
         "      \"w\" : %d,\n"
         "      \"n\" : 1,\n"
         "      \"_rank\" : 3,\n"
         "      \"h\" : 1\n"
         "    }\n"
         "  }\n"
         "}\n",
         gHidden, gHidden, gDim, gHidden, gHidden, gHidden, gDim
    ];
}

static void fill_weight_blob(float *dst, uint64_t elems, uint64_t seed) {
    uint64_t s = seed;
    for (uint64_t i = 0; i < elems; i++) {
        dst[i] = rand_fp16_like(&s);
    }
}

static NSData *build_weights_data(const EspressoLayout *layout) {
    NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)layout->total_bytes];
    if (!data) {
        return nil;
    }

    uint8_t *buf = (uint8_t *)data.mutableBytes;
    if (!buf) {
        return nil;
    }

    uint64_t words[26] = {
        12, 0,
        56, 1,
        layout->bias0_bytes, 2,
        0, 3,
        layout->weight0_bytes, 4,
        0, 5,
        layout->bias1_bytes, 6,
        0, 7,
        layout->weight1_bytes, 8,
        0, 9,
        layout->bias2_bytes, 10,
        0, 11,
        layout->weight2_bytes, 0,
    };

    memcpy(buf, words, sizeof(words));

    fill_weight_blob((float *)(buf + layout->weight0_off), layout->weight0_bytes / 4ULL, 0x1234ULL);
    fill_weight_blob((float *)(buf + layout->weight1_off), layout->weight1_bytes / 4ULL, 0x5678ULL);
    fill_weight_blob((float *)(buf + layout->weight2_off), layout->weight2_bytes / 4ULL, 0x9abcULL);

    return data;
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
        : @"/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_ffn.mlmodelc";

    return copy_if_exists([srcDir stringByAppendingPathComponent:@"coremldata.bin"], [dstDir stringByAppendingPathComponent:@"coremldata.bin"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"metadata.json"], [dstDir stringByAppendingPathComponent:@"metadata.json"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"model.mil"], [dstDir stringByAppendingPathComponent:@"model.mil"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"analytics/coremldata.bin"], [dstDir stringByAppendingPathComponent:@"analytics/coremldata.bin"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"model/coremldata.bin"], [dstDir stringByAppendingPathComponent:@"model/coremldata.bin"], err) &&
           copy_if_exists([srcDir stringByAppendingPathComponent:@"neural_network_optionals/coremldata.bin"], [dstDir stringByAppendingPathComponent:@"neural_network_optionals/coremldata.bin"], err);
}

static BOOL write_handcrafted_model(NSString *modelDir, const EspressoLayout *layout, BOOL withScaffold) {
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:modelDir error:nil];

    NSError *err = nil;
    if (![fm createDirectoryAtPath:modelDir withIntermediateDirectories:YES attributes:nil error:&err]) {
        fprintf(stderr, "create dir failed: %s\n", err_desc(err));
        return NO;
    }

    NSString *netPath = [modelDir stringByAppendingPathComponent:@"model.espresso.net"];
    NSString *shapePath = [modelDir stringByAppendingPathComponent:@"model.espresso.shape"];
    NSString *weightsPath = [modelDir stringByAppendingPathComponent:@"model.espresso.weights"];

    if (!write_text_file(netPath, build_net_json(), &err)) {
        fprintf(stderr, "write net failed: %s\n", err_desc(err));
        return NO;
    }
    if (!write_text_file(shapePath, build_shape_json(), &err)) {
        fprintf(stderr, "write shape failed: %s\n", err_desc(err));
        return NO;
    }

    NSData *weights = build_weights_data(layout);
    if (!weights) {
        fprintf(stderr, "build weights failed\n");
        return NO;
    }
    if (![weights writeToFile:weightsPath options:NSDataWritingAtomic error:&err]) {
        fprintf(stderr, "write weights failed: %s\n", err_desc(err));
        return NO;
    }

    if (withScaffold) {
        if (!stage_template_scaffold(modelDir, &err)) {
            fprintf(stderr, "stage scaffold failed: %s\n", err_desc(err));
            return NO;
        }
    }

    printf("generated model: %s\n", modelDir.UTF8String);
    printf("layout bytes: bias0=%llu w0=%llu bias1=%llu w1=%llu bias2=%llu w2=%llu total=%llu\n",
           (unsigned long long)layout->bias0_bytes,
           (unsigned long long)layout->weight0_bytes,
           (unsigned long long)layout->bias1_bytes,
           (unsigned long long)layout->weight1_bytes,
           (unsigned long long)layout->bias2_bytes,
           (unsigned long long)layout->weight2_bytes,
           (unsigned long long)layout->total_bytes);
    printf("offsets: bias0=%llu w0=%llu bias1=%llu w1=%llu bias2=%llu w2=%llu\n",
           (unsigned long long)layout->bias0_off,
           (unsigned long long)layout->weight0_off,
           (unsigned long long)layout->bias1_off,
           (unsigned long long)layout->weight1_off,
           (unsigned long long)layout->bias2_off,
           (unsigned long long)layout->weight2_off);
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

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da < db) ? -1 : (da > db ? 1 : 0);
}

static BenchResult benchmark_model(NSString *modelPath) {
    BenchResult r = {0};

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        fprintf(stderr, "_ANEClient sharedConnection=nil\n");
        return r;
    }

    const char *modelKey = NULL;
    id model = model_with_key_probe(modelPath, &modelKey);
    if (!model) {
        fprintf(stderr, "modelAtURL:key: failed for all keys path=%s\n", modelPath.UTF8String);
        return r;
    }

    NSError *err = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(compileModel:options:qos:error:), model, @{}, kQoS, &err);
    if (!ok) {
        fprintf(stderr, "compileModel failed: %s\n", err_desc(err));
        return r;
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
        fprintf(stderr, "loadModel failed: %s\n", err_desc(err));
        return r;
    }

    const int count = gDim;
    const size_t bytes = (size_t)count * sizeof(float);

    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef outSurf = make_surface(bytes);
    if (!inSurf || !outSurf) {
        fprintf(stderr, "surface allocation failed\n");
        if (inSurf) CFRelease(inSurf);
        if (outSurf) CFRelease(outSurf);
        return r;
    }

    float *inVals = (float *)calloc((size_t)count, sizeof(float));
    float *outVals = (float *)calloc((size_t)count, sizeof(float));
    if (!inVals || !outVals) {
        fprintf(stderr, "buffer allocation failed\n");
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }
    for (int i = 0; i < count; i++) {
        inVals[i] = ((float)i - (float)(count / 2)) * 0.02f;
    }
    write_surface_f32(inSurf, inVals, count);

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0);
    if (!req) {
        fprintf(stderr, "request creation failed\n");
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }

    if ([req respondsToSelector:@selector(validate)]) {
        BOOL valid = ((BOOL(*)(id, SEL))objc_msgSend)(req, @selector(validate));
        if (!valid) {
            fprintf(stderr, "request validate=false\n");
            free(inVals);
            free(outVals);
            CFRelease(inSurf);
            CFRelease(outSurf);
            return r;
        }
    }

    err = nil;
    ok = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        client,
        @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
        model,
        req,
        YES,
        &err);
    if (!ok) {
        fprintf(stderr, "mapIOSurfaces failed: %s\n", err_desc(err));
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
            fprintf(stderr, "warmup eval failed iter=%d err=%s\n", i, err_desc(err));
            free(inVals);
            free(outVals);
            CFRelease(inSurf);
            CFRelease(outSurf);
            return r;
        }
    }

    double *samples = (double *)calloc((size_t)kIters, sizeof(double));
    if (!samples) {
        fprintf(stderr, "sample allocation failed\n");
        free(inVals);
        free(outVals);
        CFRelease(inSurf);
        CFRelease(outSurf);
        return r;
    }
    for (int i = 0; i < kIters; i++) {
        double t0 = now_us();
        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client, evalSel, model, @{}, req, kQoS, &err);
        double t1 = now_us();
        if (!ok) {
            fprintf(stderr, "eval failed iter=%d err=%s\n", i, err_desc(err));
            free(samples);
            free(inVals);
            free(outVals);
            CFRelease(inSurf);
            CFRelease(outSurf);
            return r;
        }
        samples[i] = t1 - t0;
    }

    read_surface_f32(outSurf, outVals, count);

    double sum = 0.0;
    double mn = DBL_MAX;
    double mx = 0.0;
    for (int i = 0; i < kIters; i++) {
        const double v = samples[i];
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    qsort(samples, kIters, sizeof(samples[0]), cmp_double);

    r.ok = YES;
    r.key = modelKey;
    r.avg_us = sum / (double)kIters;
    r.p50_us = samples[kIters / 2];
    r.min_us = mn;
    r.max_us = mx;
    r.out0 = outVals[0];
    r.out1 = outVals[1];
    r.out2 = outVals[2];

    NSError *uerr = nil;
    ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &uerr);

    CFRelease(inSurf);
    CFRelease(outSurf);
    free(samples);
    free(inVals);
    free(outVals);
    return r;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (!setup_ane_classes()) {
            return 2;
        }

        gDim = env_int("ANE_ESPRESSO_DIM", kDefaultDim);
        gHidden = env_int("ANE_ESPRESSO_HIDDEN", kDefaultHidden);

        EspressoLayout layout;
        compute_layout(&layout);

        const char *rawOut = getenv("ANE_ESPRESSO_GEN_MODEL_PATH");
        NSString *modelDir = rawOut && rawOut[0]
            ? [NSString stringWithUTF8String:rawOut]
            : @"/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/espresso_gen_ffn.mlmodelc";

        printf("# Writing handcrafted Espresso FFN (dim=%d hidden=%d)\n", gDim, gHidden);
        if (!write_handcrafted_model(modelDir, &layout, NO)) {
            return 2;
        }

        printf("# Benchmark handcrafted model with only espresso files\n");
        BenchResult handcrafted = benchmark_model(modelDir);
        if (!handcrafted.ok) {
            printf("# Retrying with scaffold files copied from template mlmodelc\n");
            if (!write_handcrafted_model(modelDir, &layout, YES)) {
                return 2;
            }
            handcrafted = benchmark_model(modelDir);
        }
        if (!handcrafted.ok) {
            fprintf(stderr, "handcrafted benchmark failed\n");
            return 2;
        }

        printf("HANDWRITTEN path=%s key=%s iters=%d avg_us=%.3f p50_us=%.3f min_us=%.3f max_us=%.3f out=[%.6f, %.6f, %.6f]\n",
               modelDir.UTF8String, handcrafted.key ? handcrafted.key : "nil", kIters,
               handcrafted.avg_us, handcrafted.p50_us, handcrafted.min_us, handcrafted.max_us,
               handcrafted.out0, handcrafted.out1, handcrafted.out2);

        const char *rawCore = getenv("ANE_ESPRESSO_COREML_MODEL_PATH");
        if (rawCore && rawCore[0]) {
            NSString *corePath = [NSString stringWithUTF8String:rawCore];
            printf("# Benchmark coremltools model path=%s\n", corePath.UTF8String);
            BenchResult core = benchmark_model(corePath);
            if (!core.ok) {
                fprintf(stderr, "coreml benchmark failed\n");
                return 2;
            }

            printf("COREML path=%s key=%s iters=%d avg_us=%.3f p50_us=%.3f min_us=%.3f max_us=%.3f out=[%.6f, %.6f, %.6f]\n",
                   corePath.UTF8String, core.key ? core.key : "nil", kIters,
                   core.avg_us, core.p50_us, core.min_us, core.max_us,
                   core.out0, core.out1, core.out2);

            printf("COMPARE handwritten_vs_coreml avg_ratio=%.4f p50_ratio=%.4f\n",
                   handcrafted.avg_us / core.avg_us,
                   handcrafted.p50_us / core.p50_us);
        } else {
            printf("# Skip coreml compare (set ANE_ESPRESSO_COREML_MODEL_PATH to enable)\n");
        }

        return 0;
    }
}
