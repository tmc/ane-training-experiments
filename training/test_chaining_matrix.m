// test_chaining_matrix.m
// Native ANE chaining experiments:
// 1) Manual A->B chain via shared IOSurface using _ANEInMemoryModel (no CPU copy between kernels)
// 2) Two-model _ANEClient chaining probe
// 3) IOSurface identity test under _ANEClient
// 4) Signal pre-commit sequencing test

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dispatch/dispatch.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ane_mil_gen.h"

static const unsigned int kQoS = 21;
static volatile float gCpuSink = 0.0f;

static Class CDesc;
static Class CInMem;
static Class CReq;
static Class CAIO;

static Class CClient;
static Class CModel;
static Class CBuf;
static Class COutSet;
static Class CChain;
static Class CSig;
static Class CEnq;
static Class CReady;
static Class CNSURL;

typedef struct {
    id model;
    NSString *tmpDir;
    BOOL loaded;
} InMemModel;

typedef struct {
    id client;
    id model;
    NSString *path;
    NSString *key;
    BOOL loaded;
} ClientModel;

typedef struct {
    id sharedReady;
    id sharedFree;
    id readyEvent;
    id freeEvent;
    NSArray *events;
} SignalPair;

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
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

static void fill_input(float *vals, int n) {
    for (int i = 0; i < n; i++) {
        vals[i] = (float)(i + 1);
    }
}

static int env_int_or_default(const char *name, int fallback) {
    const char *raw = getenv(name);
    if (!raw || raw[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    long v = strtol(raw, &end, 10);
    if (end == raw || *end != '\0') {
        return fallback;
    }
    return (int)v;
}

static BOOL env_enabled_flag(const char *name) {
    const char *raw = getenv(name);
    if (!raw || raw[0] == '\0') {
        return NO;
    }
    if (strcmp(raw, "0") == 0 || strcasecmp(raw, "false") == 0 || strcasecmp(raw, "no") == 0 || strcasecmp(raw, "off") == 0) {
        return NO;
    }
    return YES;
}

static double max_abs_diff(const float *a, const float *b, int n) {
    double d = 0.0;
    for (int i = 0; i < n; i++) {
        double x = fabs((double)a[i] - (double)b[i]);
        if (x > d) {
            d = x;
        }
    }
    return d;
}

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
}

static BOOL is_map_failure(NSError *err) {
    if (err == nil) {
        return NO;
    }
    if (![err.domain isEqualToString:@"com.apple.appleneuralengine"]) {
        return NO;
    }
    if (err.code != 13) {
        return NO;
    }
    NSString *desc = [err description];
    return [desc containsString:@"Program IOSurfaces map failure"];
}

static void print_reboot_hint_if_needed(NSError *err) {
    if (is_map_failure(err)) {
        printf("  NOTE: map failure detected (Code=13). After ANE kernel panic this often needs full reboot.\n");
    }
}

static BOOL ane_setup_classes(void) {
    void *h = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW | RTLD_GLOBAL
    );
    if (!h) {
        fprintf(stderr, "failed to dlopen AppleNeuralEngine.framework\n");
        return NO;
    }

    CDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    CInMem = NSClassFromString(@"_ANEInMemoryModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");

    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CBuf = NSClassFromString(@"_ANEBuffer");
    COutSet = NSClassFromString(@"_ANEIOSurfaceOutputSets");
    CChain = NSClassFromString(@"_ANEChainingRequest");
    CSig = NSClassFromString(@"_ANESharedSignalEvent");
    CEnq = NSClassFromString(@"_ANEOutputSetEnqueue");
    CReady = NSClassFromString(@"_ANEInputBuffersReady");
    CNSURL = NSClassFromString(@"NSURL");

    if (!CDesc || !CInMem || !CReq || !CAIO || !CClient || !CModel || !CBuf || !COutSet || !CChain || !CSig || !CEnq || !CReady || !CNSURL) {
        fprintf(stderr, "failed to resolve one or more required classes\n");
        return NO;
    }
    return YES;
}

static id make_basic_request(id inObj, id outObj, int proc, int inSym, int outSym) {
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[inObj], @[@(inSym)], @[outObj], @[@(outSym)], nil, nil, @(proc)
    );
}

static NSString *mil_mul2(int channels, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3500.32.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
         "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_in\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> y16 = add(x=x16, y=x16)[name=string(\"mul2\")];\n"
         "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to_fp32, x=y16)[name=string(\"cast_out\")];\n"
         "    } -> (y);\n"
         "}\n",
        channels, seq,
        channels, seq,
        channels, seq,
        channels, seq
    ];
}

static NSString *mil_add1(int channels, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3500.32.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
         "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_in\")];\n"
         "        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
         "        tensor<fp16, [1,%d,1,%d]> y16 = add(x=x16, y=one)[name=string(\"add1\")];\n"
         "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to_fp32, x=y16)[name=string(\"cast_out\")];\n"
         "    } -> (y);\n"
         "}\n",
        channels,
        seq,
        channels,
        seq,
        channels,
        seq,
        channels,
        seq
    ];
}

static NSString *mil_add_two_inputs(int channels, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3500.32.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x, tensor<fp32, [1, %d, 1, %d]> z) {\n"
         "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_x\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> z16 = cast(dtype=to_fp16, x=z)[name=string(\"cast_z\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> y16 = add(x=x16, y=z16)[name=string(\"add\")];\n"
         "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to_fp32, x=y16)[name=string(\"cast_out\")];\n"
         "    } -> (y);\n"
         "}\n",
        channels, seq, channels, seq,
        channels, seq, channels, seq, channels, seq, channels, seq
    ];
}

static BOOL compile_load_inmem(NSString *mil, NSData *weightData, InMemModel *outModel, char *errBuf, size_t errBufLen) {
    *outModel = (InMemModel){0};

    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        CDesc,
        @selector(modelWithMILText:weights:optionsPlist:),
        milData,
        wdict,
        nil
    );
    if (!desc) {
        snprintf(errBuf, errBufLen, "modelWithMILText returned nil");
        return NO;
    }

    id model = ((id(*)(Class, SEL, id))objc_msgSend)(CInMem, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) {
        snprintf(errBuf, errBufLen, "inMemoryModelWithDescriptor returned nil");
        return NO;
    }

    id hx = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData) {
        [weightData writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    }

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model,
        @selector(compileWithQoS:options:error:),
        kQoS,
        @{},
        &e
    );
    if (!ok) {
        snprintf(errBuf, errBufLen, "compileWithQoS failed: %s", err_desc(e));
        [fm removeItemAtPath:tmpDir error:nil];
        return NO;
    }

    e = nil;
    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model,
        @selector(loadWithQoS:options:error:),
        kQoS,
        @{},
        &e
    );
    if (!ok) {
        usleep(100000);
        e = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model,
            @selector(loadWithQoS:options:error:),
            kQoS,
            @{},
            &e
        );
    }
    if (!ok) {
        snprintf(errBuf, errBufLen, "loadWithQoS failed after retry: %s", err_desc(e));
        [fm removeItemAtPath:tmpDir error:nil];
        return NO;
    }

    outModel->model = model;
    outModel->tmpDir = tmpDir;
    outModel->loaded = YES;
    return YES;
}

static void unload_inmem(InMemModel *m) {
    if (!m || !m->model || !m->loaded) {
        return;
    }
    NSError *e = nil;
    ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        m->model,
        @selector(unloadWithQoS:error:),
        kQoS,
        &e
    );
    if (m->tmpDir) {
        [[NSFileManager defaultManager] removeItemAtPath:m->tmpDir error:nil];
    }
    m->loaded = NO;
}

static BOOL eval_inmem(const InMemModel *m, IOSurfaceRef inSurf, IOSurfaceRef outSurf, int proc, NSError **err) {
    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    id req = make_basic_request(inObj, outObj, proc, 0, 0);
    if (req && [req respondsToSelector:@selector(validate)]) {
        BOOL valid = ((BOOL(*)(id, SEL))objc_msgSend)(req, @selector(validate));
        if (!valid) {
            if (err) {
                *err = [NSError errorWithDomain:@"test_chaining_matrix" code:1 userInfo:@{NSLocalizedDescriptionKey: @"inmem request validate=false"}];
            }
            return NO;
        }
    }
    return ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        m->model,
        @selector(evaluateWithQoS:options:request:error:),
        kQoS,
        @{},
        req,
        err
    );
}

static BOOL eval_inmem_two_inputs(
    const InMemModel *m,
    IOSurfaceRef inSurfA,
    IOSurfaceRef inSurfB,
    IOSurfaceRef outSurf,
    int proc,
    NSError **err
) {
    id inA = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurfA);
    id inB = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurfB);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[inA, inB], @[@0, @1], @[outObj], @[@0], nil, nil, @(proc)
    );
    if (req && [req respondsToSelector:@selector(validate)]) {
            BOOL valid = ((BOOL(*)(id, SEL))objc_msgSend)(req, @selector(validate));
            if (!valid) {
                if (err) {
                    *err = [NSError errorWithDomain:@"test_chaining_matrix" code:1 userInfo:@{NSLocalizedDescriptionKey: @"two-input request validate=false"}];
                }
                return NO;
            }
    }
    return ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        m->model,
        @selector(evaluateWithQoS:options:request:error:),
        kQoS,
        @{},
        req,
        err
    );
}

static BOOL compile_load_client_model(id client, NSString *path, NSString *key, ClientModel *outModel, char *errBuf, size_t errBufLen) {
    *outModel = (ClientModel){0};

    id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), path);
    id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, key);
    if (!model) {
        snprintf(errBuf, errBufLen, "modelAtURL:key: failed path=%s key=%s", path.UTF8String, key.UTF8String);
        return NO;
    }

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client,
        @selector(compileModel:options:qos:error:),
        model,
        @{},
        kQoS,
        &e
    );
    if (!ok) {
        snprintf(errBuf, errBufLen, "compileModel failed: %s", err_desc(e));
        return NO;
    }

    e = nil;
    ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client,
        @selector(loadModel:options:qos:error:),
        model,
        @{},
        kQoS,
        &e
    );
    if (!ok) {
        usleep(100000);
        e = nil;
        ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(loadModel:options:qos:error:),
            model,
            @{},
            kQoS,
            &e
        );
    }
    if (!ok) {
        snprintf(errBuf, errBufLen, "loadModel failed after retry: %s", err_desc(e));
        return NO;
    }

    outModel->client = client;
    outModel->model = model;
    outModel->path = path;
    outModel->key = key;
    outModel->loaded = YES;
    return YES;
}

static void unload_client_model(ClientModel *m) {
    if (!m || !m->loaded || !m->client || !m->model) {
        return;
    }
    NSError *e = nil;
    ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        m->client,
        @selector(unloadModel:options:qos:error:),
        m->model,
        @{},
        kQoS,
        &e
    );
    m->loaded = NO;
}

static BOOL eval_client(ClientModel *m, IOSurfaceRef inSurf, IOSurfaceRef outSurf, int proc, NSError **err) {
    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    id req = make_basic_request(inObj, outObj, proc, 0, 0);

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        m->client,
        @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
        m->model,
        req,
        YES,
        &e
    );
    if (!ok) {
        if (err) {
            *err = e;
        }
        return NO;
    }

    e = nil;
    ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        m->client,
        @selector(evaluateWithModel:options:request:qos:error:),
        m->model,
        @{},
        req,
        kQoS,
        &e
    );

    ((void(*)(id, SEL, id, id))objc_msgSend)(
        m->client,
        @selector(unmapIOSurfacesWithModel:request:),
        m->model,
        req
    );

    if (!ok && err) {
        *err = e;
    }
    return ok;
}

static id new_shared_event(void) {
    Class cls = NSClassFromString(@"IOSurfaceSharedEvent");
    if (!cls) {
        return nil;
    }
    id o = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
    if (!o) {
        return nil;
    }
    if ([o respondsToSelector:@selector(initWithOptions:)]) {
        o = ((id(*)(id, SEL, unsigned long long))objc_msgSend)(o, @selector(initWithOptions:), 0ULL);
    } else {
        o = ((id(*)(id, SEL))objc_msgSend)(o, @selector(init));
    }
    if (!o) {
        return nil;
    }
    if ([o respondsToSelector:@selector(eventPort)]) {
        unsigned int p = ((unsigned int(*)(id, SEL))objc_msgSend)(o, @selector(eventPort));
        if (p == 0) {
            return nil;
        }
    }
    return o;
}

static SignalPair make_signal_pair(unsigned int symbolIndex, unsigned long long readyValue, unsigned long long freeValue, BOOL splitPorts) {
    SignalPair sp = {0};

    sp.sharedReady = new_shared_event();
    sp.sharedFree = splitPorts ? new_shared_event() : sp.sharedReady;
    if (!sp.sharedReady || !sp.sharedFree) {
        return sp;
    }

    sp.readyEvent = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
        CSig,
        @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
        readyValue,
        symbolIndex,
        5,
        sp.sharedReady
    );
    sp.freeEvent = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
        CSig,
        @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
        freeValue,
        symbolIndex,
        4,
        sp.sharedFree
    );

    if ([sp.readyEvent respondsToSelector:@selector(setAgentMask:)]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(sp.readyEvent, @selector(setAgentMask:), 1ULL);
    }
    if ([sp.freeEvent respondsToSelector:@selector(setAgentMask:)]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(sp.freeEvent, @selector(setAgentMask:), 1ULL);
    }

    if (sp.readyEvent && sp.freeEvent) {
        sp.events = @[sp.readyEvent, sp.freeEvent];
    }
    return sp;
}

static void print_vec3(const char *label, const float *v) {
    printf("%s [%.6f, %.6f, %.6f]\n", label, v[0], v[1], v[2]);
}

static BOOL experiment1_manual_shared_iosurface(void) {
    printf("\n=== Experiment 1: Manual IOSurface Sharing (_ANEInMemoryModel) ===\n");

    const int channels = 64;
    const int seq = 16;
    const int count = channels * seq;
    const size_t bytes = (size_t)count * sizeof(float);

    NSString *milConv = mil_gen_conv(channels, channels, seq);
    size_t wCount = (size_t)channels * (size_t)channels;
    float *wA = (float *)calloc(wCount, sizeof(float));
    float *wB = (float *)calloc(wCount, sizeof(float));
    for (int oc = 0; oc < channels; oc++) {
        for (int ic = 0; ic < channels; ic++) {
            size_t idx = (size_t)oc * (size_t)channels + (size_t)ic;
            if (oc == ic) {
                wA[idx] = 2.0f;
                wB[idx] = 1.0f;
            }
        }
    }
    NSData *weightsA = mil_build_weight_blob(wA, channels, channels);
    NSData *weightsB = mil_build_weight_blob(wB, channels, channels);

    InMemModel a = {0};
    InMemModel b = {0};
    char errBufA[512] = {0};
    char errBufB[512] = {0};
    if (!compile_load_inmem(milConv, weightsA, &a, errBufA, sizeof(errBufA))) {
        printf("  FAIL: compile/load model A (conv*2) failed: %s\n", errBufA);
        free(wB);
        free(wA);
        return NO;
    }
    if (!compile_load_inmem(milConv, weightsB, &b, errBufB, sizeof(errBufB))) {
        printf("  FAIL: compile/load model B (conv identity) failed: %s\n", errBufB);
        unload_inmem(&a);
        free(wB);
        free(wA);
        return NO;
    }

    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef midSurf = make_surface(bytes);  // Shared between A output and B input.
    IOSurfaceRef outSurf = make_surface(bytes);

    float *x = (float *)calloc((size_t)count, sizeof(float));
    float *y = (float *)calloc((size_t)count, sizeof(float));
    float *want = (float *)calloc((size_t)count, sizeof(float));
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;
    want[0] = 2.0f;
    want[1] = 4.0f;
    want[2] = 6.0f;
    write_surface_f32(inSurf, x, count);

    NSError *e = nil;
    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
    // A = conv*2
    BOOL okA = eval_inmem(&a, inSurf, midSurf, 0, &e);
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
    if (!okA) {
        printf("  FAIL: eval A failed: %s\n", err_desc(e));
        free(want);
        free(y);
        free(x);
        free(wB);
        free(wA);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        unload_inmem(&b);
        unload_inmem(&a);
        return NO;
    }

    e = nil;
    // B = identity conv
    BOOL okB = eval_inmem(&b, midSurf, outSurf, 0, &e);
    CFAbsoluteTime t2 = CFAbsoluteTimeGetCurrent();
    if (!okB) {
        printf("  FAIL: eval B failed: %s\n", err_desc(e));
        free(want);
        free(y);
        free(x);
        free(wB);
        free(wA);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        unload_inmem(&b);
        unload_inmem(&a);
        return NO;
    }

    read_surface_f32(outSurf, y, count);
    double d = max_abs_diff(y, want, count);
    print_vec3("  input:", x);
    print_vec3("  output:", y);
    print_vec3("  expect:", want);

    const double evalAms = (t1 - t0) * 1000.0;
    const double evalBms = (t2 - t1) * 1000.0;

    // Latency comparison: chain vs single eval (A only) over a short loop.
    const int iters = 100;
    double singleMs = 0.0;
    double chainMs = 0.0;
    for (int i = 0; i < iters; i++) {
        write_surface_f32(inSurf, x, count);
        CFAbsoluteTime s0 = CFAbsoluteTimeGetCurrent();
        (void)eval_inmem(&a, inSurf, midSurf, 0, NULL);
        CFAbsoluteTime s1 = CFAbsoluteTimeGetCurrent();
        (void)eval_inmem(&a, inSurf, midSurf, 0, NULL);
        (void)eval_inmem(&b, midSurf, outSurf, 0, NULL);
        CFAbsoluteTime s2 = CFAbsoluteTimeGetCurrent();
        singleMs += (s1 - s0) * 1000.0;
        chainMs += (s2 - s1) * 1000.0;
    }
    singleMs /= iters;
    chainMs /= iters;
    double ratio = singleMs > 0 ? chainMs / singleMs : 0.0;

    printf("  timings: A=%.4fms B=%.4fms single_avg=%.4fms chain_avg=%.4fms ratio=%.3fx\n",
           evalAms, evalBms, singleMs, chainMs, ratio);

    free(want);
    free(y);
    free(x);
    free(wB);
    free(wA);
    CFRelease(outSurf);
    CFRelease(midSurf);
    CFRelease(inSurf);
    unload_inmem(&b);
    unload_inmem(&a);

    if (d > 1e-3) {
        printf("  FAIL: numerical mismatch maxAbsDiff=%.6f\n", d);
        return NO;
    }

    printf("  PASS: shared-IOSurface A->B chain produced expected output without CPU copy\n");
    return YES;
}

static BOOL experiment2_two_model_client_chain(void) {
    printf("\n=== Experiment 2: Two-Model _ANEClient Chain Probe ===\n");

    NSString *modelAPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
    NSString *modelBPath = modelAPath;
    const char *rawA = getenv("ANE_CHAIN_MODEL_A");
    const char *rawB = getenv("ANE_CHAIN_MODEL_B");
    if (rawA && rawA[0]) modelAPath = [NSString stringWithUTF8String:rawA];
    if (rawB && rawB[0]) modelBPath = [NSString stringWithUTF8String:rawB];
    NSString *keyA = @"s";
    NSString *keyB = @"s";

    int count = 1024;
    const char *rawN = getenv("ANE_CHAIN_TENSOR_COUNT");
    if (rawN && rawN[0]) {
        count = atoi(rawN);
    }
    if (count <= 0) {
        count = 1024;
    }
    const size_t bytes = (size_t)count * sizeof(float);

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        printf("  FAIL: _ANEClient sharedConnection returned nil\n");
        return NO;
    }

    ClientModel a = {0};
    ClientModel b = {0};
    char errBuf[512] = {0};

    if (!compile_load_client_model(client, modelAPath, keyA, &a, errBuf, sizeof(errBuf))) {
        printf("  FAIL: model A compile/load failed: %s\n", errBuf);
        return NO;
    }
    if (!compile_load_client_model(client, modelBPath, keyB, &b, errBuf, sizeof(errBuf))) {
        printf("  FAIL: model B compile/load failed: %s\n", errBuf);
        unload_client_model(&a);
        return NO;
    }

    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef midSurf = make_surface(bytes);
    IOSurfaceRef outSurf = make_surface(bytes);
    IOSurfaceRef statsSurf = make_surface(4096);

    float *x = (float *)calloc((size_t)count, sizeof(float));
    float *y = (float *)calloc((size_t)count, sizeof(float));
    fill_input(x, count);
    write_surface_f32(inSurf, x, count);

    NSError *e = nil;
    BOOL ok = eval_client(&a, inSurf, midSurf, 0, &e);
    printf("  baseline A (in->mid): ok=%d err=%s\n", ok, err_desc(e));
    if (!ok) {
        print_reboot_hint_if_needed(e);
    }
    e = nil;
    BOOL ok2 = eval_client(&b, midSurf, outSurf, 0, &e);
    printf("  baseline B (mid->out): ok=%d err=%s\n", ok2, err_desc(e));
    if (!ok2) {
        print_reboot_hint_if_needed(e);
    }

    if (ok2) {
        read_surface_f32(outSurf, y, count);
        printf("  sample output[0..2]=[%.6f, %.6f, %.6f]\n", y[0], y[1], y[2]);
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id midObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), midSurf);

    id inBuf = ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(CBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), inObj, @0, 0);
    id outBuf = ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(CBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), midObj, @0, 0);
    id outSet = ((id(*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(COutSet, @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[outBuf]);

    SignalPair sp = make_signal_pair(0, 1, 2, YES);
    id chReq = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CChain,
        @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
        @[inBuf], @[outSet], @[], @[], @0, sp.events ? sp.events : @[], @1, @0, @0
    );

    BOOL valid = NO;
    if (chReq && [chReq respondsToSelector:@selector(validate)]) {
        @try {
            valid = ((BOOL(*)(id, SEL))objc_msgSend)(chReq, @selector(validate));
        } @catch (NSException *ex) {
            printf("  chaining request validate exception: %s\n", ex.reason.UTF8String);
        }
    }
    printf("  chaining request validate=%d\n", valid ? 1 : 0);

    NSError *prepErr = nil;
    BOOL prep = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client,
        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
        a.model,
        @{},
        chReq,
        kQoS,
        &prepErr
    );
    printf("  prepare(modelA): ok=%d err=%s\n", prep, err_desc(prepErr));

    BOOL enqA = NO;
    BOOL readyA = NO;
    BOOL readyB = NO;
    if (prep) {
        id enqObj = ((id(*)(Class, SEL, unsigned int, unsigned int, unsigned long long, BOOL, BOOL))objc_msgSend)(
            CEnq,
            @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            0,
            0,
            1,
            NO,
            NO
        );
        id ready = ((id(*)(Class, SEL, unsigned int, id, id, unsigned long long))objc_msgSend)(
            CReady,
            @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
            0,
            @[@0],
            @[@2],
            0
        );

        NSError *enqErr = nil;
        enqA = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
            a.model,
            enqObj,
            @{},
            kQoS,
            &enqErr
        );
        printf("  enqueue(modelA): ok=%d err=%s\n", enqA, err_desc(enqErr));

        NSError *readyErrA = nil;
        readyA = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
            a.model,
            ready,
            @{},
            kQoS,
            &readyErrA
        );
        printf("  ready(modelA): ok=%d err=%s\n", readyA, err_desc(readyErrA));

        NSError *readyErrB = nil;
        readyB = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
            b.model,
            ready,
            @{},
            kQoS,
            &readyErrB
        );
        printf("  ready(modelB): ok=%d err=%s\n", readyB, err_desc(readyErrB));
    }

    free(y);
    free(x);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(midSurf);
    CFRelease(inSurf);
    unload_client_model(&b);
    unload_client_model(&a);

    if (!ok || !ok2) {
        printf("  RESULT: client baseline not healthy; chaining not actionable in this run\n");
        return NO;
    }
    if (!prep) {
        printf("  RESULT: two-model prepare still failed (likely daemon/kernel invariant)\n");
        return NO;
    }
    if (!(enqA && (readyA || readyB))) {
        printf("  RESULT: prepare succeeded but enqueue/ready still blocked\n");
        return NO;
    }
    printf("  RESULT: two-model chaining path succeeded (prepare+enqueue+ready)\n");
    return YES;
}

static BOOL experiment3_iosurface_identity(void) {
    printf("\n=== Experiment 3: IOSurface Identity Test (_ANEClient) ===\n");

    NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
    NSString *key = @"s";
    int count = 1024;
    const char *raw = getenv("ANE_CHAIN_TENSOR_COUNT");
    if (raw && raw[0]) {
        count = atoi(raw);
    }
    if (count <= 0) count = 1024;
    size_t bytes = (size_t)count * sizeof(float);

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        printf("  FAIL: _ANEClient sharedConnection returned nil\n");
        return NO;
    }

    ClientModel m = {0};
    char errBuf[512] = {0};
    if (!compile_load_client_model(client, modelPath, key, &m, errBuf, sizeof(errBuf))) {
        printf("  FAIL: model compile/load failed: %s\n", errBuf);
        return NO;
    }

    IOSurfaceRef shared = make_surface(bytes);
    float *x = (float *)calloc((size_t)count, sizeof(float));
    float *y = (float *)calloc((size_t)count, sizeof(float));
    fill_input(x, count);
    write_surface_f32(shared, x, count);

    id ioObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), shared);

    id b0 = ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(CBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), ioObj, @0, 0);
    id b1 = ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(CBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), ioObj, @1, 0);

    printf("  buffer objects from same IOSurface: b0=%p b1=%p sameSurfaceObj=%p\n", b0, b1, ioObj);

    id req = make_basic_request(ioObj, ioObj, 0, 0, 0);

    NSError *e = nil;
    BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        client,
        @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
        m.model,
        req,
        YES,
        &e
    );
    printf("  map(in-place request): ok=%d err=%s\n", mapped, err_desc(e));
    if (!mapped) {
        print_reboot_hint_if_needed(e);
        free(y);
        free(x);
        CFRelease(shared);
        unload_client_model(&m);
        return NO;
    }

    e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client,
        @selector(evaluateWithModel:options:request:qos:error:),
        m.model,
        @{},
        req,
        kQoS,
        &e
    );
    printf("  evaluate(in-place request): ok=%d err=%s\n", ok, err_desc(e));

    ((void(*)(id, SEL, id, id))objc_msgSend)(
        client,
        @selector(unmapIOSurfacesWithModel:request:),
        m.model,
        req
    );

    if (ok) {
        read_surface_f32(shared, y, count);
        printf("  sample in-place output[0..2]=[%.6f, %.6f, %.6f]\n", y[0], y[1], y[2]);
    }

    free(y);
    free(x);
    CFRelease(shared);
    unload_client_model(&m);

    return ok;
}

static BOOL experiment4_signal_precommit(void) {
    printf("\n=== Experiment 4: Signal Pre-Commit Test (_ANEClient) ===\n");

    NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
    NSString *key = @"s";
    int count = 1024;
    const char *raw = getenv("ANE_CHAIN_TENSOR_COUNT");
    if (raw && raw[0]) count = atoi(raw);
    if (count <= 0) count = 1024;
    size_t bytes = (size_t)count * sizeof(float);

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        printf("  FAIL: _ANEClient sharedConnection returned nil\n");
        return NO;
    }

    ClientModel m = {0};
    char errBuf[512] = {0};
    if (!compile_load_client_model(client, modelPath, key, &m, errBuf, sizeof(errBuf))) {
        printf("  FAIL: model compile/load failed: %s\n", errBuf);
        return NO;
    }

    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef outSurf = make_surface(bytes);
    IOSurfaceRef statsSurf = make_surface(4096);
    float *x = (float *)calloc((size_t)count, sizeof(float));
    fill_input(x, count);
    write_surface_f32(inSurf, x, count);

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);

    id req = make_basic_request(inObj, outObj, 0, 0, 0);
    NSError *e = nil;
    BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        client,
        @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
        m.model,
        req,
        YES,
        &e
    );
    printf("  map: ok=%d err=%s\n", mapped, err_desc(e));
    if (!mapped) {
        print_reboot_hint_if_needed(e);
        free(x);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        unload_client_model(&m);
        return NO;
    }

    id inBuf = ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(CBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), inObj, @0, 0);
    id outBuf = ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(CBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), outObj, @0, 0);
    id outSet = ((id(*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(COutSet, @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[outBuf]);

    SignalPair sp = make_signal_pair(0, 1, 2, YES);
    id chReq = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CChain,
        @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
        @[inBuf], @[outSet], @[], @[], @0, sp.events ? sp.events : @[], @1, @0, @0
    );

    NSError *prepErr = nil;
    BOOL prep = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client,
        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
        m.model,
        @{},
        chReq,
        kQoS,
        &prepErr
    );
    printf("  prepare: ok=%d err=%s\n", prep, err_desc(prepErr));

    BOOL overall = prep;
    if (prep) {
        if ([sp.sharedReady respondsToSelector:@selector(setSignaledValue:)]) {
            ((void(*)(id, SEL, unsigned long long))objc_msgSend)(sp.sharedReady, @selector(setSignaledValue:), 1ULL);
        }
        if ([sp.sharedFree respondsToSelector:@selector(setSignaledValue:)]) {
            ((void(*)(id, SEL, unsigned long long))objc_msgSend)(sp.sharedFree, @selector(setSignaledValue:), 2ULL);
        }

        id enq = ((id(*)(Class, SEL, unsigned int, unsigned int, unsigned long long, BOOL, BOOL))objc_msgSend)(
            CEnq,
            @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            0,
            0,
            1,
            NO,
            NO
        );
        id ready = ((id(*)(Class, SEL, unsigned int, id, id, unsigned long long))objc_msgSend)(
            CReady,
            @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
            0,
            @[@0],
            @[@2],
            0
        );

        NSError *enqErr = nil;
        BOOL enqOK = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
            m.model,
            enq,
            @{},
            kQoS,
            &enqErr
        );
        printf("  order1 enqueue->ready enqueue=%d err=%s\n", enqOK, err_desc(enqErr));

        NSError *readyErr = nil;
        BOOL rd = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
            m.model,
            ready,
            @{},
            kQoS,
            &readyErr
        );
        printf("  order1 enqueue->ready ready=%d err=%s\n", rd, err_desc(readyErr));

        NSError *readyErr2 = nil;
        BOOL rd2 = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
            m.model,
            ready,
            @{},
            kQoS,
            &readyErr2
        );
        NSError *enqErr2 = nil;
        BOOL enq2 = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
            m.model,
            enq,
            @{},
            kQoS,
            &enqErr2
        );
        printf("  order2 ready->enqueue ready=%d err=%s enqueue=%d err=%s\n", rd2, err_desc(readyErr2), enq2, err_desc(enqErr2));

        overall = enqOK || rd || rd2 || enq2;
    }

    ((void(*)(id, SEL, id, id))objc_msgSend)(
        client,
        @selector(unmapIOSurfacesWithModel:request:),
        m.model,
        req
    );

    free(x);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(inSurf);
    unload_client_model(&m);

    if (!overall) {
        printf("  RESULT: pre-commit did not unblock enqueue/ready in this run\n");
    }
    return overall;
}

static void fill_input_iter(float *vals, int n, int iter) {
    for (int i = 0; i < n; i++) {
        float base = (float)(((i + (iter * 5)) % 37) - 18) * 0.125f;
        vals[i] = base + (float)(iter % 11) * 0.03125f;
    }
}

static void do_cpu_work(float *buf, int n, int rounds) {
    volatile float acc = gCpuSink;
    for (int r = 0; r < rounds; r++) {
        for (int i = 0; i < n; i++) {
            float x = buf[i] + (float)(r + 1) * 0.0001f;
            acc += x * 1.00001f + (float)(i & 7) * 0.0003f;
            buf[i] = x * 0.99999f;
        }
    }
    gCpuSink = acc;
}

static void make_expected_2x(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] * 2.0f;
    }
}

static BOOL setup_espresso_mid_frames(size_t bytes, int nFrames, id *outHolder, IOSurfaceRef *midFrames, char *errBuf, size_t errBufLen) {
    *outHolder = nil;
    for (int i = 0; i < nFrames; i++) {
        midFrames[i] = NULL;
    }

    void *h = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        snprintf(errBuf, errBufLen, "dlopen Espresso failed: %s", dlerror());
        return NO;
    }

    Class cls = NSClassFromString(@"EspressoANEIOSurface");
    if (!cls) {
        snprintf(errBuf, errBufLen, "EspressoANEIOSurface class not found");
        return NO;
    }

    id obj = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
    NSDictionary *props = @{
        @"IOSurfaceWidth": @(bytes),
        @"IOSurfaceHeight": @1,
        @"IOSurfaceBytesPerElement": @1,
        @"IOSurfaceBytesPerRow": @(bytes),
        @"IOSurfaceAllocSize": @(bytes),
        @"IOSurfacePixelFormat": @0,
    };
    NSSet *formats = [NSSet setWithObject:@0u];
    obj = ((id(*)(id, SEL, id, id))objc_msgSend)(obj, @selector(initWithIOSurfaceProperties:andPixelFormats:), props, formats);
    if (!obj) {
        snprintf(errBuf, errBufLen, "EspressoANEIOSurface init failed");
        return NO;
    }
    if ([obj respondsToSelector:@selector(resizeForMultipleAsyncBuffers:)]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(obj, @selector(resizeForMultipleAsyncBuffers:), (unsigned long long)nFrames);
    }

    for (int i = 0; i < nFrames; i++) {
        if (![obj respondsToSelector:@selector(ioSurfaceForMultiBufferFrame:)]) {
            snprintf(errBuf, errBufLen, "ioSurfaceForMultiBufferFrame missing");
            return NO;
        }
        IOSurfaceRef s = ((IOSurfaceRef(*)(id, SEL, unsigned long long))objc_msgSend)(obj, @selector(ioSurfaceForMultiBufferFrame:), (unsigned long long)i);
        if (!s) {
            snprintf(errBuf, errBufLen, "ioSurfaceForMultiBufferFrame(%d) returned nil", i);
            return NO;
        }
        CFRetain(s);
        midFrames[i] = s;
    }
    *outHolder = obj;
    return YES;
}

static BOOL run_multibuffer_rotation_benchmark(BOOL useEspresso) {
    int channels = env_int_or_default("ANE_CHAIN_BENCH_CHANNELS", 64);
    int seq = env_int_or_default("ANE_CHAIN_BENCH_SEQ", 16);
    int nFrames = env_int_or_default("ANE_CHAIN_BENCH_FRAMES", 3);
    int warmup = env_int_or_default("ANE_CHAIN_BENCH_WARMUP", 100);
    BOOL churnProbe = env_enabled_flag("ANE_CHAIN_BENCH_CHURN_PROBE");
    if (channels < 8) {
        channels = 8;
    }
    if (channels > 1024) {
        channels = 1024;
    }
    if (seq < 1) {
        seq = 1;
    }
    if (seq > 1024) {
        seq = 1024;
    }
    if (nFrames < 2) {
        nFrames = 2;
    }
    if (nFrames > 6) {
        nFrames = 6;
    }
    if (warmup < 0) {
        warmup = 0;
    }
    if (warmup > 2000) {
        warmup = 2000;
    }
    const int count = channels * seq;
    const size_t bytes = (size_t)count * sizeof(float);
    int iters = 300;
    const char *rawIters = getenv("ANE_CHAIN_BENCH_ITERS");
    if (rawIters && rawIters[0]) {
        int v = atoi(rawIters);
        if (v >= 64 && v <= 5000) {
            iters = v;
        }
    }

    NSString *milConv = mil_gen_conv(channels, channels, seq);
    size_t wCount = (size_t)channels * (size_t)channels;
    float *wA = (float *)calloc(wCount, sizeof(float));
    float *wB = (float *)calloc(wCount, sizeof(float));
    for (int oc = 0; oc < channels; oc++) {
        for (int ic = 0; ic < channels; ic++) {
            size_t idx = (size_t)oc * (size_t)channels + (size_t)ic;
            if (oc == ic) {
                wA[idx] = 2.0f;
                wB[idx] = 1.0f;
            }
        }
    }
    NSData *weightsA = mil_build_weight_blob(wA, channels, channels);
    NSData *weightsB = mil_build_weight_blob(wB, channels, channels);

    InMemModel a = {0};
    InMemModel b = {0};
    char errA[256] = {0};
    char errB[256] = {0};
    if (!compile_load_inmem(milConv, weightsA, &a, errA, sizeof(errA))) {
        printf("  FAIL: benchmark model A compile/load failed: %s\n", errA);
        free(wB);
        free(wA);
        return NO;
    }
    if (!compile_load_inmem(milConv, weightsB, &b, errB, sizeof(errB))) {
        printf("  FAIL: benchmark model B compile/load failed: %s\n", errB);
        unload_inmem(&a);
        free(wB);
        free(wA);
        return NO;
    }

    IOSurfaceRef *inFrames = (IOSurfaceRef *)calloc((size_t)nFrames, sizeof(IOSurfaceRef));
    IOSurfaceRef *outFrames = (IOSurfaceRef *)calloc((size_t)nFrames, sizeof(IOSurfaceRef));
    IOSurfaceRef *midFrames = (IOSurfaceRef *)calloc((size_t)nFrames, sizeof(IOSurfaceRef));
    if (!inFrames || !outFrames || !midFrames) {
        printf("  FAIL: frame allocation failed\n");
        free(midFrames);
        free(outFrames);
        free(inFrames);
        unload_inmem(&b);
        unload_inmem(&a);
        free(wB);
        free(wA);
        return NO;
    }
    for (int i = 0; i < nFrames; i++) {
        inFrames[i] = make_surface(bytes);
        outFrames[i] = make_surface(bytes);
    }

    id espressoHolder = nil;
    char espressoErr[256] = {0};
    if (useEspresso) {
        if (!setup_espresso_mid_frames(bytes, nFrames, &espressoHolder, midFrames, espressoErr, sizeof(espressoErr))) {
            printf("  FAIL: espresso mid-frame setup failed: %s\n", espressoErr);
            for (int i = 0; i < nFrames; i++) {
                if (outFrames[i]) CFRelease(outFrames[i]);
                if (inFrames[i]) CFRelease(inFrames[i]);
            }
            free(midFrames);
            free(outFrames);
            free(inFrames);
            unload_inmem(&b);
            unload_inmem(&a);
            free(wB);
            free(wA);
            return NO;
        }
    } else {
        for (int i = 0; i < nFrames; i++) {
            midFrames[i] = make_surface(bytes);
        }
    }

    float *inVals = (float *)calloc((size_t)count, sizeof(float));
    float *tmpOut = (float *)calloc((size_t)count, sizeof(float));
    float *expect = (float *)calloc((size_t)count, sizeof(float));
    if (!inVals || !tmpOut || !expect) {
        printf("  FAIL: benchmark allocation failed\n");
        free(expect);
        free(tmpOut);
        free(inVals);
        for (int i = 0; i < nFrames; i++) {
            if (!useEspresso && midFrames[i]) CFRelease(midFrames[i]);
            if (useEspresso && midFrames[i]) CFRelease(midFrames[i]);
            if (outFrames[i]) CFRelease(outFrames[i]);
            if (inFrames[i]) CFRelease(inFrames[i]);
        }
        free(midFrames);
        free(outFrames);
        free(inFrames);
        if (espressoHolder && [espressoHolder respondsToSelector:@selector(cleanup)]) {
            ((void(*)(id, SEL))objc_msgSend)(espressoHolder, @selector(cleanup));
        }
        unload_inmem(&b);
        unload_inmem(&a);
        free(wB);
        free(wA);
        return NO;
    }

    BOOL ok = YES;
    NSError *e = nil;

    printf("  bench_config mode=%s channels=%d seq=%d count=%d frames=%d warmup=%d iters=%d churn_probe=%d\n",
           useEspresso ? "espresso" : "raw-iosurface",
           channels, seq, count, nFrames, warmup, iters, churnProbe ? 1 : 0);

    // Warm cache and allocator state prior to timed sections.
    for (int i = 0; i < warmup; i++) {
        int f = i % nFrames;
        fill_input_iter(inVals, count, i);
        write_surface_f32(inFrames[f], inVals, count);
        e = nil;
        if (!eval_inmem(&a, inFrames[f], midFrames[f], 0, &e)) {
            printf("  FAIL: warmup eval A failed iter=%d err=%s\n", i, err_desc(e));
            ok = NO;
            break;
        }
        e = nil;
        if (!eval_inmem(&b, midFrames[f], outFrames[f], 0, &e)) {
            printf("  FAIL: warmup eval B failed iter=%d err=%s\n", i, err_desc(e));
            ok = NO;
            break;
        }
    }

    double singleMs = 0.0;
    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
    for (int i = 0; ok && i < iters; i++) {
        int f = i % nFrames;
        fill_input_iter(inVals, count, i);
        write_surface_f32(inFrames[f], inVals, count);
        e = nil;
        if (!eval_inmem(&a, inFrames[f], midFrames[f], 0, &e)) {
            printf("  FAIL: single benchmark eval A failed iter=%d err=%s\n", i, err_desc(e));
            ok = NO;
            break;
        }
    }
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
    singleMs = (double)(t1 - t0) * 1000.0;

    double seqMs = 0.0;
    if (ok) {
        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < iters; i++) {
            int f = i % nFrames;
            fill_input_iter(inVals, count, i);
            write_surface_f32(inFrames[f], inVals, count);
            e = nil;
            if (!eval_inmem(&a, inFrames[f], midFrames[f], 0, &e)) {
                printf("  FAIL: sequential eval A failed iter=%d err=%s\n", i, err_desc(e));
                ok = NO;
                break;
            }
            e = nil;
            if (!eval_inmem(&b, midFrames[f], outFrames[f], 0, &e)) {
                printf("  FAIL: sequential eval B failed iter=%d err=%s\n", i, err_desc(e));
                ok = NO;
                break;
            }
        }
        t1 = CFAbsoluteTimeGetCurrent();
        seqMs = (double)(t1 - t0) * 1000.0;
    }

    double rotMs = 0.0;
    double rotConcurrentMs = 0.0;
    double churnMs = 0.0;
    double maxDiff = 0.0;
    if (ok) {
        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i <= iters; i++) {
            if (i < iters) {
                int f = i % nFrames;
                fill_input_iter(inVals, count, i);
                write_surface_f32(inFrames[f], inVals, count);
                e = nil;
                if (!eval_inmem(&a, inFrames[f], midFrames[f], 0, &e)) {
                    printf("  FAIL: rotated eval A failed iter=%d err=%s\n", i, err_desc(e));
                    ok = NO;
                    break;
                }
            }
            if (i > 0) {
                int g = (i - 1) % nFrames;
                e = nil;
                if (!eval_inmem(&b, midFrames[g], outFrames[g], 0, &e)) {
                    printf("  FAIL: rotated eval B failed iter=%d err=%s\n", i - 1, err_desc(e));
                    ok = NO;
                    break;
                }
                read_surface_f32(outFrames[g], tmpOut, count);
                fill_input_iter(inVals, count, i - 1);
                make_expected_2x(inVals, expect, count);
                double d = max_abs_diff(tmpOut, expect, count);
                if (d > maxDiff) {
                    maxDiff = d;
                }
            }
        }
        t1 = CFAbsoluteTimeGetCurrent();
        rotMs = (double)(t1 - t0) * 1000.0;
    }

    if (ok) {
        dispatch_queue_t qA = dispatch_queue_create("ane.exp.rotation.a", DISPATCH_QUEUE_SERIAL);
        dispatch_queue_t qB = dispatch_queue_create("ane.exp.rotation.b", DISPATCH_QUEUE_SERIAL);
        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i <= iters; i++) {
            int cur = i % nFrames;
            int prev = (i - 1) % nFrames;
            if (prev < 0) {
                prev += nFrames;
            }

            if (i < iters) {
                fill_input_iter(inVals, count, i);
                write_surface_f32(inFrames[cur], inVals, count);
            }

            __block BOOL okAeval = YES;
            __block BOOL okBeval = YES;
            __block NSError *errAeval = nil;
            __block NSError *errBeval = nil;

            dispatch_group_t g = dispatch_group_create();
            if (i < iters) {
                dispatch_group_async(g, qA, ^{
                    okAeval = eval_inmem(&a, inFrames[cur], midFrames[cur], 0, &errAeval);
                });
            }
            if (i > 0) {
                dispatch_group_async(g, qB, ^{
                    okBeval = eval_inmem(&b, midFrames[prev], outFrames[prev], 0, &errBeval);
                });
            }
            dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

            if (i < iters && !okAeval) {
                printf("  FAIL: concurrent rotated eval A failed iter=%d err=%s\n", i, err_desc(errAeval));
                ok = NO;
                break;
            }
            if (i > 0 && !okBeval) {
                printf("  FAIL: concurrent rotated eval B failed iter=%d err=%s\n", i - 1, err_desc(errBeval));
                ok = NO;
                break;
            }

            if (i > 0) {
                read_surface_f32(outFrames[prev], tmpOut, count);
                fill_input_iter(inVals, count, i - 1);
                make_expected_2x(inVals, expect, count);
                double d = max_abs_diff(tmpOut, expect, count);
                if (d > maxDiff) {
                    maxDiff = d;
                }
            }
        }
        t1 = CFAbsoluteTimeGetCurrent();
        rotConcurrentMs = (double)(t1 - t0) * 1000.0;
    }

    if (ok && churnProbe) {
        int churnIters = iters < 80 ? iters : 80;
        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < churnIters; i++) {
            IOSurfaceRef inS = make_surface(bytes);
            IOSurfaceRef midS = make_surface(bytes);
            IOSurfaceRef outS = make_surface(bytes);
            if (!inS || !midS || !outS) {
                printf("  FAIL: churn surface alloc failed iter=%d\n", i);
                if (outS) CFRelease(outS);
                if (midS) CFRelease(midS);
                if (inS) CFRelease(inS);
                ok = NO;
                break;
            }
            fill_input_iter(inVals, count, i);
            write_surface_f32(inS, inVals, count);
            e = nil;
            if (!eval_inmem(&a, inS, midS, 0, &e)) {
                printf("  FAIL: churn eval A failed iter=%d err=%s\n", i, err_desc(e));
                ok = NO;
            } else {
                e = nil;
                if (!eval_inmem(&b, midS, outS, 0, &e)) {
                    printf("  FAIL: churn eval B failed iter=%d err=%s\n", i, err_desc(e));
                    ok = NO;
                }
            }
            CFRelease(outS);
            CFRelease(midS);
            CFRelease(inS);
            if (!ok) {
                break;
            }
        }
        t1 = CFAbsoluteTimeGetCurrent();
        churnMs = (double)(t1 - t0) * 1000.0 / (double)(iters < 80 ? iters : 80);
    }

    double singlePer = singleMs / (double)iters;
    double seqPer = seqMs / (double)iters;
    double rotPer = rotMs / (double)iters;
    double rotConPer = rotConcurrentMs / (double)iters;
    double seqVsSingle = singlePer > 0.0 ? seqPer / singlePer : 0.0;
    double rotVsSingle = singlePer > 0.0 ? rotPer / singlePer : 0.0;
    double rotConVsSingle = singlePer > 0.0 ? rotConPer / singlePer : 0.0;
    double speedup = rotPer > 0.0 ? seqPer / rotPer : 0.0;
    double speedupCon = rotConPer > 0.0 ? seqPer / rotConPer : 0.0;
    double overlapGain = rotConPer > 0.0 ? rotPer / rotConPer : 0.0;
    double churnPenalty = (churnProbe && seqPer > 0.0) ? churnMs / seqPer : 0.0;

    printf("  timings mode=%s single=%.4fms seq2stage=%.4fms rotated_serial=%.4fms rotated_concurrent=%.4fms seq/single=%.3fx rot_serial/single=%.3fx rot_concurrent/single=%.3fx speedup(seq/rot_serial)=%.3fx speedup(seq/rot_concurrent)=%.3fx overlap_gain=%.3fx",
           useEspresso ? "espresso" : "raw-iosurface",
           singlePer,
           seqPer,
           rotPer,
           rotConPer,
           seqVsSingle,
           rotVsSingle,
           rotConVsSingle,
           speedup,
           speedupCon,
           overlapGain);
    if (churnProbe) {
        printf(" seq2stage_churn=%.4fms churn_penalty=%.3fx", churnMs, churnPenalty);
    }
    printf(" maxDiff=%g\n", maxDiff);

    free(expect);
    free(tmpOut);
    free(inVals);
    for (int i = 0; i < nFrames; i++) {
        if (midFrames[i]) CFRelease(midFrames[i]);
        if (outFrames[i]) CFRelease(outFrames[i]);
        if (inFrames[i]) CFRelease(inFrames[i]);
    }
    free(midFrames);
    free(outFrames);
    free(inFrames);
    if (espressoHolder && [espressoHolder respondsToSelector:@selector(cleanup)]) {
        ((void(*)(id, SEL))objc_msgSend)(espressoHolder, @selector(cleanup));
    }
    unload_inmem(&b);
    unload_inmem(&a);
    free(wB);
    free(wA);

    if (!ok) {
        return NO;
    }
    if (maxDiff > 1e-3) {
        printf("  FAIL: rotated output mismatch maxAbsDiff=%g\n", maxDiff);
        return NO;
    }
    return YES;
}

static BOOL experiment5_multibuffer_rotation_raw(void) {
    printf("\n=== Experiment 5: Multi-Buffer Rotation (Raw IOSurface x3) ===\n");
    return run_multibuffer_rotation_benchmark(NO);
}

static BOOL experiment6_multibuffer_rotation_espresso(void) {
    printf("\n=== Experiment 6: Multi-Buffer Rotation (EspressoANEIOSurface x3) ===\n");
    return run_multibuffer_rotation_benchmark(YES);
}

static BOOL experiment7_async_selector_probe(void) {
    printf("\n=== Experiment 7: Async Eval Selector Probe ===\n");
    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    id inmemInstance = ((id(*)(Class, SEL))objc_msgSend)(CInMem, @selector(alloc));

    const char *clientSels[] = {
        "evaluateWithModel:options:request:qos:error:",
        "doEvaluateDirectWithModel:options:request:qos:error:",
        "doEvaluateWithModel:options:request:qos:completionEvent:error:",
        "doEvaluateWithModel:options:request:qos:error:",
    };
    const char *inmemSels[] = {
        "evaluateWithQoS:options:request:error:",
        "evaluateAsyncWithQoS:options:request:error:",
        "evaluateWithQoS:options:request:completionHandler:",
        "prepareChainingWithModel:options:chainingReq:qos:error:",
    };

    int foundClient = 0;
    for (size_t i = 0; i < sizeof(clientSels) / sizeof(clientSels[0]); i++) {
        SEL s = sel_registerName(clientSels[i]);
        BOOL ok = [client respondsToSelector:s];
        printf("  _ANEClient selector %s present=%d\n", clientSels[i], ok ? 1 : 0);
        if (ok) {
            foundClient++;
        }
    }

    int foundInmem = 0;
    for (size_t i = 0; i < sizeof(inmemSels) / sizeof(inmemSels[0]); i++) {
        SEL s = sel_registerName(inmemSels[i]);
        BOOL ok = [inmemInstance respondsToSelector:s];
        printf("  _ANEInMemoryModel selector %s present=%d\n", inmemSels[i], ok ? 1 : 0);
        if (ok) {
            foundInmem++;
        }
    }

    printf("  probe_summary client_found=%d inmem_found=%d\n", foundClient, foundInmem);
    return foundClient > 0;
}

static BOOL experiment8_gcd_cpu_ane_overlap(void) {
    printf("\n=== Experiment 8: GCD CPU+ANE Overlap Benchmark ===\n");

    int channels = env_int_or_default("ANE_CHAIN_OVERLAP_CHANNELS", 128);
    int seq = env_int_or_default("ANE_CHAIN_OVERLAP_SEQ", 16);
    int iters = env_int_or_default("ANE_CHAIN_OVERLAP_ITERS", 300);
    if (channels < 8) channels = 8;
    if (channels > 1024) channels = 1024;
    if (seq < 1) seq = 1;
    if (seq > 1024) seq = 1024;
    if (iters < 50) iters = 50;
    if (iters > 5000) iters = 5000;

    int count = channels * seq;
    size_t bytes = (size_t)count * sizeof(float);
    int roundsList[16] = {0};
    int roundsCount = 0;
    NSString *sweep = [NSString stringWithUTF8String:getenv("ANE_CHAIN_OVERLAP_SWEEP") ?: ""];
    if (sweep.length > 0) {
        NSArray<NSString *> *parts = [sweep componentsSeparatedByString:@","];
        for (NSString *p in parts) {
            if (roundsCount >= 16) break;
            int v = [p intValue];
            if (v < 0) v = 0;
            if (v > 2000) v = 2000;
            roundsList[roundsCount++] = v;
        }
    }
    if (roundsCount == 0) {
        int defaults[] = {0, 4, 8, 16, 32, 64, 128, 200};
        for (int i = 0; i < 8; i++) {
            roundsList[roundsCount++] = defaults[i];
        }
    }

    printf("  overlap_config channels=%d seq=%d count=%d iters=%d sweep_count=%d\n", channels, seq, count, iters, roundsCount);

    NSString *csvPath = [NSString stringWithUTF8String:getenv("ANE_CHAIN_OVERLAP_CSV") ?: ""];
    FILE *csv = NULL;
    if (csvPath.length > 0) {
        csv = fopen(csvPath.UTF8String, "a+");
        if (csv) {
            fseek(csv, 0, SEEK_END);
            long sz = ftell(csv);
            if (sz == 0) {
                fprintf(csv, "experiment,channels,seq,count,iters,cpu_rounds,ane_ms,cpu_ms,serial_ms,overlap_ms,speedup,overlap_pct,ideal_eff\n");
                fflush(csv);
            }
        } else {
            printf("  WARN: failed to open overlap csv path=%s\n", csvPath.UTF8String);
        }
    }

    NSString *milConv = mil_gen_conv(channels, channels, seq);
    size_t wCount = (size_t)channels * (size_t)channels;
    float *w = (float *)calloc(wCount, sizeof(float));
    for (int c = 0; c < channels; c++) {
        w[(size_t)c * (size_t)channels + (size_t)c] = 2.0f;
    }
    NSData *weights = mil_build_weight_blob(w, channels, channels);

    InMemModel m = {0};
    char errBuf[256] = {0};
    if (!compile_load_inmem(milConv, weights, &m, errBuf, sizeof(errBuf))) {
        printf("  FAIL: overlap model compile/load failed: %s\n", errBuf);
        free(w);
        return NO;
    }

    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef outSurf = make_surface(bytes);
    float *inVals = (float *)calloc((size_t)count, sizeof(float));
    float *outVals = (float *)calloc((size_t)count, sizeof(float));
    float *cpuBuf = (float *)calloc((size_t)count, sizeof(float));
    if (!inSurf || !outSurf || !inVals || !outVals || !cpuBuf) {
        printf("  FAIL: overlap alloc failed\n");
        if (outSurf) CFRelease(outSurf);
        if (inSurf) CFRelease(inSurf);
        free(cpuBuf);
        free(outVals);
        free(inVals);
        unload_inmem(&m);
        free(w);
        return NO;
    }

    NSError *e = nil;
    for (int i = 0; i < 50; i++) {
        fill_input_iter(inVals, count, i);
        write_surface_f32(inSurf, inVals, count);
        e = nil;
        if (!eval_inmem(&m, inSurf, outSurf, 0, &e)) {
            printf("  FAIL: overlap warmup eval failed iter=%d err=%s\n", i, err_desc(e));
            CFRelease(outSurf);
            CFRelease(inSurf);
            free(cpuBuf);
            free(outVals);
            free(inVals);
            unload_inmem(&m);
            free(w);
            return NO;
        }
    }

    dispatch_queue_t aneQ = dispatch_queue_create("ane.eval.overlap", DISPATCH_QUEUE_SERIAL);
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block NSError *asyncErr = nil;

    BOOL anyGain = NO;
    for (int ri = 0; ri < roundsCount; ri++) {
        int cpuRounds = roundsList[ri];

        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < iters; i++) {
            fill_input_iter(inVals, count, i);
            write_surface_f32(inSurf, inVals, count);
            e = nil;
            if (!eval_inmem(&m, inSurf, outSurf, 0, &e)) {
                printf("  FAIL: ane-only eval failed iter=%d err=%s\n", i, err_desc(e));
                CFRelease(outSurf);
                CFRelease(inSurf);
                free(cpuBuf);
                free(outVals);
                free(inVals);
                unload_inmem(&m);
                free(w);
                return NO;
            }
        }
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
        double anePer = ((double)(t1 - t0) * 1000.0) / (double)iters;

        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < iters; i++) {
            fill_input_iter(inVals, count, i);
            memcpy(cpuBuf, inVals, bytes);
            do_cpu_work(cpuBuf, count, cpuRounds);
        }
        t1 = CFAbsoluteTimeGetCurrent();
        double cpuPer = ((double)(t1 - t0) * 1000.0) / (double)iters;

        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < iters; i++) {
            fill_input_iter(inVals, count, i);
            write_surface_f32(inSurf, inVals, count);
            e = nil;
            if (!eval_inmem(&m, inSurf, outSurf, 0, &e)) {
                printf("  FAIL: serial overlap eval failed iter=%d err=%s\n", i, err_desc(e));
                CFRelease(outSurf);
                CFRelease(inSurf);
                free(cpuBuf);
                free(outVals);
                free(inVals);
                unload_inmem(&m);
                free(w);
                return NO;
            }
            memcpy(cpuBuf, inVals, bytes);
            do_cpu_work(cpuBuf, count, cpuRounds);
        }
        t1 = CFAbsoluteTimeGetCurrent();
        double serialPer = ((double)(t1 - t0) * 1000.0) / (double)iters;

        t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < iters; i++) {
            fill_input_iter(inVals, count, i);
            write_surface_f32(inSurf, inVals, count);
            asyncErr = nil;
            dispatch_async(aneQ, ^{
                NSError *localErr = nil;
                BOOL okEval = eval_inmem(&m, inSurf, outSurf, 0, &localErr);
                if (!okEval) {
                    asyncErr = localErr;
                }
                dispatch_semaphore_signal(sem);
            });
            memcpy(cpuBuf, inVals, bytes);
            do_cpu_work(cpuBuf, count, cpuRounds);
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            if (asyncErr != nil) {
                printf("  FAIL: overlapped eval failed iter=%d err=%s\n", i, err_desc(asyncErr));
                CFRelease(outSurf);
                CFRelease(inSurf);
                free(cpuBuf);
                free(outVals);
                free(inVals);
                unload_inmem(&m);
                free(w);
                return NO;
            }
        }
        t1 = CFAbsoluteTimeGetCurrent();
        double overlapPer = ((double)(t1 - t0) * 1000.0) / (double)iters;

        double sum = anePer + cpuPer;
        double overlapPct = sum > 0.0 ? 1.0 - (overlapPer / sum) : 0.0;
        double serialSpeedup = overlapPer > 0.0 ? serialPer / overlapPer : 0.0;
        double ideal = fmax(anePer, cpuPer);
        double eff = overlapPer > 0.0 ? ideal / overlapPer : 0.0;

        if (serialSpeedup > 1.05) {
            anyGain = YES;
        }
        printf("  sweep cpuRounds=%d ane=%.4fms cpu=%.4fms serial=%.4fms overlap=%.4fms speedup=%.3fx overlap_pct=%.1f%% ideal_eff=%.3fx\n",
               cpuRounds, anePer, cpuPer, serialPer, overlapPer, serialSpeedup, overlapPct * 100.0, eff);
        if (csv) {
            fprintf(csv, "exp8,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    channels, seq, count, iters, cpuRounds,
                    anePer, cpuPer, serialPer, overlapPer, serialSpeedup, overlapPct * 100.0, eff);
            fflush(csv);
        }
    }

    fill_input_iter(inVals, count, 777);
    write_surface_f32(inSurf, inVals, count);
    e = nil;
    if (!eval_inmem(&m, inSurf, outSurf, 0, &e)) {
        printf("  FAIL: overlap verify eval failed err=%s\n", err_desc(e));
    } else {
        read_surface_f32(outSurf, outVals, count);
        printf("  verify output[0..2]=[%.6f, %.6f, %.6f]\n", outVals[0], outVals[1], outVals[2]);
    }
    printf("  overlap_probe cpu_sink=%f\n", gCpuSink);
    if (csv) {
        fclose(csv);
    }

    CFRelease(outSurf);
    CFRelease(inSurf);
    free(cpuBuf);
    free(outVals);
    free(inVals);
    unload_inmem(&m);
    free(w);

    return anyGain;
}

static BOOL experiment9_metal_shared_event_bridge_probe(void) {
    printf("\n=== Experiment 9: Metal Shared Event Bridge Probe ===\n");

    Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
    if (!sharedCls) {
        printf("  FAIL: IOSurfaceSharedEvent class missing\n");
        return NO;
    }
    id iosEv = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
    if ([iosEv respondsToSelector:@selector(initWithOptions:)]) {
        iosEv = ((id(*)(id, SEL, unsigned long long))objc_msgSend)(iosEv, @selector(initWithOptions:), 0ULL);
    } else {
        iosEv = ((id(*)(id, SEL))objc_msgSend)(iosEv, @selector(init));
    }
    if (!iosEv || ![iosEv respondsToSelector:@selector(eventPort)]) {
        printf("  FAIL: IOSurfaceSharedEvent init/eventPort unavailable\n");
        return NO;
    }
    unsigned int port = ((unsigned int(*)(id, SEL))objc_msgSend)(iosEv, @selector(eventPort));
    printf("  iosurface_shared_event port=%u\n", port);
    if (port == 0) {
        printf("  FAIL: shared event port is 0\n");
        return NO;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        printf("  FAIL: no Metal device\n");
        return NO;
    }
    SEL newWithPortSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
    if (![dev respondsToSelector:newWithPortSel]) {
        printf("  XFAIL: MTLDevice newSharedEventWithMachPort: selector missing\n");
        return NO;
    }
    id mtlEv = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newWithPortSel, port);
    if (!mtlEv) {
        printf("  FAIL: newSharedEventWithMachPort returned nil\n");
        return NO;
    }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    if (!cb) {
        printf("  FAIL: failed to create command buffer\n");
        return NO;
    }
    if (![cb respondsToSelector:@selector(encodeWaitForEvent:value:)]) {
        printf("  FAIL: command buffer lacks encodeWaitForEvent:value:\n");
        return NO;
    }

    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(__unused id<MTLCommandBuffer> b) {
        dispatch_semaphore_signal(done);
    }];
    ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb, @selector(encodeWaitForEvent:value:), mtlEv, 1ULL);
    [cb commit];

    if ([iosEv respondsToSelector:@selector(setSignaledValue:)]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(iosEv, @selector(setSignaledValue:), 1ULL);
    } else {
        printf("  FAIL: IOSurfaceSharedEvent lacks setSignaledValue:\n");
        return NO;
    }

    long wait = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)));
    if (wait != 0) {
        printf("  FAIL: Metal command buffer did not unblock from IOSurfaceSharedEvent signal\n");
        return NO;
    }

    printf("  PASS: Metal wait event unblocked via IOSurfaceSharedEvent mach port bridge\n");
    return YES;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (!ane_setup_classes()) {
            return 2;
        }

        int only = env_int_or_default("ANE_CHAIN_EXPERIMENT", 0);
        printf("run_config experiment=%d (0 means all)\n", only);

        BOOL ok1 = (only == 0 || only == 1) ? experiment1_manual_shared_iosurface() : YES;
        BOOL ok2 = (only == 0 || only == 2) ? experiment2_two_model_client_chain() : YES;
        BOOL ok3 = (only == 0 || only == 3) ? experiment3_iosurface_identity() : YES;
        BOOL ok4 = (only == 0 || only == 4) ? experiment4_signal_precommit() : YES;
        BOOL ok5 = (only == 0 || only == 5) ? experiment5_multibuffer_rotation_raw() : YES;
        BOOL ok6 = (only == 0 || only == 6) ? experiment6_multibuffer_rotation_espresso() : YES;
        BOOL ok7 = (only == 0 || only == 7) ? experiment7_async_selector_probe() : YES;
        BOOL ok8 = (only == 0 || only == 8) ? experiment8_gcd_cpu_ane_overlap() : YES;
        BOOL ok9 = (only == 0 || only == 9) ? experiment9_metal_shared_event_bridge_probe() : YES;

        printf("\n=== Summary ===\n");
        printf("  Experiment1 manual shared-IOSurface chain: %s\n", ok1 ? "PASS" : "FAIL");
        printf("  Experiment2 two-model _ANEClient chain:    %s\n", ok2 ? "PASS" : "FAIL");
        printf("  Experiment3 IOSurface identity test:       %s\n", ok3 ? "PASS" : "FAIL");
        printf("  Experiment4 signal pre-commit test:        %s\n", ok4 ? "PASS" : "FAIL");
        printf("  Experiment5 raw triple-buffer rotation:    %s\n", ok5 ? "PASS" : "FAIL");
        printf("  Experiment6 Espresso triple-buffer:        %s\n", ok6 ? "PASS" : "FAIL");
        printf("  Experiment7 async selector probe:          %s\n", ok7 ? "PASS" : "FAIL");
        printf("  Experiment8 GCD CPU+ANE overlap:           %s\n", ok8 ? "PASS" : "FAIL");
        printf("  Experiment9 Metal event bridge probe:      %s\n", ok9 ? "PASS" : "FAIL");

        if (only >= 1 && only <= 9) {
            BOOL ok = YES;
            switch (only) {
                case 1: ok = ok1; break;
                case 2: ok = ok2; break;
                case 3: ok = ok3; break;
                case 4: ok = ok4; break;
                case 5: ok = ok5; break;
                case 6: ok = ok6; break;
                case 7: ok = ok7; break;
                case 8: ok = ok8; break;
                case 9: ok = ok9; break;
            }
            return ok ? 0 : 1;
        }
        // Default success criterion for full run: at least manual chain passes.
        return ok1 ? 0 : 1;
    }
}
