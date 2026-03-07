// test_asymmetric_pipeline.m
// Experiments 10-12: Asymmetric Metal->ANE wait path and Espresso integration.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static const unsigned int kQoS = 21;

static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIO;
static Class CWait;
static Class CSharedEvents;
static Class CNSURL;

static void *gANEHandle;

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
}

static int env_int_or(const char *name, int defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) {
        return defv;
    }
    return atoi(s);
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

static id load_anef_constant_obj(const char *symbol) {
    if (gANEHandle == NULL || symbol == NULL || symbol[0] == '\0') {
        return nil;
    }
    void *sym = dlsym(gANEHandle, symbol);
    if (sym == NULL && symbol[0] != '_') {
        char buf[128] = {0};
        snprintf(buf, sizeof(buf), "_%s", symbol);
        sym = dlsym(gANEHandle, buf);
    }
    if (sym == NULL) {
        return nil;
    }
    return *((__unsafe_unretained id *)sym);
}

static BOOL setup_classes(void) {
    gANEHandle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!gANEHandle) {
        fprintf(stderr, "FAIL: dlopen AppleNeuralEngine failed: %s\n", dlerror());
        return NO;
    }
    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");
    CWait = NSClassFromString(@"_ANESharedWaitEvent");
    CSharedEvents = NSClassFromString(@"_ANESharedEvents");
    CNSURL = NSClassFromString(@"NSURL");
    if (!CClient || !CModel || !CReq || !CAIO || !CWait || !CSharedEvents || !CNSURL) {
        fprintf(stderr, "FAIL: class resolution failed\n");
        return NO;
    }
    return YES;
}

static BOOL compile_load_model(id client, NSString *path, NSString *key, id *outModel) {
    id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), path);
    id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, key);
    if (!model) {
        return NO;
    }

    NSError *err = nil;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
        client, @selector(compileModel:options:qos:error:), model, @{}, kQoS, &err);
    if (!ok) {
        fprintf(stderr, "compileModel failed: %s\n", err_desc(err));
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
        fprintf(stderr, "loadModel failed: %s\n", err_desc(err));
        return NO;
    }
    *outModel = model;
    return YES;
}

static id make_simple_request(id inObj, id outObj) {
    SEL sel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    if (![CReq respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq, sel, @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0);
}

static id make_wait_request(id inObj, id outObj, id waitSharedEvent, unsigned long long waitValue) {
    id waitEvent = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(
        CWait, @selector(waitEventWithValue:sharedEvent:), waitValue, waitSharedEvent);
    if (!waitEvent) {
        return nil;
    }
    id sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(
        CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[], @[waitEvent]);
    if (!sharedEvents) {
        return nil;
    }
    SEL sel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
    if (![CReq respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CReq, sel, @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0, sharedEvents, @1);
}

static BOOL map_request(id client, id model, id req, NSError **err) {
    return ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), model, req, YES, err);
}

static void unmap_request(id client, id model, id req) {
    ((void(*)(id, SEL, id, id))objc_msgSend)(client, @selector(unmapIOSurfacesWithModel:request:), model, req);
}

static BOOL eval_request(id client, id model, id req, NSDictionary *opts, NSError **err) {
    SEL sel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    if (![client respondsToSelector:sel]) {
        sel = @selector(evaluateWithModel:options:request:qos:error:);
    }
    return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client, sel, model, opts ?: @{}, req, kQoS, err);
}

static NSDictionary *make_wait_options(void) {
    NSMutableDictionary *opts = [NSMutableDictionary dictionary];
    id disableFenceKey = load_anef_constant_obj("kANEFDisableIOFencesUseSharedEventsKey");
    if (!disableFenceKey) {
        disableFenceKey = @"kANEFDisableIOFencesUseSharedEventsKey";
    }
    id fwSignalKey = load_anef_constant_obj("kANEFEnableFWToFWSignal");
    if (!fwSignalKey) {
        fwSignalKey = @"kANEFEnableFWToFWSignal";
    }
    opts[disableFenceKey] = @YES;
    opts[fwSignalKey] = @YES;
    return opts;
}

static id<MTLComputePipelineState> make_writer_pso(id<MTLDevice> dev, int gpuRounds, NSError **err) {
    NSString *src = [NSString stringWithFormat:
                     @"using namespace metal;\n"
                     "kernel void writer(device const float *srcv [[buffer(0)]], device float *dstv [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
                     "  float v = srcv[gid];\n"
                     "  for (uint i = 0; i < %d; i++) { v = fma(v, 1.000001f, 0.000001f); }\n"
                     "  dstv[gid] = v;\n"
                     "}\n", gpuRounds];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:err];
    if (!lib) {
        // MTLCompilerService can transiently disappear; retry once.
        usleep(100000);
        *err = nil;
        lib = [dev newLibraryWithSource:src options:nil error:err];
    }
    if (!lib) {
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"writer"];
    if (!fn) {
        return nil;
    }
    return [dev newComputePipelineStateWithFunction:fn error:err];
}

static id<MTLCommandBuffer> encode_writer_and_signal(
    id<MTLCommandQueue> cq,
    id<MTLComputePipelineState> pso,
    id<MTLBuffer> srcBuf,
    id<MTLBuffer> dstBuf,
    int count,
    id mtlEvent,
    unsigned long long signalValue
) {
    id<MTLCommandBuffer> cb = [cq commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:srcBuf offset:0 atIndex:0];
    [enc setBuffer:dstBuf offset:0 atIndex:1];
    MTLSize grid = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger tgW = pso.maxTotalThreadsPerThreadgroup;
    if (tgW > (NSUInteger)count) {
        tgW = (NSUInteger)count;
    }
    MTLSize tg = MTLSizeMake(tgW, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    if ([cb respondsToSelector:@selector(encodeSignalEvent:value:)]) {
        ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb, @selector(encodeSignalEvent:value:), mtlEvent, signalValue);
    }
    return cb;
}

static double now_ms(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000.0;
}

static BOOL read_espresso_frames(size_t bytes, IOSurfaceRef *outInSurf, IOSurfaceRef *outOutSurf, id *outInObj, id *outOutObj, id dev) {
    void *espresso = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW | RTLD_GLOBAL);
    if (!espresso) {
        fprintf(stderr, "XFAIL: Espresso not available: %s\n", dlerror());
        return NO;
    }
    Class cls = NSClassFromString(@"EspressoANEIOSurface");
    if (!cls) {
        fprintf(stderr, "XFAIL: EspressoANEIOSurface missing\n");
        return NO;
    }
    SEL initSel = sel_registerName("initWithIOSurfaceProperties:andPixelFormats:");
    if (![cls instancesRespondToSelector:initSel]) {
        fprintf(stderr, "XFAIL: Espresso init selector missing\n");
        return NO;
    }
    NSDictionary *props = @{
        @"IOSurfaceWidth": @(bytes),
        @"IOSurfaceHeight": @1,
        @"IOSurfaceBytesPerElement": @1,
        @"IOSurfaceBytesPerRow": @(bytes),
        @"IOSurfaceAllocSize": @(bytes),
        @"IOSurfacePixelFormat": @0,
    };
    NSSet *formats = [NSSet setWithObject:@0u];

    id inWrap = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
    inWrap = ((id(*)(id, SEL, id, id))objc_msgSend)(inWrap, initSel, props, formats);
    id outWrap = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
    outWrap = ((id(*)(id, SEL, id, id))objc_msgSend)(outWrap, initSel, props, formats);
    if (!inWrap || !outWrap) {
        fprintf(stderr, "XFAIL: Espresso object init failed\n");
        return NO;
    }
    SEL resizeSel = sel_registerName("resizeForMultipleAsyncBuffers:");
    if ([inWrap respondsToSelector:resizeSel]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(inWrap, resizeSel, 2ULL);
    }
    if ([outWrap respondsToSelector:resizeSel]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(outWrap, resizeSel, 2ULL);
    }
    SEL frameSel = sel_registerName("ioSurfaceForMultiBufferFrame:");
    IOSurfaceRef inSurf = [inWrap respondsToSelector:frameSel]
        ? ((IOSurfaceRef(*)(id, SEL, unsigned long long))objc_msgSend)(inWrap, frameSel, 0ULL)
        : NULL;
    IOSurfaceRef outSurf = [outWrap respondsToSelector:frameSel]
        ? ((IOSurfaceRef(*)(id, SEL, unsigned long long))objc_msgSend)(outWrap, frameSel, 0ULL)
        : NULL;
    if (!inSurf || !outSurf) {
        fprintf(stderr, "XFAIL: Espresso frame surfaces are nil\n");
        return NO;
    }

    SEL metalSel = sel_registerName("metalBufferWithDevice:multiBufferFrame:");
    if (![inWrap respondsToSelector:metalSel]) {
        fprintf(stderr, "XFAIL: Espresso metalBufferWithDevice:multiBufferFrame: missing\n");
        return NO;
    }
    id metalBuf = ((id(*)(id, SEL, id, unsigned long long))objc_msgSend)(inWrap, metalSel, dev, 0ULL);
    if (!metalBuf) {
        fprintf(stderr, "XFAIL: Espresso metalBufferWithDevice returned nil\n");
        return NO;
    }
    *outInSurf = inSurf;
    *outOutSurf = outSurf;
    *outInObj = inWrap;
    *outOutObj = outWrap;
    return YES;
}

static BOOL experiment10_asymmetric_pipeline(void) {
    printf("\n=== Experiment 10: Asymmetric Metal->ANE Pipeline ===\n");
    BOOL skipUnmap = YES;
    const char *rsu = getenv("ANE_ASYM_SKIP_UNMAP");
    if (rsu && rsu[0] == '0') {
        skipUnmap = NO;
    }
    int count = env_int_or("ANE_ASYM_COUNT", 1024);
    int gpuRounds = env_int_or("ANE_ASYM_GPU_ROUNDS", 200);
    if (count < 32) count = 32;
    if (count > 65536) count = 65536;
    if (gpuRounds < 0) gpuRounds = 0;
    if (gpuRounds > 10000) gpuRounds = 10000;

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        printf("  FAIL: _ANEClient sharedConnection nil\n");
        return NO;
    }
    printf("  step: client ok\n");

    NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
    const char *rawPath = getenv("ANE_CHAIN_MODEL_PATH");
    if (rawPath && rawPath[0]) {
        modelPath = [NSString stringWithUTF8String:rawPath];
    }

    id model = nil;
    if (!compile_load_model(client, modelPath, @"s", &model)) {
        printf("  FAIL: compile/load model failed\n");
        return NO;
    }
    printf("  step: model compile/load ok\n");

    size_t bytes = (size_t)count * sizeof(float);
    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef outSurf = make_surface(bytes);
    if (!inSurf || !outSurf) {
        printf("  FAIL: IOSurface allocation failed\n");
        return NO;
    }
    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    printf("  step: iosurfaces and io objects ready\n");

    float *srcVals = (float *)calloc((size_t)count, sizeof(float));
    float *seqOut = (float *)calloc((size_t)count, sizeof(float));
    float *asyncOut = (float *)calloc((size_t)count, sizeof(float));
    if (!srcVals || !seqOut || !asyncOut) {
        printf("  FAIL: allocation failed\n");
        return NO;
    }
    for (int i = 0; i < count; i++) {
        srcVals[i] = (float)(i + 1);
    }

    Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
    id waitShared = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
    waitShared = [waitShared respondsToSelector:@selector(initWithOptions:)]
        ? ((id(*)(id, SEL, unsigned long long))objc_msgSend)(waitShared, @selector(initWithOptions:), 0ULL)
        : ((id(*)(id, SEL))objc_msgSend)(waitShared, @selector(init));
    unsigned int waitPort = ((unsigned int(*)(id, SEL))objc_msgSend)(waitShared, @selector(eventPort));
    printf("  step: iosurface shared event port=%u\n", waitPort);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
    id mtlWaitEvent = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, waitPort);
    if (!dev || !mtlWaitEvent) {
        printf("  FAIL: Metal shared event bridge unavailable\n");
        return NO;
    }
    printf("  step: metal shared event bridge ok\n");
    unsigned long long base = 0ULL;
    unsigned long long asyncValue = base + 1;

    NSError *merr = nil;
    id<MTLComputePipelineState> pso = make_writer_pso(dev, gpuRounds, &merr);
    if (!pso) {
        printf("  FAIL: failed to create Metal writer PSO err=%s\n", err_desc(merr));
        return NO;
    }
    printf("  step: writer pso ready\n");
    id<MTLCommandQueue> cq = [dev newCommandQueue];
    id<MTLBuffer> srcBuf = [dev newBufferWithBytes:srcVals length:bytes options:MTLResourceStorageModeShared];

    IOSurfaceLock(inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(inSurf);
    id<MTLBuffer> inBuf = [dev newBufferWithBytesNoCopy:inBase length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    IOSurfaceUnlock(inSurf, 0, NULL);
    if (!inBuf || !srcBuf) {
        printf("  FAIL: Metal buffers unavailable\n");
        return NO;
    }
    printf("  step: metal buffers ready\n");

    NSDictionary *opts = make_wait_options();
    NSError *err = nil;
    id reqSimple = make_simple_request(inObj, outObj);
    if (!reqSimple || !map_request(client, model, reqSimple, &err)) {
        printf("  FAIL: map simple request err=%s\n", err_desc(err));
        return NO;
    }
    double t0 = now_ms();
    write_surface_f32(inSurf, srcVals, count);
    BOOL ok = eval_request(client, model, reqSimple, @{}, &err);
    double t1 = now_ms();
    if (!ok) {
        printf("  FAIL: baseline simple eval failed err=%s\n", err_desc(err));
        if (!skipUnmap) {
            unmap_request(client, model, reqSimple);
        }
        return NO;
    }
    read_surface_f32(outSurf, seqOut, count);
    if (!skipUnmap) {
        unmap_request(client, model, reqSimple);
    }
    double seqMs = t1 - t0;
    printf("  step: baseline simple eval ok\n");

    id reqAsync = make_wait_request(inObj, outObj, waitShared, asyncValue);
    err = nil;
    if (!reqAsync || !map_request(client, model, reqAsync, &err)) {
        printf("  FAIL: map asynchronous wait request err=%s\n", err_desc(err));
        return NO;
    }
    printf("  step: reqAsync mapped (value=%llu)\n", asyncValue);

    dispatch_queue_t aneQ = dispatch_queue_create("ane.asym.eval", DISPATCH_QUEUE_SERIAL);
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block BOOL asyncOK = NO;
    __block NSError *asyncErr = nil;
    t0 = now_ms();
    dispatch_async(aneQ, ^{
        NSError *localErr = nil;
        BOOL localOK = eval_request(client, model, reqAsync, opts, &localErr);
        asyncOK = localOK;
        asyncErr = localErr;
        dispatch_semaphore_signal(done);
    });
    usleep(200);
    id<MTLCommandBuffer> cbAsync = encode_writer_and_signal(cq, pso, srcBuf, inBuf, count, mtlWaitEvent, asyncValue);
    [cbAsync commit];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    [cbAsync waitUntilCompleted];
    t1 = now_ms();
    if (!asyncOK) {
        printf("  FAIL: asynchronous wait-eval failed err=%s\n", err_desc(asyncErr));
        if (!skipUnmap) {
            unmap_request(client, model, reqAsync);
        }
        return NO;
    }
    read_surface_f32(outSurf, asyncOut, count);
    if (!skipUnmap) {
        unmap_request(client, model, reqAsync);
    }
    double asyncMs = t1 - t0;

    double maxDiff = 0.0;
    for (int i = 0; i < count; i++) {
        double d = fabs((double)seqOut[i] - (double)asyncOut[i]);
        if (d > maxDiff) {
            maxDiff = d;
        }
    }
    double speedup = asyncMs > 0.0 ? seqMs / asyncMs : 0.0;
    BOOL pass = maxDiff < 1e-3;
    printf("  count=%d gpuRounds=%d seq_ms=%.4f asym_ms=%.4f speedup=%.3fx maxDiff=%g\n",
           count, gpuRounds, seqMs, asyncMs, speedup, maxDiff);
    printf("  output_seq[0..2]=[%.6f, %.6f, %.6f]\n", seqOut[0], seqOut[1], seqOut[2]);
    printf("  output_asym[0..2]=[%.6f, %.6f, %.6f]\n", asyncOut[0], asyncOut[1], asyncOut[2]);
    if (getenv("ANE_ASYM_HARD_EXIT")) {
        _exit(pass ? 0 : 1);
    }
    return pass;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static BOOL probe_espresso_metal_values(id wrap, id<MTLDevice> dev, id<MTLCommandQueue> cq,
                                      size_t bytes, float *out0, float *out1, float *out2) {
    if (!wrap || !dev || !cq || !out0 || !out1 || !out2) {
        return NO;
    }
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        return NO;
    }
    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        @autoreleasepool {
            struct {
                int ok;
                float v0;
                float v1;
                float v2;
            } res;
            memset(&res, 0, sizeof(res));
            SEL metalSel = sel_registerName("metalBufferWithDevice:multiBufferFrame:");
            id outMetalBuf = [wrap respondsToSelector:metalSel]
                ? ((id(*)(id, SEL, id, unsigned long long))objc_msgSend)(wrap, metalSel, dev, 0ULL)
                : nil;
            if (!outMetalBuf) {
                write(fds[1], &res, sizeof(res));
                close(fds[1]);
                _exit(2);
            }
            float *ptr = NULL;
            if ([outMetalBuf respondsToSelector:@selector(contents)]) {
                ptr = (float *)((id<MTLBuffer>)outMetalBuf).contents;
            }
            if (ptr) {
                res.ok = 1;
                res.v0 = ptr[0];
                res.v1 = ptr[1];
                res.v2 = ptr[2];
            } else {
                id<MTLBuffer> snap = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (snap) {
                    id<MTLCommandBuffer> cb = [cq commandBuffer];
                    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
                    [bl copyFromBuffer:(id<MTLBuffer>)outMetalBuf sourceOffset:0 toBuffer:snap destinationOffset:0 size:bytes];
                    [bl endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    float *sp = (float *)snap.contents;
                    if (sp) {
                        res.ok = 1;
                        res.v0 = sp[0];
                        res.v1 = sp[1];
                        res.v2 = sp[2];
                    }
                }
            }
            write(fds[1], &res, sizeof(res));
            close(fds[1]);
            _exit(res.ok ? 0 : 3);
        }
    }
    close(fds[1]);
    if (pid < 0) {
        close(fds[0]);
        return NO;
    }
    struct {
        int ok;
        float v0;
        float v1;
        float v2;
    } res;
    memset(&res, 0, sizeof(res));
    ssize_t n = read(fds[0], &res, sizeof(res));
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return NO;
    }
    if (n != (ssize_t)sizeof(res) || !WIFEXITED(status) || WEXITSTATUS(status) != 0 || !res.ok) {
        return NO;
    }
    *out0 = res.v0;
    *out1 = res.v1;
    *out2 = res.v2;
    return YES;
}

static BOOL parse_sample_metrics(const char *buf, double *signalUs, double *evalUs, double *totalUs) {
    if (!buf || !signalUs || !evalUs || !totalUs) {
        return NO;
    }
    const char *hit = NULL;
    const char *p = buf;
    while ((p = strstr(p, "SAMPLE signal_us=")) != NULL) {
        hit = p;
        p += 1;
    }
    if (!hit) {
        return NO;
    }
    double s = 0.0, e = 0.0, t = 0.0;
    if (sscanf(hit, "SAMPLE signal_us=%lf eval_us=%lf total_us=%lf", &s, &e, &t) != 3) {
        return NO;
    }
    *signalUs = s;
    *evalUs = e;
    *totalUs = t;
    return YES;
}

static BOOL run_latency_sample_subprocess(double *signalUs, double *evalUs, double *totalUs) {
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        return NO;
    }
    pid_t pid = fork();
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        setenv("ANE_ASYM_CHILD_SAMPLE", "1", 1);
        setenv("ANE_ASYM_NO_FORK", "1", 1);
        execl("./test_asymmetric_pipeline", "./test_asymmetric_pipeline", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    if (pid < 0) {
        close(fds[0]);
        return NO;
    }
    char out[32768];
    size_t used = 0;
    memset(out, 0, sizeof(out));
    while (used + 1 < sizeof(out)) {
        ssize_t n = read(fds[0], out + used, sizeof(out) - used - 1);
        if (n <= 0) {
            break;
        }
        used += (size_t)n;
    }
    out[used] = '\0';
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return NO;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return NO;
    }
    return parse_sample_metrics(out, signalUs, evalUs, totalUs);
}

static int run_latency_sample_child(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        if (!setup_classes()) {
            printf("SAMPLE_FAIL setup_classes\n");
            fflush(stdout);
            _exit(2);
        }

        id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
        if (!client) {
            printf("SAMPLE_FAIL client\n");
            fflush(stdout);
            _exit(2);
        }
        NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
        const char *rawPath = getenv("ANE_CHAIN_MODEL_PATH");
        if (rawPath && rawPath[0]) {
            modelPath = [NSString stringWithUTF8String:rawPath];
        }
        id model = nil;
        if (!compile_load_model(client, modelPath, @"s", &model)) {
            printf("SAMPLE_FAIL compile_load\n");
            fflush(stdout);
            _exit(2);
        }

        int count = 1024;
        size_t bytes = (size_t)count * sizeof(float);
        IOSurfaceRef inSurf = make_surface(bytes);
        IOSurfaceRef outSurf = make_surface(bytes);
        if (!inSurf || !outSurf) {
            printf("SAMPLE_FAIL iosurface\n");
            fflush(stdout);
            _exit(2);
        }
        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
        float *vals = (float *)calloc((size_t)count, sizeof(float));
        if (!vals) {
            printf("SAMPLE_FAIL alloc\n");
            fflush(stdout);
            _exit(2);
        }
        for (int i = 0; i < count; i++) {
            vals[i] = (float)(i + 1);
        }
        write_surface_f32(inSurf, vals, count);

        Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
        id waitShared = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
        waitShared = [waitShared respondsToSelector:@selector(initWithOptions:)]
            ? ((id(*)(id, SEL, unsigned long long))objc_msgSend)(waitShared, @selector(initWithOptions:), 0ULL)
            : ((id(*)(id, SEL))objc_msgSend)(waitShared, @selector(init));
        if (!waitShared) {
            printf("SAMPLE_FAIL wait_shared\n");
            fflush(stdout);
            _exit(2);
        }
        unsigned int waitPort = ((unsigned int(*)(id, SEL))objc_msgSend)(waitShared, @selector(eventPort));
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> cq = [dev newCommandQueue];
        SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
        if (!dev || !cq || ![dev respondsToSelector:newSharedSel]) {
            printf("SAMPLE_FAIL metal\n");
            fflush(stdout);
            _exit(2);
        }
        id mtlWaitEvent = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, waitPort);
        if (!mtlWaitEvent) {
            printf("SAMPLE_FAIL metal_event\n");
            fflush(stdout);
            _exit(2);
        }
        unsigned long long v = 1ULL;
        id reqWait = make_wait_request(inObj, outObj, waitShared, v);
        NSError *err = nil;
        if (!reqWait || !map_request(client, model, reqWait, &err)) {
            printf("SAMPLE_FAIL map %s\n", err_desc(err));
            fflush(stdout);
            _exit(2);
        }

        dispatch_queue_t aneQ = dispatch_queue_create("ane.lat.sample", DISPATCH_QUEUE_SERIAL);
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block BOOL ok = NO;
        __block double evalStartMs = 0.0;
        __block double evalEndMs = 0.0;
        dispatch_async(aneQ, ^{
            NSError *localErr = nil;
            NSDictionary *opts = make_wait_options();
            evalStartMs = now_ms();
            BOOL localOK = eval_request(client, model, reqWait, opts, &localErr);
            ok = localOK;
            evalEndMs = now_ms();
            dispatch_semaphore_signal(done);
        });
        usleep(200);
        double launchMs = now_ms();
        double sigMs = now_ms();
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        if ([cb respondsToSelector:@selector(encodeSignalEvent:value:)]) {
            ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb, @selector(encodeSignalEvent:value:), mtlWaitEvent, v);
        }
        [cb commit];
        [cb waitUntilCompleted];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (!ok) {
            printf("SAMPLE_FAIL eval\n");
            fflush(stdout);
            _exit(2);
        }
        double signalUs = (evalEndMs - sigMs) * 1000.0;
        double evalUs = (evalEndMs - evalStartMs) * 1000.0;
        double totalUs = (evalEndMs - launchMs) * 1000.0;
        printf("SAMPLE signal_us=%.3f eval_us=%.3f total_us=%.3f\n", signalUs, evalUs, totalUs);
        fflush(stdout);
        _exit(0);
    }
}

static BOOL experiment11_latency_characterization(void) {
    printf("\n=== Experiment 11: Metal->ANE Handoff Latency ===\n");
    BOOL skipUnmap = YES;
    const char *rsu = getenv("ANE_ASYM_SKIP_UNMAP");
    if (rsu && rsu[0] == '0') {
        skipUnmap = NO;
    }
    int iters = env_int_or("ANE_ASYM_LAT_ITERS", 200);
    int warmup = env_int_or("ANE_ASYM_LAT_WARMUP", 10);
    if (iters < 20) iters = 20;
    if (iters > 5000) iters = 5000;
    if (warmup < 0) warmup = 0;
    if (warmup > 500) warmup = 500;

    NSString *csvPath = [NSString stringWithUTF8String:getenv("ANE_ASYM_LAT_CSV") ?: ""];

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        printf("  FAIL: _ANEClient sharedConnection nil\n");
        return NO;
    }
    printf("  step: client ok\n");
    NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
    const char *rawPath = getenv("ANE_CHAIN_MODEL_PATH");
    if (rawPath && rawPath[0]) {
        modelPath = [NSString stringWithUTF8String:rawPath];
    }

    id model = nil;
    if (!compile_load_model(client, modelPath, @"s", &model)) {
        printf("  FAIL: compile/load model failed\n");
        return NO;
    }
    printf("  step: model compile/load ok\n");

    int count = 1024;
    size_t bytes = (size_t)count * sizeof(float);
    IOSurfaceRef inSurf = make_surface(bytes);
    IOSurfaceRef outSurf = make_surface(bytes);
    if (!inSurf || !outSurf) {
        printf("  FAIL: IOSurface allocation failed\n");
        return NO;
    }
    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    float *vals = (float *)calloc((size_t)count, sizeof(float));
    for (int i = 0; i < count; i++) vals[i] = (float)(i + 1);
    write_surface_f32(inSurf, vals, count);

    id reqSimple = make_simple_request(inObj, outObj);
    NSError *err = nil;
    if (!reqSimple || !map_request(client, model, reqSimple, &err)) {
        printf("  FAIL: map simple request failed err=%s\n", err_desc(err));
        return NO;
    }
    printf("  step: baseline request mapped\n");
    double aneBaseUs = 0.0;
    for (int i = 0; i < 20; i++) {
        double t0 = now_ms();
        err = nil;
        if (!eval_request(client, model, reqSimple, @{}, &err)) {
            printf("  FAIL: baseline eval failed err=%s\n", err_desc(err));
            if (!skipUnmap) {
                unmap_request(client, model, reqSimple);
            }
            return NO;
        }
        double t1 = now_ms();
        if (i >= 5) {
            aneBaseUs += (t1 - t0) * 1000.0;
        }
    }
    aneBaseUs /= 15.0;
    printf("  step: baseline ane_base_us=%.3f\n", aneBaseUs);
    if (!skipUnmap) {
        unmap_request(client, model, reqSimple);
    }

    Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
    if (!sharedCls) {
        printf("  FAIL: IOSurfaceSharedEvent class missing\n");
        return NO;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        printf("  FAIL: no Metal device\n");
        return NO;
    }
    id<MTLCommandQueue> cq = [dev newCommandQueue];
    if (!cq) {
        printf("  FAIL: no Metal command queue\n");
        return NO;
    }
    SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
    if (![dev respondsToSelector:newSharedSel]) {
        printf("  FAIL: newSharedEventWithMachPort missing\n");
        return NO;
    }
    printf("  step: shared event bridge selectors ready\n");
    unsigned long long base = 0ULL;
    printf("  step: wait options prepared\n");
    double *handoff = (double *)calloc((size_t)iters, sizeof(double));
    double *signalToReturn = (double *)calloc((size_t)iters, sizeof(double));
    if (!handoff || !signalToReturn) {
        printf("  FAIL: latency allocation failed\n");
        free(handoff);
        free(signalToReturn);
        return NO;
    }
    int samples = 0;
    (void)base;

    FILE *csv = NULL;
    if (csvPath.length > 0) {
        csv = fopen(csvPath.UTF8String, "a+");
        if (csv) {
            fseek(csv, 0, SEEK_END);
            if (ftell(csv) == 0) {
                fprintf(csv, "iter,signal_us,eval_us,total_us,handoff_us,ane_base_us\n");
            }
        }
    }

    printf("  step: entering latency loop iters=%d warmup=%d (exec-per-sample)\n", iters, warmup);
    int total = iters + warmup;
    for (int i = 0; i < total; i++) {
        double sampleSignalUs = 0.0;
        double sampleEvalUs = 0.0;
        double sampleTotalUs = 0.0;
        if (!run_latency_sample_subprocess(&sampleSignalUs, &sampleEvalUs, &sampleTotalUs)) {
            printf("  WARN: sample failed iter=%d\n", i);
            continue;
        }
        double handoffUs = sampleSignalUs;
        if (i >= warmup) {
            handoff[samples++] = handoffUs;
            signalToReturn[samples - 1] = sampleSignalUs;
            if (csv) {
                fprintf(csv, "%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        i, sampleSignalUs, sampleEvalUs, sampleTotalUs, handoffUs, aneBaseUs);
            }
        }
    }
    if (csv) {
        fclose(csv);
    }

    if (samples < 10) {
        printf("  FAIL: insufficient latency samples\n");
        free(handoff);
        free(signalToReturn);
        return NO;
    }
    qsort(handoff, (size_t)samples, sizeof(double), cmp_double);
    qsort(signalToReturn, (size_t)samples, sizeof(double), cmp_double);
    int i50 = (int)(0.50 * (double)(samples - 1));
    int i95 = (int)(0.95 * (double)(samples - 1));
    int i99 = (int)(0.99 * (double)(samples - 1));
    BOOL pass = YES;
    printf("  signal_to_return_us p50=%.3f p95=%.3f p99=%.3f max=%.3f n=%d\n",
           signalToReturn[i50], signalToReturn[i95], signalToReturn[i99], signalToReturn[samples - 1], samples);
    printf("  handoff_us p50=%.3f p95=%.3f p99=%.3f max=%.3f n=%d ane_base_us=%.3f\n",
           handoff[i50], handoff[i95], handoff[i99], handoff[samples - 1], samples, aneBaseUs);
    free(handoff);
    free(signalToReturn);
    if (getenv("ANE_ASYM_HARD_EXIT")) {
        _exit(pass ? 0 : 1);
    }
    return pass;
}

static BOOL experiment12_espresso_integration(void) {
    printf("\n=== Experiment 12: EspressoANEIOSurface Asymmetric Integration ===\n");
    BOOL skipUnmap = YES;
    const char *rsu = getenv("ANE_ASYM_SKIP_UNMAP");
    if (rsu && rsu[0] == '0') {
        skipUnmap = NO;
    }
    int count = env_int_or("ANE_ASYM_COUNT", 1024);
    int gpuRounds = env_int_or("ANE_ASYM_GPU_ROUNDS", 64);
    if (count < 32) count = 32;
    if (count > 65536) count = 65536;
    if (gpuRounds < 0) gpuRounds = 0;
    if (gpuRounds > 5000) gpuRounds = 5000;
    size_t bytes = (size_t)count * sizeof(float);

    id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (!client) {
        printf("  FAIL: _ANEClient sharedConnection nil\n");
        return NO;
    }
    printf("  step: client ok\n");
    NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
    const char *rawPath = getenv("ANE_CHAIN_MODEL_PATH");
    if (rawPath && rawPath[0]) {
        modelPath = [NSString stringWithUTF8String:rawPath];
    }
    id model = nil;
    if (!compile_load_model(client, modelPath, @"s", &model)) {
        printf("  FAIL: compile/load model failed\n");
        return NO;
    }
    printf("  step: model compile/load ok\n");

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> cq = [dev newCommandQueue];
    if (!dev || !cq) {
        printf("  FAIL: Metal device/queue unavailable\n");
        return NO;
    }
    IOSurfaceRef inSurf = NULL, outSurf = NULL;
    id inWrap = nil, outWrap = nil;
    if (!read_espresso_frames(bytes, &inSurf, &outSurf, &inWrap, &outWrap, dev)) {
        return NO;
    }
    printf("  step: espresso frames ready\n");

    float *src = (float *)calloc((size_t)count, sizeof(float));
    float *outA = (float *)calloc((size_t)count, sizeof(float));
    if (!src || !outA) {
        printf("  FAIL: allocation failed\n");
        return NO;
    }
    for (int i = 0; i < count; i++) src[i] = (float)(i + 1);

    write_surface_f32(inSurf, src, count);
    float *zeros = (float *)calloc((size_t)count, sizeof(float));
    if (zeros) {
        write_surface_f32(outSurf, zeros, count);
        free(zeros);
    }
    printf("  step: espresso frame0 populated in_id=%u out_id=%u\n", IOSurfaceGetID(inSurf), IOSurfaceGetID(outSurf));

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    id reqBase = make_simple_request(inObj, outObj);
    NSError *baseErr = nil;
    if (reqBase && map_request(client, model, reqBase, &baseErr)) {
        baseErr = nil;
        BOOL baseOK = eval_request(client, model, reqBase, @{}, &baseErr);
        float baseHead[3] = {0};
        read_surface_f32(outSurf, baseHead, 3);
        printf("  step: baseline espresso eval ok=%d err=%s out[0..2]=[%.6f, %.6f, %.6f]\n",
               baseOK ? 1 : 0, err_desc(baseErr), baseHead[0], baseHead[1], baseHead[2]);
        if (!skipUnmap) {
            unmap_request(client, model, reqBase);
        }
    } else {
        printf("  step: baseline espresso eval map failed err=%s\n", err_desc(baseErr));
    }

    Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
    id waitShared = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
    waitShared = [waitShared respondsToSelector:@selector(initWithOptions:)]
        ? ((id(*)(id, SEL, unsigned long long))objc_msgSend)(waitShared, @selector(initWithOptions:), 0ULL)
        : ((id(*)(id, SEL))objc_msgSend)(waitShared, @selector(init));
    unsigned int waitPort = ((unsigned int(*)(id, SEL))objc_msgSend)(waitShared, @selector(eventPort));
    SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
    id mtlWaitEvent = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, waitPort);
    unsigned long long base = [mtlWaitEvent respondsToSelector:@selector(signaledValue)]
        ? ((unsigned long long(*)(id, SEL))objc_msgSend)(mtlWaitEvent, @selector(signaledValue))
        : 0ULL;
    unsigned long long v = base + 1;
    printf("  step: shared event bridge ready port=%u value=%llu\n", waitPort, v);

    id reqWait = make_wait_request(inObj, outObj, waitShared, v);
    NSError *err = nil;
    if (!reqWait || !map_request(client, model, reqWait, &err)) {
        printf("  FAIL: map wait request failed err=%s\n", err_desc(err));
        return NO;
    }
    printf("  step: reqWait mapped\n");

    dispatch_queue_t aneQ = dispatch_queue_create("ane.espresso.eval", DISPATCH_QUEUE_SERIAL);
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block BOOL ok = NO;
    __block NSError *evalErr = nil;
    NSDictionary *opts = make_wait_options();
    dispatch_async(aneQ, ^{
        NSError *localErr = nil;
        BOOL localOK = eval_request(client, model, reqWait, opts, &localErr);
        ok = localOK;
        evalErr = localErr;
        dispatch_semaphore_signal(done);
    });
    printf("  step: ane wait eval dispatched\n");
    usleep(200);
    id<MTLCommandBuffer> cb = [cq commandBuffer];
    if ([cb respondsToSelector:@selector(encodeSignalEvent:value:)]) {
        ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb, @selector(encodeSignalEvent:value:), mtlWaitEvent, v);
    }
    [cb commit];
    printf("  step: metal blit+signal committed\n");
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    [cb waitUntilCompleted];
    printf("  step: ane eval and metal signal completed\n");
    if (!ok) {
        printf("  FAIL: Espresso asymmetric eval failed err=%s\n", err_desc(evalErr));
        if (!skipUnmap) {
            unmap_request(client, model, reqWait);
        }
        return NO;
    }
    printf("  step: reading asymmetric output\n");
    read_surface_f32(outSurf, outA, count);
    printf("  step: asymmetric output read\n");
    if (!skipUnmap) {
        unmap_request(client, model, reqWait);
    }

    double maxAbs = fmax(fabs((double)outA[0]), fmax(fabs((double)outA[1]), fabs((double)outA[2])));
    float mz0 = 0.0f, mz1 = 0.0f, mz2 = 0.0f;
    BOOL doMetalProbe = NO;
    const char *rmp = getenv("ANE_ASYM_ESPRESSO_METAL_PROBE");
    if (rmp && rmp[0] == '1') {
        doMetalProbe = YES;
    }
    BOOL metalProbeOK = NO;
    if (doMetalProbe) {
        metalProbeOK = probe_espresso_metal_values(outWrap, dev, cq, bytes, &mz0, &mz1, &mz2);
    }
    double metalDiff = 0.0;
    if (doMetalProbe && metalProbeOK) {
        metalDiff = fmax(fabs((double)mz0 - (double)outA[0]),
                         fmax(fabs((double)mz1 - (double)outA[1]), fabs((double)mz2 - (double)outA[2])));
    }
    BOOL pass = isfinite(maxAbs) && maxAbs > 0.0;
    if (doMetalProbe) {
        pass = pass && metalProbeOK && isfinite(metalDiff) && metalDiff < 1e-3;
    }
    printf("  espresso output_async[0..2]=[%.6f, %.6f, %.6f] probe_enabled=%d metal_probe_ok=%d metal[0..2]=[%.6f, %.6f, %.6f] maxAbs3=%g metalDiff=%g\n",
           outA[0], outA[1], outA[2], doMetalProbe ? 1 : 0, metalProbeOK ? 1 : 0, mz0, mz1, mz2, maxAbs, metalDiff);
    if (getenv("ANE_ASYM_HARD_EXIT")) {
        _exit(pass ? 0 : 1);
    }
    return pass;
}

static int run_all(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        if (!setup_classes()) {
            return 2;
        }
        int only = env_int_or("ANE_ASYM_EXPERIMENT", 0);
        if (only == 10 || only == 11 || only == 12) {
            setenv("ANE_ASYM_HARD_EXIT", "1", 1);
        }
        printf("run_config asym_experiment=%d (0=all,10/11/12)\n", only);

        BOOL ok10 = (only == 0 || only == 10) ? experiment10_asymmetric_pipeline() : YES;
        BOOL ok11 = (only == 0 || only == 11) ? experiment11_latency_characterization() : YES;
        BOOL ok12 = (only == 0 || only == 12) ? experiment12_espresso_integration() : YES;

        printf("\n=== Asymmetric Summary ===\n");
        printf("  Experiment10 asymmetric pipeline: %s\n", ok10 ? "PASS" : "FAIL");
        printf("  Experiment11 handoff latency:     %s\n", ok11 ? "PASS" : "FAIL");
        printf("  Experiment12 Espresso integration:%s\n", ok12 ? "PASS" : "FAIL");
        int rc = (ok10 && ok11 && ok12) ? 0 : 1;
        if (only == 10) rc = ok10 ? 0 : 1;
        if (only == 11) rc = ok11 ? 0 : 1;
        if (only == 12) rc = ok12 ? 0 : 1;
        return rc;
    }
}

int main(void) {
    const char *childSample = getenv("ANE_ASYM_CHILD_SAMPLE");
    if (childSample && childSample[0] == '1') {
        return run_latency_sample_child();
    }

    const char *noFork = getenv("ANE_ASYM_NO_FORK");
    if (noFork && noFork[0] == '1') {
        return run_all();
    }

    pid_t pid = fork();
    if (pid == 0) {
        setenv("ANE_ASYM_NO_FORK", "1", 1);
        int rc = run_all();
        _exit(rc);
    }
    if (pid < 0) {
        fprintf(stderr, "fork failed\n");
        return 2;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "waitpid failed\n");
        return 2;
    }
    if (WIFSIGNALED(status)) {
        printf("XFAIL: child crashed with signal %d in asymmetric pipeline test\n", WTERMSIG(status));
        return 1;
    }
    if (!WIFEXITED(status)) {
        printf("XFAIL: child did not exit cleanly\n");
        return 1;
    }
    int code = WEXITSTATUS(status);
    if (code != 0) {
        printf("XFAIL: child exited with code %d\n", code);
    }
    return code;
}
