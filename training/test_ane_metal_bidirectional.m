// test_ane_metal_bidirectional.m
// Crash-isolated probes for bidirectional ANE<->Metal shared-event handoff.
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
#include <sys/wait.h>
#include <unistd.h>

static const unsigned int kQoS = 21;

static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIO;
static Class CSig;
static Class CWait;
static Class CSharedEvents;
static Class CNSURL;
static char kReqCompletionAssocKey;

#define TRACEF(fmt, ...) do { \
    fprintf(stderr, "[bidir] " fmt "\n", ##__VA_ARGS__); \
    fflush(stderr); \
} while (0)

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

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
}

static id load_anef_constant_obj(void *aneHandle, const char *symbol) {
    if (aneHandle == NULL || symbol == NULL || symbol[0] == '\0') {
        return nil;
    }
    void *sym = dlsym(aneHandle, symbol);
    if (sym == NULL && symbol[0] != '_') {
        char buf[128] = {0};
        snprintf(buf, sizeof(buf), "_%s", symbol);
        sym = dlsym(aneHandle, buf);
    }
    if (sym == NULL) {
        return nil;
    }
    return *((__unsafe_unretained id *)sym);
}

static BOOL setup_classes(void **outAne) {
    void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!ane) {
        fprintf(stderr, "failed to dlopen AppleNeuralEngine: %s\n", dlerror());
        return NO;
    }
    *outAne = ane;

    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");
    CSig = NSClassFromString(@"_ANESharedSignalEvent");
    CWait = NSClassFromString(@"_ANESharedWaitEvent");
    CSharedEvents = NSClassFromString(@"_ANESharedEvents");
    CNSURL = NSClassFromString(@"NSURL");
    if (!CClient || !CModel || !CReq || !CAIO || !CSig || !CWait || !CSharedEvents || !CNSURL) {
        fprintf(stderr, "failed to resolve ANE classes\n");
        return NO;
    }
    return YES;
}

static BOOL compile_load_model(id client, NSString *path, NSString *key, id *outModel) {
    id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), path);
    id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, key);
    if (!model) return NO;

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

static id make_full_request(id inObj, id outObj, id sharedEvents) {
    SEL sel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
    if (![CReq respondsToSelector:sel]) return nil;
    unsigned long long txnHandle = 1ULL;
    const char *rawTxn = getenv("ANE_BIDIR_TXN_HANDLE");
    if (rawTxn && rawTxn[0]) {
        unsigned long long v = strtoull(rawTxn, NULL, 10);
        txnHandle = v;
    }
    id txn = txnHandle == 0 ? nil : @(txnHandle);
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        sel,
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0, sharedEvents, txn
    );
}

static BOOL eval_direct(id client, id model, id req, id opts, NSError **err) {
    BOOL useDirect = NO;
    const char *raw = getenv("ANE_BIDIR_USE_DIRECT");
    if (raw && raw[0] == '1') {
        useDirect = YES;
    }
    SEL sel = @selector(evaluateWithModel:options:request:qos:error:);
    if (useDirect && [client respondsToSelector:@selector(doEvaluateDirectWithModel:options:request:qos:error:)]) {
        sel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    }
    return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client, sel, model, opts ?: @{}, req, kQoS, err
    );
}

static void trace_completion_selectors(id obj, const char *label) {
    if (!obj) {
        TRACEF("%s selector dump: nil", label ? label : "obj");
        return;
    }
    Class cls = object_getClass(obj);
    TRACEF("%s selector dump class=%s", label ? label : "obj", class_getName(cls));
    while (cls) {
        unsigned int count = 0;
        Method *methods = class_copyMethodList(cls, &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            const char *name = sel_getName(s);
            if (name && (strstr(name, "completionEvent") || strstr(name, "doEvaluateWithModel"))) {
                TRACEF("  %s::%s", class_getName(cls), name);
            }
        }
        free(methods);
        cls = class_getSuperclass(cls);
    }
}

static id resolve_virtual_client(id client) {
    if (!client) {
        return nil;
    }
    id v = nil;
    @try {
        if ([client respondsToSelector:@selector(virtualClient)]) {
            v = ((id(*)(id, SEL))objc_msgSend)(client, @selector(virtualClient));
            if (v) return v;
        }
    } @catch (__unused NSException *e) {}

    @try {
        v = [client valueForKey:@"virtualClient"];
        if (v) return v;
    } @catch (__unused NSException *e) {}

    Class cls = object_getClass(client);
    while (cls) {
        unsigned int count = 0;
        Ivar *ivars = class_copyIvarList(cls, &count);
        for (unsigned int i = 0; i < count; i++) {
            Ivar iv = ivars[i];
            const char *name = ivar_getName(iv);
            if (!name) continue;
            if (strstr(name, "virtualClient") || strstr(name, "virtual_client")) {
                id cand = object_getIvar(client, iv);
                if (cand) {
                    free(ivars);
                    return cand;
                }
            }
        }
        free(ivars);
        cls = class_getSuperclass(cls);
    }
    return nil;
}

static BOOL try_eval_with_completion_target(id target, SEL sel, id model, id opts, id req,
                                            unsigned int qos, id completionObj, NSError **err, BOOL *outOK) {
    if (!target || !sel || !outOK) {
        return NO;
    }
    @try {
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, id, NSError **))objc_msgSend)(
            target, sel, model, opts, req, qos, completionObj, err);
        *outOK = ok;
        return YES;
    } @catch (NSException *e) {
        TRACEF("completion call threw exception selector=%s reason=%s", sel_getName(sel), e.reason.UTF8String ?: "");
        return NO;
    }
}

static int run_probe(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        TRACEF("start probe");

        void *ane = NULL;
        if (!setup_classes(&ane)) {
            TRACEF("setup_classes failed");
            return 2;
        }
        TRACEF("classes resolved");

        NSString *mode = [NSString stringWithUTF8String:getenv("ANE_BIDIR_MODE") ?: "ane_to_metal"];
        int iters = 1000;
        const char *ri = getenv("ANE_BIDIR_ITERS");
        if (ri && ri[0]) {
            int v = atoi(ri);
            if (v > 10 && v <= 10000) iters = v;
        }
        int warmup = 10;
        const char *rw = getenv("ANE_BIDIR_WARMUP");
        if (rw && rw[0]) {
            int v = atoi(rw);
            if (v >= 0 && v <= 1000) warmup = v;
        }
        BOOL setDisableFence = YES;
        const char *rdf = getenv("ANE_BIDIR_DISABLE_FENCES");
        if (rdf && rdf[0] == '0') {
            setDisableFence = NO;
        }
        BOOL setFWSignal = YES;
        const char *rfs = getenv("ANE_BIDIR_FW_SIGNAL");
        if (rfs && rfs[0] == '0') {
            setFWSignal = NO;
        }
        BOOL skipCleanup = YES;
        const char *rsc = getenv("ANE_BIDIR_SKIP_CLEANUP");
        if (rsc && rsc[0] == '0') {
            skipCleanup = NO;
        }
        BOOL skipMetalWait = NO;
        const char *rsmw = getenv("ANE_BIDIR_SKIP_METAL_WAIT");
        if (rsmw && rsmw[0] == '1') {
            skipMetalWait = YES;
        }
        NSString *csvPath = [NSString stringWithUTF8String:getenv("ANE_BIDIR_CSV") ?: ""];

        id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
        if (!client) {
            TRACEF("sharedConnection nil");
            return 2;
        }
        TRACEF("sharedConnection ok");
        id virtualClient = resolve_virtual_client(client);
        if (!virtualClient) {
            Class CVirtualClient = NSClassFromString(@"_ANEVirtualClient");
            if (CVirtualClient && [CVirtualClient respondsToSelector:@selector(sharedConnection)]) {
                @try {
                    virtualClient = ((id(*)(Class, SEL))objc_msgSend)(CVirtualClient, @selector(sharedConnection));
                } @catch (__unused NSException *e) {}
            }
        }
        TRACEF("virtualClient available=%d", virtualClient ? 1 : 0);
        NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
        const char *rawPath = getenv("ANE_CHAIN_MODEL_PATH");
        if (rawPath && rawPath[0]) modelPath = [NSString stringWithUTF8String:rawPath];

        id model = nil;
        if (!compile_load_model(client, modelPath, @"s", &model)) {
            TRACEF("compile/load failed");
            return 2;
        }
        TRACEF("model compile/load ok");

        const int count = 1024;
        const size_t bytes = (size_t)count * sizeof(float);
        IOSurfaceRef inSurf = make_surface(bytes);
        IOSurfaceRef outSurf = make_surface(bytes);
        float *inVals = (float *)calloc((size_t)count, sizeof(float));
        float *outVals = (float *)calloc((size_t)count, sizeof(float));
        for (int i = 0; i < count; i++) inVals[i] = (float)(i + 1);
        write_surface_f32(inSurf, inVals, count);

        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
        TRACEF("io surfaces wrapped");

        Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
        if (!sharedCls) {
            TRACEF("IOSurfaceSharedEvent missing");
            return 2;
        }
        id evA = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
        evA = [evA respondsToSelector:@selector(initWithOptions:)]
            ? ((id(*)(id, SEL, unsigned long long))objc_msgSend)(evA, @selector(initWithOptions:), 0ULL)
            : ((id(*)(id, SEL))objc_msgSend)(evA, @selector(init));
        id evB = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
        evB = [evB respondsToSelector:@selector(initWithOptions:)]
            ? ((id(*)(id, SEL, unsigned long long))objc_msgSend)(evB, @selector(initWithOptions:), 0ULL)
            : ((id(*)(id, SEL))objc_msgSend)(evB, @selector(init));
        unsigned int portA = ((unsigned int(*)(id, SEL))objc_msgSend)(evA, @selector(eventPort));
        unsigned int portB = ((unsigned int(*)(id, SEL))objc_msgSend)(evB, @selector(eventPort));
        TRACEF("shared events created ports A=%u B=%u", portA, portB);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
        if (!dev || ![dev respondsToSelector:newSharedSel]) {
            TRACEF("Metal shared event API unavailable");
            return 2;
        }
        id mtlA = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, portA);
        id mtlB = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, portB);
        if (!mtlA || !mtlB) {
            TRACEF("newSharedEventWithMachPort failed");
            return 2;
        }
        TRACEF("metal shared events created");

        unsigned long long baseA = [mtlA respondsToSelector:@selector(signaledValue)] ? ((unsigned long long(*)(id, SEL))objc_msgSend)(mtlA, @selector(signaledValue)) : 0ULL;
        unsigned long long baseB = [mtlB respondsToSelector:@selector(signaledValue)] ? ((unsigned long long(*)(id, SEL))objc_msgSend)(mtlB, @selector(signaledValue)) : 0ULL;

        unsigned long long vA = baseA + 1;
        unsigned long long vB = baseB + 1;
        long long sigType = 5;
        const char *rst = getenv("ANE_BIDIR_SIG_TYPE");
        if (rst && rst[0]) {
            long long t = strtoll(rst, NULL, 10);
            if (t >= 0 && t <= 16) {
                sigType = t;
            }
        }
        unsigned int sigSymbol = 0;
        const char *rss = getenv("ANE_BIDIR_SIG_SYMBOL");
        if (rss && rss[0]) {
            unsigned long long s = strtoull(rss, NULL, 10);
            if (s <= 1024) {
                sigSymbol = (unsigned int)s;
            }
        }

        id sigA = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(CSig, @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:), vA, sigSymbol, sigType, evA);
        id waitB = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(CWait, @selector(waitEventWithValue:sharedEvent:), vB, evB);
        if (!sigA || !waitB) {
            TRACEF("failed to create signal/wait events");
            return 2;
        }
        TRACEF("ane signal/wait events created sigType=%lld sigSymbol=%u", sigType, sigSymbol);

        id sharedEvents = nil;
        if ([mode isEqualToString:@"ane_to_metal"] || [mode isEqualToString:@"ane_to_metal_block"]) {
            sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[sigA], @[]);
        } else if ([mode isEqualToString:@"metal_to_ane"]) {
            sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[], @[waitB]);
        } else {
            sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[sigA], @[waitB]);
        }

        id req = make_full_request(inObj, outObj, sharedEvents);
        if (!req) {
            TRACEF("request creation failed");
            return 2;
        }
        TRACEF("request created");
        dispatch_semaphore_t reqCompletionSem = nil;
        __block BOOL reqCompletionFired = NO;
        __block BOOL reqCompletionSuccess = NO;
        __block NSError *reqCompletionError = nil;
        BOOL reqCompletionInstalled = NO;
        if ([mode isEqualToString:@"ane_to_metal"]) {
            SEL setCompletionSel = NSSelectorFromString(@"setCompletionHandler:");
            if ([req respondsToSelector:setCompletionSel]) {
                reqCompletionSem = dispatch_semaphore_create(0);
                void (^handler)(BOOL, NSError *) = ^(BOOL success, NSError *error) {
                    reqCompletionFired = YES;
                    reqCompletionSuccess = success;
                    reqCompletionError = error;
                    dispatch_semaphore_signal(reqCompletionSem);
                };
                id copiedHandler = [handler copy];
                ((void(*)(id, SEL, id))objc_msgSend)(req, setCompletionSel, copiedHandler);
                objc_setAssociatedObject(req, &kReqCompletionAssocKey, copiedHandler, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                reqCompletionInstalled = YES;
                TRACEF("request completion handler installed for ane_to_metal");
            } else {
                TRACEF("request missing setCompletionHandler:");
            }
        }
        if ([req respondsToSelector:@selector(validate)]) {
            BOOL valid = ((BOOL(*)(id, SEL))objc_msgSend)(req, @selector(validate));
            if (!valid) {
                TRACEF("request validate=false");
                return 2;
            }
        }
        TRACEF("request validated");

        NSError *err = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), model, req, YES, &err);
        if (!ok) {
            TRACEF("map failed: %s", err_desc(err));
            return 2;
        }
        TRACEF("map ok");

        NSMutableDictionary *opts = [NSMutableDictionary dictionary];
        id disableFenceKey = load_anef_constant_obj(ane, "kANEFDisableIOFencesUseSharedEventsKey");
        if (!disableFenceKey) disableFenceKey = @"kANEFDisableIOFencesUseSharedEventsKey";
        if (setDisableFence) {
            opts[disableFenceKey] = @YES;
        }
        id fwKey = load_anef_constant_obj(ane, "kANEFEnableFWToFWSignal");
        if (!fwKey) fwKey = @"kANEFEnableFWToFWSignal";
        if (setFWSignal) {
            opts[fwKey] = @YES;
        }
        TRACEF("opts disable_fence=%d fw_signal=%d skip_cleanup=%d skip_metal_wait=%d warmup=%d iters=%d", setDisableFence ? 1 : 0, setFWSignal ? 1 : 0, skipCleanup ? 1 : 0, skipMetalWait ? 1 : 0, warmup, iters);

        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];

        if ([mode isEqualToString:@"ane_to_metal"] || [mode isEqualToString:@"ane_to_metal_block"] || [mode isEqualToString:@"latency"]) {
            ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb, @selector(encodeWaitForEvent:value:), mtlA, vA);
            [cb commit];
            TRACEF("metal wait command committed vA=%llu", vA);
        }

        if ([mode isEqualToString:@"metal_to_ane"] || [mode isEqualToString:@"latency"]) {
            id<MTLCommandBuffer> sig = [cq commandBuffer];
            if ([sig respondsToSelector:@selector(encodeSignalEvent:value:)]) {
                ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(sig, @selector(encodeSignalEvent:value:), mtlB, vB);
                [sig commit];
                TRACEF("metal signal command committed vB=%llu", vB);
            } else {
                TRACEF("encodeSignalEvent:value: unavailable");
            }
        }

        BOOL metalUnblocked = NO;
        double handoffUs = -1.0;
        if ([mode isEqualToString:@"latency"]) {
            NSMutableArray<NSNumber *> *samples = [NSMutableArray arrayWithCapacity:(NSUInteger)iters];
            FILE *csv = NULL;
            if (csvPath.length > 0) {
                csv = fopen(csvPath.UTF8String, "a+");
                if (csv != NULL) {
                    fseek(csv, 0, SEEK_END);
                    long sz = ftell(csv);
                    if (sz == 0) {
                        fprintf(csv, "mode,iter,ane_us,handoff_wait_us,total_us\n");
                    }
                }
            }
            for (int i = 0; i < iters; i++) {
                unsigned long long value = vA + (unsigned long long)i;
                id<MTLCommandBuffer> wcb = [cq commandBuffer];
                ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(wcb, @selector(encodeWaitForEvent:value:), mtlA, value);
                [wcb commit];

                id sigEvent = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(CSig, @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:), value, 0, 5, evA);
                id se = ((id(*)(Class, SEL, id, id))objc_msgSend)(CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[sigEvent], @[]);
                id rq = make_full_request(inObj, outObj, se);

                NSError *e2 = nil;
                CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
                BOOL okEval = eval_direct(client, model, rq, opts, &e2);
                CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
                if (!okEval) {
                    TRACEF("latency eval failed at iter=%d err=%s", i, err_desc(e2));
                    break;
                }
                [wcb waitUntilCompleted];
                CFAbsoluteTime t2 = CFAbsoluteTimeGetCurrent();
                double aneUs = (t1 - t0) * 1000000.0;
                double handoffWaitUs = (t2 - t1) * 1000000.0;
                double totalUs = (t2 - t0) * 1000000.0;
                if (i >= warmup) {
                    [samples addObject:@(handoffWaitUs)];
                    if (csv != NULL) {
                        fprintf(csv, "latency,%d,%.3f,%.3f,%.3f\n", i, aneUs, handoffWaitUs, totalUs);
                    }
                }
            }
            if (csv != NULL) {
                fclose(csv);
            }
            if (samples.count > 10) {
                NSArray<NSNumber *> *sorted = [samples sortedArrayUsingSelector:@selector(compare:)];
                NSUInteger n = sorted.count;
                double p50 = sorted[(NSUInteger)(0.50 * (n - 1))].doubleValue;
                double p95 = sorted[(NSUInteger)(0.95 * (n - 1))].doubleValue;
                double p99 = sorted[(NSUInteger)(0.99 * (n - 1))].doubleValue;
                double mx = sorted[n - 1].doubleValue;
                printf("latency_eval_us p50=%.3f p95=%.3f p99=%.3f max=%.3f n=%lu\n", p50, p95, p99, mx, (unsigned long)n);
            }
            metalUnblocked = YES;
            ok = YES;
        } else {
            CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
            err = nil;
            TRACEF("about to eval mode=%s", mode.UTF8String);
            if ([mode isEqualToString:@"ane_to_metal_block"]) {
                __block BOOL completionFired = NO;
                __block BOOL completionSuccess = NO;
                __block NSError *completionError = nil;
                dispatch_semaphore_t completionSem = dispatch_semaphore_create(0);
                void (^completionBlock)(BOOL, NSError *) = ^(BOOL success, NSError *error) {
                    completionFired = YES;
                    completionSuccess = success;
                    completionError = error;
                    dispatch_semaphore_signal(completionSem);
                };
                id completionObj = [completionBlock copy];
                SEL legacySel = @selector(doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:);
                SEL doSel = @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:);
                id fwdLegacy = nil;
                id fwdDo = nil;
                SEL fwdSel = @selector(forwardingTargetForSelector:);
                if ([client respondsToSelector:fwdSel]) {
                    @try {
                        fwdLegacy = ((id(*)(id, SEL, SEL))objc_msgSend)(client, fwdSel, legacySel);
                        fwdDo = ((id(*)(id, SEL, SEL))objc_msgSend)(client, fwdSel, doSel);
                    } @catch (__unused NSException *e) {}
                }
                if (fwdLegacy || fwdDo) {
                    TRACEF("forward targets legacy=%p do=%p", (__bridge void *)fwdLegacy, (__bridge void *)fwdDo);
                }
                struct {
                    id target;
                    SEL sel;
                    const char *name;
                } attempts[] = {
                    { virtualClient, legacySel, "virtualClient.legacy" },
                    { virtualClient, doSel,     "virtualClient.do" },
                    { fwdLegacy,     legacySel, "forward.legacy" },
                    { fwdDo,         doSel,     "forward.do" },
                    { client,       legacySel, "client.legacy" },
                    { client,       doSel,     "client.do" },
                };
                BOOL called = NO;
                for (size_t i = 0; i < sizeof(attempts)/sizeof(attempts[0]); i++) {
                    if (!attempts[i].target) continue;
                    TRACEF("attempt completion call via %s selector=%s", attempts[i].name, sel_getName(attempts[i].sel));
                    BOOL callOK = NO;
                    if (try_eval_with_completion_target(attempts[i].target, attempts[i].sel, model, opts, req, kQoS, completionObj, &err, &callOK)) {
                        ok = callOK;
                        called = YES;
                        TRACEF("completion call accepted via %s ok=%d err=%s", attempts[i].name, ok ? 1 : 0, err_desc(err));
                        break;
                    }
                }
                if (called) {
                    long waited = -1;
                    if (ok) {
                        waited = dispatch_semaphore_wait(
                            completionSem,
                            dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * NSEC_PER_SEC)));
                    }
                    printf("completion fired=%d success=%d waited=%ld evalErr=%s blockErr=%s\n",
                           completionFired ? 1 : 0,
                           completionSuccess ? 1 : 0,
                           waited,
                           err_desc(err),
                           err_desc(completionError));
                    if (!ok || waited != 0 || !completionFired || !completionSuccess) {
                        ok = NO;
                    }
                } else {
                    TRACEF("completion call path unavailable on both client and virtualClient");
                    trace_completion_selectors(client, "client");
                    trace_completion_selectors(virtualClient, "virtualClient");
                    ok = NO;
                }
            } else {
                ok = eval_direct(client, model, req, opts, &err);
            }
            CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
            handoffUs = (t1 - t0) * 1000000.0;
            printf("evaluate mode=%s ok=%d err=%s eval_us=%.3f\n", mode.UTF8String, ok ? 1 : 0, err_desc(err), handoffUs);
            TRACEF("eval returned ok=%d", ok ? 1 : 0);
            if ([mode isEqualToString:@"ane_to_metal"] && reqCompletionInstalled) {
                long waited = dispatch_semaphore_wait(
                    reqCompletionSem,
                    dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * NSEC_PER_SEC)));
                printf("request_completion fired=%d success=%d waited=%ld error=%s\n",
                       reqCompletionFired ? 1 : 0,
                       reqCompletionSuccess ? 1 : 0,
                       waited,
                       err_desc(reqCompletionError));
                if (waited != 0 || !reqCompletionFired || !reqCompletionSuccess) {
                    ok = NO;
                }
            }
            if ([evA respondsToSelector:@selector(signaledValue)]) {
                unsigned long long sigVal = ((unsigned long long(*)(id, SEL))objc_msgSend)(evA, @selector(signaledValue));
                printf("eventA.signaledValue=%llu target=%llu\n", sigVal, vA);
            }

            if ([mode isEqualToString:@"ane_to_metal"] || [mode isEqualToString:@"ane_to_metal_block"] || [mode isEqualToString:@"latency"]) {
                if (skipMetalWait) {
                    metalUnblocked = YES;
                    printf("metal_wait skipped=1\n");
                } else {
                    [cb waitUntilCompleted];
                    MTLCommandBufferStatus st = cb.status;
                    metalUnblocked = (st == MTLCommandBufferStatusCompleted);
                    printf("metal_wait unblocked=%d\n", metalUnblocked ? 1 : 0);
                }
            } else {
                metalUnblocked = YES;
            }
        }

        if (ok) {
            read_surface_f32(outSurf, outVals, count);
            printf("output[0..2]=[%.6f, %.6f, %.6f]\n", outVals[0], outVals[1], outVals[2]);
        }
        int rc = 1;
        if (ok && metalUnblocked) {
            printf("PASS: bidirectional mode=%s completed\n", mode.UTF8String);
            rc = 0;
        } else {
            printf("XFAIL: bidirectional mode=%s failed\n", mode.UTF8String);
            rc = 1;
        }

        if (skipCleanup) {
            TRACEF("skip cleanup and _exit rc=%d", rc);
            _exit(rc);
        }

        TRACEF("begin cleanup");
        ((void(*)(id, SEL, id, id))objc_msgSend)(client, @selector(unmapIOSurfacesWithModel:request:), model, req);
        TRACEF("unmap done");
        NSError *uerr = nil;
        ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &uerr);
        TRACEF("unload done err=%s", err_desc(uerr));

        CFRelease(outSurf);
        CFRelease(inSurf);
        free(outVals);
        free(inVals);
        TRACEF("cleanup done rc=%d", rc);
        return rc;
    }
}

int main(void) {
    const char *noFork = getenv("ANE_BIDIR_NO_FORK");
    if (noFork && noFork[0] == '1') {
        return run_probe();
    }

    pid_t pid = fork();
    if (pid == 0) {
        return run_probe();
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
        printf("XFAIL: child crashed with signal %d in bidirectional probe\n", WTERMSIG(status));
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
