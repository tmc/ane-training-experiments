// test_metal_ane_bridge.m
// Probe ANE -> IOSurfaceSharedEvent -> MTLSharedEvent wait bridging.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

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
static Class CSharedEvents;
static Class CNSURL;

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

static BOOL setup_classes(void **outHandle) {
    void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!ane) {
        fprintf(stderr, "failed to dlopen AppleNeuralEngine: %s\n", dlerror());
        return NO;
    }
    *outHandle = ane;

    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");
    CSig = NSClassFromString(@"_ANESharedSignalEvent");
    CSharedEvents = NSClassFromString(@"_ANESharedEvents");
    CNSURL = NSClassFromString(@"NSURL");

    if (!CClient || !CModel || !CReq || !CAIO || !CSig || !CSharedEvents || !CNSURL) {
        fprintf(stderr, "failed to resolve required ANE classes\n");
        return NO;
    }
    return YES;
}

static int run_bridge(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        void *ane = NULL;
        if (!setup_classes(&ane)) {
            return 2;
        }

        NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
        const char *rawPath = getenv("ANE_CHAIN_MODEL_PATH");
        if (rawPath && rawPath[0]) {
            modelPath = [NSString stringWithUTF8String:rawPath];
        }
        NSString *modelKey = @"s";

        id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
        if (!client) {
            fprintf(stderr, "_ANEClient sharedConnection returned nil\n");
            return 2;
        }

        id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), modelPath);
        id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, modelKey);
        if (!model) {
            fprintf(stderr, "modelAtURL:key: returned nil path=%s\n", modelPath.UTF8String);
            return 2;
        }

        NSError *err = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(compileModel:options:qos:error:), model, @{}, kQoS, &err);
        if (!ok) {
            fprintf(stderr, "compileModel failed: %s\n", err_desc(err));
            return 2;
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
            return 2;
        }

        const int count = 1024;
        const size_t bytes = (size_t)count * sizeof(float);
        IOSurfaceRef inSurf = make_surface(bytes);
        IOSurfaceRef outSurf = make_surface(bytes);
        if (!inSurf || !outSurf) {
            fprintf(stderr, "surface allocation failed\n");
            return 2;
        }

        float *inVals = (float *)calloc((size_t)count, sizeof(float));
        float *outVals = (float *)calloc((size_t)count, sizeof(float));
        for (int i = 0; i < count; i++) {
            inVals[i] = (float)(i + 1);
        }
        write_surface_f32(inSurf, inVals, count);

        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);

        Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
        id iosEvent = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
        if ([iosEvent respondsToSelector:@selector(initWithOptions:)]) {
            iosEvent = ((id(*)(id, SEL, unsigned long long))objc_msgSend)(iosEvent, @selector(initWithOptions:), 0ULL);
        } else {
            iosEvent = ((id(*)(id, SEL))objc_msgSend)(iosEvent, @selector(init));
        }
        if (!iosEvent || ![iosEvent respondsToSelector:@selector(eventPort)]) {
            fprintf(stderr, "IOSurfaceSharedEvent creation failed\n");
            return 2;
        }
        unsigned int port = ((unsigned int(*)(id, SEL))objc_msgSend)(iosEvent, @selector(eventPort));
        if (port == 0) {
            fprintf(stderr, "IOSurfaceSharedEvent eventPort=0\n");
            return 2;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
        if (![dev respondsToSelector:newSharedSel]) {
            fprintf(stderr, "MTLDevice missing newSharedEventWithMachPort:\n");
            return 2;
        }
        id mtlEvent = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, port);
        if (!mtlEvent) {
            fprintf(stderr, "newSharedEventWithMachPort returned nil\n");
            return 2;
        }

        unsigned long long baseValue = 0;
        if ([mtlEvent respondsToSelector:@selector(signaledValue)]) {
            baseValue = ((unsigned long long(*)(id, SEL))objc_msgSend)(mtlEvent, @selector(signaledValue));
        }
        unsigned long long waitValue = baseValue + 1;

        id sig = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
            CSig,
            @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
            waitValue,
            0,
            5,
            iosEvent
        );
        if (!sig) {
            fprintf(stderr, "signalEventWithValue failed\n");
            return 2;
        }

        id sharedWrap = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            CSharedEvents,
            @selector(sharedEventsWithSignalEvents:waitEvents:),
            @[sig],
            @[]
        );
        if (!sharedWrap) {
            fprintf(stderr, "sharedEventsWithSignalEvents failed\n");
            return 2;
        }

        SEL reqSel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
        if (![CReq respondsToSelector:reqSel]) {
            fprintf(stderr, "requestWith...sharedEvents selector missing\n");
            return 2;
        }
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            CReq,
            reqSel,
            @[inObj],
            @[@0],
            @[outObj],
            @[@0],
            nil,
            nil,
            @0,
            sharedWrap,
            @1
        );
        if (!req) {
            fprintf(stderr, "request creation failed\n");
            return 2;
        }

        if ([req respondsToSelector:@selector(validate)]) {
            BOOL valid = ((BOOL(*)(id, SEL))objc_msgSend)(req, @selector(validate));
            if (!valid) {
                fprintf(stderr, "request validate=false\n");
                return 2;
            }
        }

        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            client,
            @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
            model,
            req,
            YES,
            &err
        );
        if (!ok) {
            fprintf(stderr, "mapIOSurfaces failed: %s\n", err_desc(err));
            return 2;
        }

        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        if (!cb || ![cb respondsToSelector:@selector(encodeWaitForEvent:value:)]) {
            fprintf(stderr, "command buffer/event wait not available\n");
            return 2;
        }

        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(__unused id<MTLCommandBuffer> b) {
            dispatch_semaphore_signal(done);
        }];
        ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb, @selector(encodeWaitForEvent:value:), mtlEvent, waitValue);
        [cb commit];

        NSMutableDictionary *opts = [NSMutableDictionary dictionary];
        id disableFenceKey = load_anef_constant_obj(ane, "kANEFDisableIOFencesUseSharedEventsKey");
        if (!disableFenceKey) disableFenceKey = @"kANEFDisableIOFencesUseSharedEventsKey";
        opts[disableFenceKey] = @YES;
        id fwKey = load_anef_constant_obj(ane, "kANEFEnableFWToFWSignal");
        if (!fwKey) fwKey = @"kANEFEnableFWToFWSignal";
        opts[fwKey] = @YES;

        SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
        if (![client respondsToSelector:evalSel]) {
            evalSel = @selector(evaluateWithModel:options:request:qos:error:);
        }

        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            evalSel,
            model,
            opts,
            req,
            kQoS,
            &err
        );
        printf("evaluate selector=%s ok=%d err=%s\n", sel_getName(evalSel), ok ? 1 : 0, err_desc(err));

        long wait = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(3 * NSEC_PER_SEC)));
        BOOL metalUnblocked = (wait == 0);
        printf("metal_wait value=%llu unblocked=%d\n", waitValue, metalUnblocked ? 1 : 0);

        if (ok) {
            read_surface_f32(outSurf, outVals, count);
            printf("output[0..2]=[%.6f, %.6f, %.6f]\n", outVals[0], outVals[1], outVals[2]);
        }

        ((void(*)(id, SEL, id, id))objc_msgSend)(client, @selector(unmapIOSurfacesWithModel:request:), model, req);
        NSError *uerr = nil;
        ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &uerr);

        CFRelease(outSurf);
        CFRelease(inSurf);
        free(outVals);
        free(inVals);

        if (ok && metalUnblocked) {
            printf("PASS: ANE evaluate signaled shared event and Metal wait unblocked\n");
            return 0;
        }

        if (!ok) {
            printf("XFAIL: ANE evaluate failed before signaling path completed\n");
        } else if (!metalUnblocked) {
            printf("XFAIL: ANE evaluate succeeded but Metal wait did not unblock (no ANE->Metal signal observed)\n");
        }
        return 1;
    }
}

int main(void) {
    pid_t pid = fork();
    if (pid == 0) {
        return run_bridge();
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
        printf("XFAIL: child crashed with signal %d during ANE shared-events bridge\n", WTERMSIG(status));
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
