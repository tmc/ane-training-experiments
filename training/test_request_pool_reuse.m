// test_request_pool_reuse.m
// Validate safe _ANERequest reuse after completion barrier and print queue diagnostics.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>

#include "../bridge/ane_bridge.h"

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
static void *gANE = NULL;
static const void *kReqAssoc = &kReqAssoc;

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
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

static id load_const_obj(const char *sym) {
    if (!gANE || !sym || !sym[0]) {
        return nil;
    }
    void *p = dlsym(gANE, sym);
    if (!p && sym[0] != '_') {
        char buf[128] = {0};
        snprintf(buf, sizeof(buf), "_%s", sym);
        p = dlsym(gANE, buf);
    }
    return p ? *((__unsafe_unretained id *)p) : nil;
}

static BOOL setup_classes(void) {
    gANE = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!gANE) {
        fprintf(stderr, "FAIL: dlopen AppleNeuralEngine failed\n");
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

static id make_wait_request(id inObj, id outObj, id waitShared, unsigned long long waitValue) {
    id waitEvent = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(
        CWait, @selector(waitEventWithValue:sharedEvent:), waitValue, waitShared);
    if (!waitEvent) {
        return nil;
    }
    id sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(
        CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[], @[waitEvent]);
    if (!sharedEvents) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0, sharedEvents, @1);
}

static BOOL set_request_shared_events(id req, id sharedEvents) {
    SEL setSel = NSSelectorFromString(@"setSharedEvents:");
    if ([req respondsToSelector:setSel]) {
        ((void(*)(id, SEL, id))objc_msgSend)(req, setSel, sharedEvents);
        return YES;
    }
    @try {
        [req setValue:sharedEvents forKey:@"sharedEvents"];
        return YES;
    } @catch (__unused NSException *e) {
        return NO;
    }
}

static BOOL eval_with_completion(id client,
                                 id model,
                                 id req,
                                 NSDictionary *opts,
                                 dispatch_semaphore_t sem,
                                 BOOL *fired,
                                 BOOL *success,
                                 NSError **cbErr,
                                 NSError **evalErr) {
    *fired = NO;
    *success = NO;
    *cbErr = nil;

    BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), model, req, YES, evalErr);
    if (!mapped) {
        return NO;
    }

    SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    if (![client respondsToSelector:evalSel]) {
        evalSel = @selector(evaluateWithModel:options:request:qos:error:);
    }
    BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client, evalSel, model, opts, req, kQoS, evalErr);
    long waited = ok ? dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * NSEC_PER_SEC))) : -1;

    ((void(*)(id, SEL, id, id))objc_msgSend)(client, @selector(unmapIOSurfacesWithModel:request:), model, req);

    if (!ok) {
        return NO;
    }
    return waited == 0 && *fired && *success && (*cbErr == nil);
}

static void print_queue_diag(id model) {
    int qd = -1;
    int inflight = -1;

    @try {
        id v = [model valueForKey:@"queueDepth"];
        if ([v respondsToSelector:@selector(intValue)]) {
            qd = ((int(*)(id, SEL))objc_msgSend)(v, @selector(intValue));
        }
    } @catch (__unused NSException *e) {}

    id program = nil;
    @try {
        program = [model valueForKey:@"programForEvaluation"];
    } @catch (__unused NSException *e) {}
    if (!program) {
        @try {
            program = [model valueForKey:@"program"];
        } @catch (__unused NSException *e) {}
    }
    if (program) {
        @try {
            id v = [program valueForKey:@"currentAsyncRequestsInFlight"];
            if ([v respondsToSelector:@selector(intValue)]) {
                inflight = ((int(*)(id, SEL))objc_msgSend)(v, @selector(intValue));
            }
        } @catch (__unused NSException *e) {}
    }

    printf("queueDepth=%d inflight=%d\n", qd, inflight);
}

static int run_once(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (ane_bridge_init() != 0) {
            printf("FAIL: ane_bridge_init\n");
            return 1;
        }
        if (!setup_classes()) {
            return 1;
        }

        id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
        if (!client) {
            printf("FAIL: sharedConnection nil\n");
            return 1;
        }

        NSString *modelPath = @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
        const char *raw = getenv("ANE_CHAIN_MODEL_PATH");
        if (raw && raw[0]) {
            modelPath = [NSString stringWithUTF8String:raw];
        }

        id model = nil;
        if (!compile_load_model(client, modelPath, @"s", &model)) {
            return 1;
        }

        print_queue_diag(model);

        const int count = 1024;
        const size_t bytes = (size_t)count * sizeof(float);
        IOSurfaceRef inSurf = make_surface(bytes);
        IOSurfaceRef outSurf = make_surface(bytes);
        float *inVals = (float *)calloc((size_t)count, sizeof(float));
        float *outA1 = (float *)calloc((size_t)count, sizeof(float));
        float *outA2 = (float *)calloc((size_t)count, sizeof(float));
        float *outB = (float *)calloc((size_t)count, sizeof(float));
        if (!inSurf || !outSurf || !inVals || !outA1 || !outA2 || !outB) {
            printf("FAIL: alloc/surface\n");
            return 1;
        }
        for (int i = 0; i < count; i++) {
            inVals[i] = (float)(i + 1);
        }
        write_surface_f32(inSurf, inVals, count);

        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
        if (!inObj || !outObj) {
            printf("FAIL: iosurface object wrap\n");
            return 1;
        }

        Class sharedCls = NSClassFromString(@"IOSurfaceSharedEvent");
        id waitShared = ((id(*)(Class, SEL))objc_msgSend)(sharedCls, @selector(alloc));
        waitShared = [waitShared respondsToSelector:@selector(initWithOptions:)]
            ? ((id(*)(id, SEL, unsigned long long))objc_msgSend)(waitShared, @selector(initWithOptions:), 0ULL)
            : ((id(*)(id, SEL))objc_msgSend)(waitShared, @selector(init));
        if (!waitShared) {
            printf("FAIL: create wait shared event\n");
            return 1;
        }
        mach_port_t waitPort = (mach_port_t)((unsigned int(*)(id, SEL))objc_msgSend)(waitShared, @selector(eventPort));
        if (waitPort == MACH_PORT_NULL) {
            printf("FAIL: wait event port\n");
            return 1;
        }

        id reqA = make_wait_request(inObj, outObj, waitShared, 1ULL);
        id reqB = make_wait_request(inObj, outObj, waitShared, 3ULL);
        if (!reqA || !reqB) {
            printf("FAIL: request creation\n");
            return 1;
        }

        dispatch_semaphore_t semA = dispatch_semaphore_create(0);
        __block BOOL firedA = NO;
        __block BOOL successA = NO;
        __block NSError *cbErrA = nil;
        void (^cbA)(BOOL, NSError *) = ^(BOOL success, NSError *error) {
            firedA = YES;
            successA = success;
            cbErrA = error;
            dispatch_semaphore_signal(semA);
        };
        id copiedA = [cbA copy];
        ((void(*)(id, SEL, id))objc_msgSend)(reqA, NSSelectorFromString(@"setCompletionHandler:"), copiedA);
        objc_setAssociatedObject(reqA, kReqAssoc, copiedA, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

        dispatch_semaphore_t semB = dispatch_semaphore_create(0);
        __block BOOL firedB = NO;
        __block BOOL successB = NO;
        __block NSError *cbErrB = nil;
        void (^cbB)(BOOL, NSError *) = ^(BOOL success, NSError *error) {
            firedB = YES;
            successB = success;
            cbErrB = error;
            dispatch_semaphore_signal(semB);
        };
        id copiedB = [cbB copy];
        ((void(*)(id, SEL, id))objc_msgSend)(reqB, NSSelectorFromString(@"setCompletionHandler:"), copiedB);
        objc_setAssociatedObject(reqB, kReqAssoc, copiedB, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

        id disableKey = load_const_obj("kANEFDisableIOFencesUseSharedEventsKey");
        if (!disableKey) {
            disableKey = @"kANEFDisableIOFencesUseSharedEventsKey";
        }
        id fwKey = load_const_obj("kANEFEnableFWToFWSignal");
        if (!fwKey) {
            fwKey = @"kANEFEnableFWToFWSignal";
        }
        NSDictionary *opts = @{ disableKey: @YES, fwKey: @NO };

        NSError *evalErr = nil;
        int sigRC = ane_bridge_signal_event_cpu(waitPort, 1ULL);
        if (sigRC != 0) {
            printf("FAIL: cpu signal A1 rc=%d\n", sigRC);
            return 1;
        }
        BOOL okA1 = eval_with_completion(client, model, reqA, opts, semA, &firedA, &successA, &cbErrA, &evalErr);
        read_surface_f32(outSurf, outA1, count);
        printf("reqA first ok=%d evalErr=%s cbErr=%s out[0..2]=[%.6f, %.6f, %.6f]\n",
               okA1 ? 1 : 0, err_desc(evalErr), err_desc(cbErrA), outA1[0], outA1[1], outA1[2]);
        if (!okA1) {
            return 1;
        }

        id waitEvent2 = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(
            CWait, @selector(waitEventWithValue:sharedEvent:), 2ULL, waitShared);
        id shared2 = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            CSharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[], @[waitEvent2]);
        BOOL updated = waitEvent2 && shared2 && set_request_shared_events(reqA, shared2);
        printf("reqA update sharedEvents=%d\n", updated ? 1 : 0);
        if (!updated) {
            return 1;
        }

        evalErr = nil;
        sigRC = ane_bridge_signal_event_cpu(waitPort, 2ULL);
        if (sigRC != 0) {
            printf("FAIL: cpu signal A2 rc=%d\n", sigRC);
            return 1;
        }
        BOOL okA2 = eval_with_completion(client, model, reqA, opts, semA, &firedA, &successA, &cbErrA, &evalErr);
        read_surface_f32(outSurf, outA2, count);
        printf("reqA second ok=%d evalErr=%s cbErr=%s out[0..2]=[%.6f, %.6f, %.6f]\n",
               okA2 ? 1 : 0, err_desc(evalErr), err_desc(cbErrA), outA2[0], outA2[1], outA2[2]);
        if (!okA2) {
            return 1;
        }

        evalErr = nil;
        sigRC = ane_bridge_signal_event_cpu(waitPort, 3ULL);
        if (sigRC != 0) {
            printf("FAIL: cpu signal B rc=%d\n", sigRC);
            return 1;
        }
        BOOL okB = eval_with_completion(client, model, reqB, opts, semB, &firedB, &successB, &cbErrB, &evalErr);
        read_surface_f32(outSurf, outB, count);
        printf("reqB ok=%d evalErr=%s cbErr=%s out[0..2]=[%.6f, %.6f, %.6f]\n",
               okB ? 1 : 0, err_desc(evalErr), err_desc(cbErrB), outB[0], outB[1], outB[2]);

        double dA = fmax(fabs((double)outA1[0] - (double)outA2[0]),
                         fmax(fabs((double)outA1[1] - (double)outA2[1]), fabs((double)outA1[2] - (double)outA2[2])));
        bool pass = okA1 && okA2 && okB && dA < 1e-6;
        printf("%s: request reuse A->A and pool B dA=%g\n", pass ? "PASS" : "FAIL", dA);

        _exit(pass ? 0 : 1);
    }
}

int main(void) {
    setbuf(stdout, NULL);
    const int attempts = 3;
    for (int i = 1; i <= attempts; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            int rc = run_once();
            _exit(rc);
        }
        if (pid < 0) {
            printf("FAIL: fork\n");
            return 1;
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            printf("FAIL: waitpid\n");
            return 1;
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            return 0;
        }
        if (WIFSIGNALED(status)) {
            printf("WARN: request reuse attempt=%d crashed signal=%d\n", i, WTERMSIG(status));
        } else if (WIFEXITED(status)) {
            printf("WARN: request reuse attempt=%d exit=%d\n", i, WEXITSTATUS(status));
        }
    }
    printf("FAIL: request reuse retries exhausted\n");
    return 1;
}
