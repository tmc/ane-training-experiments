// test_chaining_suite.m - pure Objective-C/C chaining test suite derived from mlx-go-ane research.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const unsigned int kDefaultQoS = 21;
static const int kDefaultTensorCount = 1024; // 1*64*1*16 fp32
static const double kCompareTolerance = 2e-2;
static void *gANEFramework = NULL;

static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIOSurfaceObject;
static Class CBuffer;
static Class COutputSets;
static Class CChainingReq;
static Class CSharedSignalEvent;
static Class CSharedEvents;
static Class COutputSetEnqueue;
static Class CInputBuffersReady;
static Class CNSURL;

@interface ANEContext : NSObject
@property(nonatomic, strong) id client;
@property(nonatomic, strong) id model;
@property(nonatomic, copy) NSString *modelPath;
@property(nonatomic, copy) NSString *modelKey;
@property(nonatomic) unsigned int qos;
@property(nonatomic) int inCount;
@property(nonatomic) int outCount;
@end

@implementation ANEContext
@end

typedef NS_ENUM(int, CaseStatus) {
    CasePass = 0,
    CaseFail = 1,
    CaseXFail = 2,
    CaseSkip = 3,
};

typedef struct {
    const char *name;
    CaseStatus status;
    char detail[640];
} CaseResult;

static CaseResult make_result(const char *name, CaseStatus status, const char *fmt, ...) {
    CaseResult out;
    out.name = name;
    out.status = status;
    out.detail[0] = '\0';
    if (fmt != NULL && fmt[0] != '\0') {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(out.detail, sizeof(out.detail), fmt, ap);
        va_end(ap);
    }
    return out;
}

static NSString *env_string(const char *name, NSString *fallback) {
    const char *raw = getenv(name);
    if (raw == NULL || raw[0] == '\0') {
        return fallback;
    }
    return [NSString stringWithUTF8String:raw];
}

static int env_int(const char *name, int fallback) {
    const char *raw = getenv(name);
    if (raw == NULL || raw[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    long v = strtol(raw, &end, 10);
    if (end == raw || *end != '\0') {
        return fallback;
    }
    return (int)v;
}

static BOOL env_enabled(const char *name) {
    const char *raw = getenv(name);
    if (raw == NULL || raw[0] == '\0') {
        return NO;
    }
    if (strcmp(raw, "0") == 0 || strcasecmp(raw, "false") == 0 || strcasecmp(raw, "no") == 0 || strcasecmp(raw, "off") == 0) {
        return NO;
    }
    return YES;
}

static id load_anef_constant_obj(const char *symbol) {
    if (gANEFramework == NULL || symbol == NULL || symbol[0] == '\0') {
        return nil;
    }
    void *sym = dlsym(gANEFramework, symbol);
    if (sym == NULL && symbol[0] != '_') {
        char buf[128] = {0};
        snprintf(buf, sizeof(buf), "_%s", symbol);
        sym = dlsym(gANEFramework, buf);
    }
    if (sym == NULL) {
        return nil;
    }
    id obj = *((__unsafe_unretained id *)sym);
    return obj;
}

static NSMutableDictionary *make_shared_event_options(void) {
    NSMutableDictionary *opts = [NSMutableDictionary dictionaryWithCapacity:2];

    id disableFenceKey = load_anef_constant_obj("kANEFDisableIOFencesUseSharedEventsKey");
    if (disableFenceKey == nil) {
        disableFenceKey = @"kANEFDisableIOFencesUseSharedEventsKey";
    }
    opts[disableFenceKey] = @YES;

    BOOL enableFWToFW = YES;
    const char *raw = getenv("ANE_CHAIN_ENABLE_FW_TO_FW_SIGNAL");
    if (raw != NULL && raw[0] != '\0') {
        if (strcmp(raw, "0") == 0 || strcasecmp(raw, "false") == 0 || strcasecmp(raw, "no") == 0 || strcasecmp(raw, "off") == 0) {
            enableFWToFW = NO;
        }
    }
    if (enableFWToFW) {
        id fwKey = load_anef_constant_obj("kANEFEnableFWToFWSignal");
        if (fwKey == nil) {
            fwKey = @"kANEFEnableFWToFWSignal";
        }
        opts[fwKey] = @YES;
    }
    return opts;
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

static BOOL write_surface_f32(IOSurfaceRef surf, const float *vals, int count, char *detail, size_t detailLen) {
    if (surf == NULL || vals == NULL || count < 0) {
        snprintf(detail, detailLen, "invalid write_surface args");
        return NO;
    }
    const size_t need = (size_t)count * sizeof(float);
    const size_t got = IOSurfaceGetAllocSize(surf);
    if (got < need) {
        snprintf(detail, detailLen, "surface too small: got=%zu need=%zu", got, need);
        return NO;
    }
    if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) {
        snprintf(detail, detailLen, "IOSurfaceLock failed");
        return NO;
    }
    void *base = IOSurfaceGetBaseAddress(surf);
    if (base == NULL) {
        IOSurfaceUnlock(surf, 0, NULL);
        snprintf(detail, detailLen, "IOSurfaceGetBaseAddress returned nil");
        return NO;
    }
    memcpy(base, vals, need);
    IOSurfaceUnlock(surf, 0, NULL);
    return YES;
}

static BOOL read_surface_f32(IOSurfaceRef surf, float *out, int count, char *detail, size_t detailLen) {
    if (surf == NULL || out == NULL || count < 0) {
        snprintf(detail, detailLen, "invalid read_surface args");
        return NO;
    }
    const size_t need = (size_t)count * sizeof(float);
    const size_t got = IOSurfaceGetAllocSize(surf);
    if (got < need) {
        snprintf(detail, detailLen, "surface too small: got=%zu need=%zu", got, need);
        return NO;
    }
    if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) {
        snprintf(detail, detailLen, "IOSurfaceLock failed");
        return NO;
    }
    const void *base = IOSurfaceGetBaseAddress(surf);
    if (base == NULL) {
        IOSurfaceUnlock(surf, 0, NULL);
        snprintf(detail, detailLen, "IOSurfaceGetBaseAddress returned nil");
        return NO;
    }
    memcpy(out, base, need);
    IOSurfaceUnlock(surf, 0, NULL);
    return YES;
}

static float *make_input(int count) {
    if (count <= 0) {
        return NULL;
    }
    float *out = (float *)calloc((size_t)count, sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        out[i] = (float)((i % 23) - 11) * 0.03125f;
    }
    return out;
}

static double max_abs_diff(const float *a, const float *b, int n) {
    double maxDiff = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > maxDiff) {
            maxDiff = d;
        }
    }
    return maxDiff;
}

static const char *err_text(NSError *err) {
    if (err == nil) {
        return "nil";
    }
    return [[err description] UTF8String];
}

static BOOL is_program_iosurface_map_failure(NSError *err) {
    if (err == nil) {
        return NO;
    }
    NSString *desc = [err description];
    if (desc == nil) {
        return NO;
    }
    if ([desc containsString:@"Program IOSurfaces map failure"]) {
        return YES;
    }
    if ([desc containsString:@"Code=13"] || [desc containsString:@"code=13"]) {
        return YES;
    }
    return NO;
}

static BOOL call_compile(id client, id model, unsigned int qos, NSError **err) {
    @try {
        return ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(compileModel:options:qos:error:),
            model,
            @{},
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "compile exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_load(id client, id model, unsigned int qos, NSError **err) {
    @try {
        return ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(loadModel:options:qos:error:),
            model,
            @{},
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "load exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_load_with_retry(id client, id model, unsigned int qos, NSError **err) {
    BOOL ok = call_load(client, model, qos, err);
    if (ok) {
        return YES;
    }
    usleep(100000);
    if (err != NULL) {
        *err = nil;
    }
    return call_load(client, model, qos, err);
}

static BOOL call_unload(id client, id model, unsigned int qos, NSError **err) {
    @try {
        return ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(unloadModel:options:qos:error:),
            model,
            @{},
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "unload exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_map(id client, id model, id request, NSError **err) {
    @try {
        return ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            client,
            @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
            model,
            request,
            YES,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "map exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL map_selector_matches_signature(id target, SEL sel) {
    if (target == nil || sel == NULL || ![target respondsToSelector:sel]) {
        return NO;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (sig == nil) {
        return NO;
    }
    // self + _cmd + model + payload + cacheInference + error
    return sig.numberOfArguments == 6;
}

static BOOL call_map_alt_sel(id client, SEL sel, id model, id payload, NSError **err) {
    if (!map_selector_matches_signature(client, sel)) {
        if (err != NULL) {
            *err = [NSError errorWithDomain:@"chaining" code:2 userInfo:@{NSLocalizedDescriptionKey: @"map selector missing/signature mismatch"}];
        }
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            client,
            sel,
            model,
            payload,
            YES,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "map alt exception (%s): %s\n", sel_getName(sel), [[ex reason] UTF8String]);
        return NO;
    }
}

static void call_unmap(id client, id model, id request) {
    @try {
        ((void(*)(id, SEL, id, id))objc_msgSend)(
            client,
            @selector(unmapIOSurfacesWithModel:request:),
            model,
            request
        );
    } @catch (NSException *ex) {
        fprintf(stderr, "unmap exception: %s\n", [[ex reason] UTF8String]);
    }
}

static BOOL call_eval_opt(id client, id model, id options, id request, unsigned int qos, BOOL direct, NSError **err) {
    SEL sel = direct
        ? @selector(doEvaluateDirectWithModel:options:request:qos:error:)
        : @selector(evaluateWithModel:options:request:qos:error:);
    if (![client respondsToSelector:sel]) {
        if (err != NULL) {
            *err = [NSError errorWithDomain:@"chaining" code:1 userInfo:@{NSLocalizedDescriptionKey: @"evaluate selector missing"}];
        }
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            sel,
            model,
            options != nil ? options : @{},
            request,
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "evaluate exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_eval(id client, id model, id request, unsigned int qos, BOOL direct, NSError **err) {
    return call_eval_opt(client, model, @{}, request, qos, direct, err);
}

static BOOL call_prepare_sel_opt(id client, SEL sel, id model, id options, id chainReq, unsigned int qos, NSError **err) {
    if (![client respondsToSelector:sel]) {
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            sel,
            model,
            options != nil ? options : @{},
            chainReq,
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "prepare exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_enqueue_sel_opt(id client, SEL sel, id model, id options, id enqueueObj, unsigned int qos, NSError **err) {
    if (![client respondsToSelector:sel]) {
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            sel,
            model,
            enqueueObj,
            options != nil ? options : @{},
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "enqueue exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_ready_sel_opt(id client, SEL sel, id model, id options, id readyObj, unsigned int qos, NSError **err) {
    if (![client respondsToSelector:sel]) {
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            sel,
            model,
            readyObj,
            options != nil ? options : @{},
            qos,
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "ready exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static id model_program(id model) {
    if (model == nil || ![model respondsToSelector:@selector(program)]) {
        return nil;
    }
    return ((id(*)(id, SEL))objc_msgSend)(model, @selector(program));
}

static BOOL call_program_process_output(id model, id outputSetObj, id options, NSError **err) {
    id program = model_program(model);
    if (program == nil) {
        return NO;
    }
    SEL sel = @selector(processOutputSet:model:options:error:);
    if (![program respondsToSelector:sel]) {
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, id, NSError **))objc_msgSend)(
            program,
            sel,
            outputSetObj,
            model,
            options != nil ? options : @{},
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "processOutputSet exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static BOOL call_program_process_input(id model, id inputReadyObj, id options, NSError **err) {
    id program = model_program(model);
    if (program == nil) {
        return NO;
    }
    SEL sel = @selector(processInputBuffers:model:options:error:);
    if (![program respondsToSelector:sel]) {
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL, id, id, id, NSError **))objc_msgSend)(
            program,
            sel,
            inputReadyObj,
            model,
            options != nil ? options : @{},
            err
        );
    } @catch (NSException *ex) {
        if (err != NULL) {
            *err = nil;
        }
        fprintf(stderr, "processInputBuffers exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

static id make_simple_request(id inObj, id outObj, int proc, int inSymbol, int outSymbol) {
    if (CReq == Nil) {
        return nil;
    }
    SEL sel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    if (![CReq respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        sel,
        @[inObj],
        @[@(inSymbol)],
        @[outObj],
        @[@(outSymbol)],
        nil,
        nil,
        @(proc)
    );
}

static id make_full_request(id inObj, id outObj, int proc, int inSymbol, int outSymbol, id sharedEvents, unsigned long long txnHandle) {
    if (CReq == Nil) {
        return nil;
    }
    SEL sel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:);
    if (![CReq respondsToSelector:sel]) {
        return nil;
    }
    id txn = txnHandle == 0 ? nil : @(txnHandle);
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CReq,
        sel,
        @[inObj],
        @[@(inSymbol)],
        @[outObj],
        @[@(outSymbol)],
        nil,
        nil,
        @(proc),
        sharedEvents,
        txn
    );
}

static id make_shared_event(void) {
    Class cls = NSClassFromString(@"IOSurfaceSharedEvent");
    if (cls == Nil) {
        return nil;
    }
    id ev = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
    if (ev == nil) {
        return nil;
    }
    if ([ev respondsToSelector:@selector(initWithOptions:)]) {
        ev = ((id(*)(id, SEL, unsigned long long))objc_msgSend)(ev, @selector(initWithOptions:), 0ULL);
    } else {
        ev = ((id(*)(id, SEL))objc_msgSend)(ev, @selector(init));
    }
    if (ev == nil) {
        return nil;
    }
    if ([ev respondsToSelector:@selector(eventPort)]) {
        unsigned int port = ((unsigned int(*)(id, SEL))objc_msgSend)(ev, @selector(eventPort));
        if (port == 0) {
            return nil;
        }
    }
    return ev;
}

static id make_signal_event(unsigned long long value, unsigned int symbolIndex, long long eventType, id sharedEvent, unsigned long long agentMask) {
    if (CSharedSignalEvent == Nil || sharedEvent == nil) {
        return nil;
    }
    SEL sel = @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:);
    if (![CSharedSignalEvent respondsToSelector:sel]) {
        return nil;
    }
    id event = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
        CSharedSignalEvent,
        sel,
        value,
        symbolIndex,
        eventType,
        sharedEvent
    );
    if (event != nil && [event respondsToSelector:@selector(setAgentMask:)]) {
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(event, @selector(setAgentMask:), agentMask);
    }
    return event;
}

static id make_shared_events_wrapper(id signalEvents) {
    if (CSharedEvents == Nil || signalEvents == nil) {
        return nil;
    }
    SEL sel = @selector(sharedEventsWithSignalEvents:waitEvents:);
    if (![CSharedEvents respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id))objc_msgSend)(CSharedEvents, sel, signalEvents, @[]);
}

static id make_buffer(id ioSurfaceObj, unsigned int symbolIndex, long long source) {
    if (CBuffer == Nil || ioSurfaceObj == nil) {
        return nil;
    }
    SEL sel = @selector(bufferWithIOSurfaceObject:symbolIndex:source:);
    if (![CBuffer respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, long long))objc_msgSend)(
        CBuffer,
        sel,
        ioSurfaceObj,
        @(symbolIndex),
        source
    );
}

static id make_output_set(IOSurfaceRef statsSurf, id outputBuffers) {
    if (COutputSets == Nil || outputBuffers == nil) {
        return nil;
    }
    SEL selA = NSSelectorFromString(@"objectWithstatsSurRef:outputBuffer:");
    SEL selB = NSSelectorFromString(@"outputSetsWithstatsSurRef:outputBuffer:");
    if ([COutputSets respondsToSelector:selA]) {
        return ((id(*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(COutputSets, selA, statsSurf, outputBuffers);
    }
    if ([COutputSets respondsToSelector:selB]) {
        return ((id(*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(COutputSets, selB, statsSurf, outputBuffers);
    }
    return nil;
}

static id make_chaining_request(
    id inputs,
    id outputSets,
    id lbInputSymbolId,
    id lbOutputSymbolId,
    int proc,
    id signalEvents,
    unsigned long long txn,
    unsigned long long fwDelay,
    unsigned long long memPool
) {
    if (CChainingReq == Nil) {
        return nil;
    }
    SEL sel = @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:);
    if (![CChainingReq respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        CChainingReq,
        sel,
        inputs,
        outputSets,
        lbInputSymbolId != nil ? lbInputSymbolId : @[],
        lbOutputSymbolId != nil ? lbOutputSymbolId : @[],
        @(proc),
        signalEvents,
        @(txn),
        @(fwDelay),
        @(memPool)
    );
}

static id make_enqueue(unsigned int proc, unsigned int setIdx, unsigned long long signalValue, BOOL signalNotRequired, BOOL isOpenLoop) {
    if (COutputSetEnqueue == Nil) {
        return nil;
    }
    SEL sel = @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:);
    if (![COutputSetEnqueue respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, unsigned int, unsigned int, unsigned long long, BOOL, BOOL))objc_msgSend)(
        COutputSetEnqueue,
        sel,
        proc,
        setIdx,
        signalValue,
        signalNotRequired,
        isOpenLoop
    );
}

static id make_ready(unsigned int proc, NSArray *inputIndices, NSArray *inputFreeValues, unsigned long long executionDelay) {
    if (CInputBuffersReady == Nil || inputIndices == nil || inputFreeValues == nil) {
        return nil;
    }
    SEL sel = @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:);
    if (![CInputBuffersReady respondsToSelector:sel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, unsigned int, id, id, unsigned long long))objc_msgSend)(
        CInputBuffersReady,
        sel,
        proc,
        inputIndices,
        inputFreeValues,
        executionDelay
    );
}

static BOOL request_validate(id req) {
    if (req == nil || ![req respondsToSelector:@selector(validate)]) {
        return NO;
    }
    @try {
        return ((BOOL(*)(id, SEL))objc_msgSend)(req, @selector(validate));
    } @catch (NSException *ex) {
        fprintf(stderr, "validate exception: %s\n", [[ex reason] UTF8String]);
        return NO;
    }
}

enum {
    kANEDeviceMethodOffsetProgramOutputSetEnqueue = 0x28,
    kANEDeviceMethodOffsetProgramInputsReady = 0x30,
    kANEDeviceMethodOffsetProgramChainingSetActiveProc = 0x38,
};

typedef struct {
    uint64_t programHandle;
    uint32_t procedureIndex;
    uint32_t reserved;
    uint8_t extra[16];
} ANESetActiveProcedurePayload;

static BOOL model_program_handle_and_device(id model, uint64_t *programHandle, void **device, char *detail, size_t detailLen) {
    if (model == nil) {
        snprintf(detail, detailLen, "model=nil");
        return NO;
    }
    id program = model_program(model);
    if (program == nil) {
        snprintf(detail, detailLen, "model.program=nil");
        return NO;
    }
    uint64_t handle = 0;
    if ([model respondsToSelector:@selector(programHandle)]) {
        handle = ((uint64_t(*)(id, SEL))objc_msgSend)(model, @selector(programHandle));
    }
    if (handle == 0 && [program respondsToSelector:@selector(programHandle)]) {
        handle = ((uint64_t(*)(id, SEL))objc_msgSend)(program, @selector(programHandle));
    }
    if (handle == 0) {
        snprintf(detail, detailLen, "programHandle=0");
        return NO;
    }
    if (![program respondsToSelector:@selector(controller)]) {
        snprintf(detail, detailLen, "program.controller unavailable");
        return NO;
    }
    id controller = ((id(*)(id, SEL))objc_msgSend)(program, @selector(controller));
    if (controller == nil || ![controller respondsToSelector:@selector(device)]) {
        snprintf(detail, detailLen, "controller/device unavailable");
        return NO;
    }
    void *dev = ((void *(*)(id, SEL))objc_msgSend)(controller, @selector(device));
    if (dev == NULL) {
        snprintf(detail, detailLen, "device pointer is nil");
        return NO;
    }
    *programHandle = handle;
    *device = dev;
    return YES;
}

static BOOL call_device_method(void *device, uintptr_t methodOffset, void *arg, int32_t *rc, char *detail, size_t detailLen) {
    if (device == NULL) {
        snprintf(detail, detailLen, "device=nil");
        return NO;
    }
    uintptr_t vtable = *((uintptr_t *)device);
    if (vtable == 0) {
        snprintf(detail, detailLen, "device vtable=nil");
        return NO;
    }
    uintptr_t fn = *((uintptr_t *)(vtable + methodOffset));
    if (fn == 0) {
        snprintf(detail, detailLen, "device method offset 0x%zx is nil", (size_t)methodOffset);
        return NO;
    }
    typedef int32_t (*DeviceFn)(void *, void *);
    int32_t out = ((DeviceFn)fn)(device, arg);
    if (rc != NULL) {
        *rc = out;
    }
    return YES;
}

static BOOL setup_classes(char *detail, size_t detailLen) {
    gANEFramework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);

    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIOSurfaceObject = NSClassFromString(@"_ANEIOSurfaceObject");
    CBuffer = NSClassFromString(@"_ANEBuffer");
    COutputSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
    CChainingReq = NSClassFromString(@"_ANEChainingRequest");
    CSharedSignalEvent = NSClassFromString(@"_ANESharedSignalEvent");
    CSharedEvents = NSClassFromString(@"_ANESharedEvents");
    COutputSetEnqueue = NSClassFromString(@"_ANEOutputSetEnqueue");
    CInputBuffersReady = NSClassFromString(@"_ANEInputBuffersReady");
    CNSURL = NSClassFromString(@"NSURL");

    struct {
        const char *name;
        Class cls;
    } reqs[] = {
        {"_ANEClient", CClient},
        {"_ANEModel", CModel},
        {"_ANERequest", CReq},
        {"_ANEIOSurfaceObject", CAIOSurfaceObject},
        {"_ANEBuffer", CBuffer},
        {"_ANEIOSurfaceOutputSets", COutputSets},
        {"_ANEChainingRequest", CChainingReq},
        {"_ANESharedSignalEvent", CSharedSignalEvent},
        {"_ANEOutputSetEnqueue", COutputSetEnqueue},
        {"_ANEInputBuffersReady", CInputBuffersReady},
        {"NSURL", CNSURL},
    };

    for (size_t i = 0; i < sizeof(reqs) / sizeof(reqs[0]); i++) {
        if (reqs[i].cls == Nil) {
            snprintf(detail, detailLen, "missing required class %s", reqs[i].name);
            return NO;
        }
    }
    return YES;
}

static BOOL load_model_at_path(id client, NSString *modelPath, NSString *modelKey, unsigned int qos, id *outModel, char *detail, size_t detailLen) {
    if (client == nil || modelPath == nil || modelKey == nil || outModel == NULL) {
        snprintf(detail, detailLen, "invalid load_model_at_path args");
        return NO;
    }
    id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), modelPath);
    if (url == nil) {
        snprintf(detail, detailLen, "fileURLWithPath failed for %s", modelPath.UTF8String);
        return NO;
    }

    id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, modelKey);
    if (model == nil) {
        snprintf(detail, detailLen, "_ANEModel modelAtURL:key: returned nil for path=%s key=%s", modelPath.UTF8String, modelKey.UTF8String);
        return NO;
    }

    NSError *err = nil;
    BOOL ok = call_compile(client, model, qos, &err);
    if (!ok) {
        snprintf(detail, detailLen, "compileModel failed path=%s key=%s err=%s", modelPath.UTF8String, modelKey.UTF8String, err_text(err));
        return NO;
    }

    err = nil;
    ok = call_load_with_retry(client, model, qos, &err);
    if (!ok) {
        snprintf(detail, detailLen, "loadModel failed path=%s key=%s err=%s", modelPath.UTF8String, modelKey.UTF8String, err_text(err));
        return NO;
    }

    *outModel = model;
    return YES;
}

static BOOL setup_context(ANEContext *ctx, char *detail, size_t detailLen) {
    if (!setup_classes(detail, detailLen)) {
        return NO;
    }

    ctx.modelPath = env_string(
        "ANE_CHAIN_MODEL_PATH",
        @"/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc"
    );
    ctx.modelKey = env_string("ANE_CHAIN_MODEL_KEY", @"s");
    ctx.qos = (unsigned int)env_int("ANE_CHAIN_QOS", (int)kDefaultQoS);
    ctx.inCount = env_int("ANE_CHAIN_INPUT_COUNT", kDefaultTensorCount);
    ctx.outCount = env_int("ANE_CHAIN_OUTPUT_COUNT", kDefaultTensorCount);
    if (ctx.inCount <= 0 || ctx.outCount <= 0) {
        snprintf(detail, detailLen, "invalid counts in=%d out=%d", ctx.inCount, ctx.outCount);
        return NO;
    }

    ctx.client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
    if (ctx.client == nil) {
        snprintf(detail, detailLen, "_ANEClient sharedConnection returned nil");
        return NO;
    }

    char loadDetail[320] = {0};
    id model = nil;
    if (!load_model_at_path(ctx.client, ctx.modelPath, ctx.modelKey, ctx.qos, &model, loadDetail, sizeof(loadDetail))) {
        snprintf(detail, detailLen, "%s", loadDetail);
        return NO;
    }
    ctx.model = model;

    snprintf(detail, detailLen, "model loaded path=%s key=%s qos=%u in=%d out=%d", [ctx.modelPath UTF8String], [ctx.modelKey UTF8String], ctx.qos, ctx.inCount, ctx.outCount);
    return YES;
}

static void teardown_context(ANEContext *ctx) {
    if (ctx == nil || ctx.client == nil || ctx.model == nil) {
        return;
    }
    NSError *err = nil;
    BOOL ok = call_unload(ctx.client, ctx.model, ctx.qos, &err);
    printf("teardown: unloadModel ok=%d err=%s\n", ok, err_text(err));
}

static BOOL run_eval_for_proc(
    ANEContext *ctx,
    const float *input,
    int proc,
    float *out,
    double *mapMs,
    double *evalMs,
    char *detail,
    size_t detailLen
) {
    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    if (inSurf == NULL || outSurf == NULL) {
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        snprintf(detail, detailLen, "failed to create IOSurface(s)");
        return NO;
    }

    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, detailLen)) {
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id req = make_simple_request(inObj, outObj, proc, 0, 0);
    if (req == nil) {
        snprintf(detail, detailLen, "request creation failed proc=%d", proc);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }
    if (!request_validate(req)) {
        snprintf(detail, detailLen, "request validate=false proc=%d", proc);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    NSError *err = nil;
    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
    BOOL ok = call_map(ctx.client, ctx.model, req, &err);
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
    if (mapMs != NULL) {
        *mapMs = (double)(t1 - t0) * 1000.0;
    }
    if (!ok) {
        snprintf(detail, detailLen, "map failed proc=%d err=%s", proc, err_text(err));
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    err = nil;
    t0 = CFAbsoluteTimeGetCurrent();
    ok = call_eval(ctx.client, ctx.model, req, ctx.qos, NO, &err);
    t1 = CFAbsoluteTimeGetCurrent();
    if (evalMs != NULL) {
        *evalMs = (double)(t1 - t0) * 1000.0;
    }
    if (!ok) {
        call_unmap(ctx.client, ctx.model, req);
        snprintf(detail, detailLen, "evaluate failed proc=%d err=%s", proc, err_text(err));
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    if (!read_surface_f32(outSurf, out, ctx.outCount, detail, detailLen)) {
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    call_unmap(ctx.client, ctx.model, req);
    CFRelease(outSurf);
    CFRelease(inSurf);
    return YES;
}

typedef struct {
    BOOL direct;
    BOOL useSimple;
    BOOL useShared;
    BOOL lateAttach;
} PureCfg;

static BOOL run_pure_fenced(
    ANEContext *ctx,
    const float *input,
    PureCfg cfg,
    float *out,
    char *detail,
    size_t detailLen
) {
    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    if (inSurf == NULL || outSurf == NULL) {
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        snprintf(detail, detailLen, "failed to create IOSurface(s)");
        return NO;
    }

    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, detailLen)) {
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);

    id sharedWrapper = nil;
    if (cfg.useShared) {
        id shared = make_shared_event();
        id sig = make_signal_event(1, 0, 5, shared, 0);
        if (sig != nil) {
            sharedWrapper = make_shared_events_wrapper(@[sig]);
        }
    }

    id req = nil;
    if (cfg.useSimple) {
        req = make_simple_request(inObj, outObj, 0, 0, 0);
    } else {
        id preShared = (cfg.useShared && !cfg.lateAttach) ? sharedWrapper : nil;
        req = make_full_request(inObj, outObj, 0, 0, 0, preShared, 1);
    }

    if (req == nil) {
        snprintf(detail, detailLen, "pure fenced request creation failed");
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    if (!request_validate(req)) {
        snprintf(detail, detailLen, "pure fenced request validate=false");
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    NSError *err = nil;
    BOOL ok = call_map(ctx.client, ctx.model, req, &err);
    if (!ok) {
        snprintf(detail, detailLen, "pure fenced map failed err=%s", err_text(err));
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    if (!cfg.useSimple && cfg.useShared && cfg.lateAttach && sharedWrapper != nil && [req respondsToSelector:@selector(setSharedEvents:)]) {
        ((void(*)(id, SEL, id))objc_msgSend)(req, @selector(setSharedEvents:), sharedWrapper);
    }

    id options = make_shared_event_options();
    err = nil;
    ok = call_eval_opt(ctx.client, ctx.model, options, req, ctx.qos, cfg.direct, &err);
    if (!ok) {
        call_unmap(ctx.client, ctx.model, req);
        snprintf(detail, detailLen, "pure fenced evaluate failed err=%s", err_text(err));
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    if (!read_surface_f32(outSurf, out, ctx.outCount, detail, detailLen)) {
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return NO;
    }

    call_unmap(ctx.client, ctx.model, req);
    CFRelease(outSurf);
    CFRelease(inSurf);
    return YES;
}

typedef struct {
    BOOL doCalls;
    BOOL enqueueFirst;
    int proc;
    int freeSeq[8];
    int freeSeqLen;
} ChainCfg;

typedef struct {
    BOOL prepareOK;
    BOOL enqueueOK;
    BOOL readyOK;
    char detail[640];
} ChainOutcome;

static BOOL first_working_prepare_for_model(
    ANEContext *ctx,
    id model,
    id chainReq,
    id options,
    BOOL doCalls,
    char *detail,
    size_t detailLen
) {
    SEL sels[2];
    int selCount = 0;
    if (doCalls) {
        sels[selCount++] = @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:);
    }
    sels[selCount++] = @selector(prepareChainingWithModel:options:chainingReq:qos:error:);

    for (int i = 0; i < selCount; i++) {
        NSError *err = nil;
        BOOL ok = call_prepare_sel_opt(ctx.client, sels[i], model, options, chainReq, ctx.qos, &err);
        if (ok) {
            snprintf(detail, detailLen, "prepare selector=%s", sel_getName(sels[i]));
            return YES;
        }
        snprintf(detail, detailLen, "prepare failed selector=%s err=%s", sel_getName(sels[i]), err_text(err));
    }
    return NO;
}

static BOOL first_working_prepare(
    ANEContext *ctx,
    id chainReq,
    id options,
    BOOL doCalls,
    char *detail,
    size_t detailLen
) {
    return first_working_prepare_for_model(ctx, ctx.model, chainReq, options, doCalls, detail, detailLen);
}

static BOOL try_enqueue_for_model(ANEContext *ctx, id model, id enqueueObj, id options, BOOL doCalls, char *detail, size_t detailLen) {
    SEL sels[2];
    int selCount = 0;
    if (doCalls) {
        sels[selCount++] = @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:);
    }
    sels[selCount++] = @selector(enqueueSetsWithModel:outputSet:options:qos:error:);

    for (int i = 0; i < selCount; i++) {
        NSError *err = nil;
        BOOL ok = call_enqueue_sel_opt(ctx.client, sels[i], model, options, enqueueObj, ctx.qos, &err);
        if (ok) {
            snprintf(detail, detailLen, "enqueue selector=%s", sel_getName(sels[i]));
            return YES;
        }
        snprintf(detail, detailLen, "enqueue failed selector=%s err=%s", sel_getName(sels[i]), err_text(err));
    }
    return NO;
}

static BOOL try_enqueue(ANEContext *ctx, id enqueueObj, id options, BOOL doCalls, char *detail, size_t detailLen) {
    return try_enqueue_for_model(ctx, ctx.model, enqueueObj, options, doCalls, detail, detailLen);
}

static BOOL try_ready_for_model(ANEContext *ctx, id model, NSArray *readyCandidates, id options, BOOL doCalls, char *detail, size_t detailLen) {
    SEL sels[2];
    int selCount = 0;
    if (doCalls) {
        sels[selCount++] = @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:);
    }
    sels[selCount++] = @selector(buffersReadyWithModel:inputBuffers:options:qos:error:);
    if (!doCalls) {
        sels[selCount++] = @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:);
    }

    for (id readyObj in readyCandidates) {
        for (int i = 0; i < selCount; i++) {
            NSError *err = nil;
            BOOL ok = call_ready_sel_opt(ctx.client, sels[i], model, options, readyObj, ctx.qos, &err);
            if (ok) {
                snprintf(detail, detailLen, "ready selector=%s", sel_getName(sels[i]));
                return YES;
            }
            snprintf(detail, detailLen, "ready failed selector=%s err=%s", sel_getName(sels[i]), err_text(err));
        }
    }
    return NO;
}

static BOOL try_ready(ANEContext *ctx, NSArray *readyCandidates, id options, BOOL doCalls, char *detail, size_t detailLen) {
    return try_ready_for_model(ctx, ctx.model, readyCandidates, options, doCalls, detail, detailLen);
}

static ChainOutcome run_single_chaining(ANEContext *ctx, const float *input, ChainCfg cfg) {
    ChainOutcome out;
    memset(&out, 0, sizeof(out));

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef statsSurf = make_surface(1024);
    if (inSurf == NULL || outSurf == NULL || statsSurf == NULL) {
        snprintf(out.detail, sizeof(out.detail), "failed to create IOSurface(s)");
        if (statsSurf != NULL) {
            CFRelease(statsSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        return out;
    }

    if (!write_surface_f32(inSurf, input, ctx.inCount, out.detail, sizeof(out.detail))) {
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id req = make_simple_request(inObj, outObj, cfg.proc, 0, 0);
    if (req == nil || !request_validate(req)) {
        snprintf(out.detail, sizeof(out.detail), "base request creation/validation failed");
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    NSError *err = nil;
    BOOL ok = call_map(ctx.client, ctx.model, req, &err);
    if (!ok) {
        snprintf(out.detail, sizeof(out.detail), "chaining map failed err=%s", err_text(err));
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    id inBuf = make_buffer(inObj, 0, 0);
    id outBuf = make_buffer(outObj, 0, 0);
    id outSet = make_output_set(statsSurf, @[outBuf]);
    if (inBuf == nil || outBuf == nil || outSet == nil) {
        snprintf(out.detail, sizeof(out.detail), "buffer/output-set creation failed");
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    id sharedReady = make_shared_event();
    id sharedFree = make_shared_event();
    if (sharedReady == nil || sharedFree == nil) {
        snprintf(out.detail, sizeof(out.detail), "shared event creation failed");
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    id evReady = make_signal_event(1, 0, 5, sharedReady, 1);
    id evFree = make_signal_event(2, 0, 4, sharedFree, 1);
    if (evReady == nil || evFree == nil) {
        snprintf(out.detail, sizeof(out.detail), "signal event creation failed");
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    id chainReq = make_chaining_request(@[inBuf], @[outSet], nil, nil, cfg.proc, @[evReady, evFree], 1, 0, 0);
    if (chainReq == nil || !request_validate(chainReq)) {
        snprintf(out.detail, sizeof(out.detail), "chaining request creation/validation failed");
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    id options = make_shared_event_options();
    char prepDetail[256] = {0};
    BOOL prepared = first_working_prepare(ctx, chainReq, options, cfg.doCalls, prepDetail, sizeof(prepDetail));
    if (!prepared) {
        snprintf(out.detail, sizeof(out.detail), "%s", prepDetail);
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }
    out.prepareOK = YES;

    id enqueueObj = make_enqueue((unsigned int)cfg.proc, 0, 1, NO, NO);
    if (enqueueObj == nil) {
        snprintf(out.detail, sizeof(out.detail), "enqueue object creation failed");
        call_unmap(ctx.client, ctx.model, req);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return out;
    }

    NSMutableArray *readyCandidates = [NSMutableArray array];
    if (cfg.freeSeqLen <= 0) {
        cfg.freeSeq[0] = 2;
        cfg.freeSeqLen = 1;
    }
    for (int i = 0; i < cfg.freeSeqLen; i++) {
        int fv = cfg.freeSeq[i];
        id r1 = make_ready((unsigned int)cfg.proc, @[@0], @[@(fv)], 0);
        if (r1 != nil) {
            [readyCandidates addObject:r1];
        }
        id r2 = make_ready((unsigned int)cfg.proc, @[@0, @(UINT32_MAX)], @[@((unsigned long long)fv), @(UINT64_MAX)], 0);
        if (r2 != nil) {
            [readyCandidates addObject:r2];
        }
    }

    char stageDetail[256] = {0};
    if (cfg.enqueueFirst) {
        out.enqueueOK = try_enqueue(ctx, enqueueObj, options, cfg.doCalls, stageDetail, sizeof(stageDetail));
        if (!out.enqueueOK) {
            snprintf(out.detail, sizeof(out.detail), "%s | %s", prepDetail, stageDetail);
            call_unmap(ctx.client, ctx.model, req);
            CFRelease(statsSurf);
            CFRelease(outSurf);
            CFRelease(inSurf);
            return out;
        }
        out.readyOK = try_ready(ctx, readyCandidates, options, cfg.doCalls, stageDetail, sizeof(stageDetail));
    } else {
        out.readyOK = try_ready(ctx, readyCandidates, options, cfg.doCalls, stageDetail, sizeof(stageDetail));
        if (out.readyOK) {
            out.enqueueOK = try_enqueue(ctx, enqueueObj, options, cfg.doCalls, stageDetail, sizeof(stageDetail));
        }
    }

    snprintf(
        out.detail,
        sizeof(out.detail),
        "%s | %s | prepare=%d ready=%d enqueue=%d",
        prepDetail,
        stageDetail,
        out.prepareOK,
        out.readyOK,
        out.enqueueOK
    );

    call_unmap(ctx.client, ctx.model, req);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(inSurf);
    return out;
}

static int run_shared_subcase(ANEContext *ctx, NSString *subcase) {
    float *input = make_input(ctx.inCount);
    float *out = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || out == NULL) {
        free(out);
        free(input);
        fprintf(stderr, "subcase allocation failed\n");
        return 2;
    }
    PureCfg cfg = {0};
    if ([subcase isEqualToString:@"path_shared_inline"] || [subcase isEqualToString:@"pure_shared_inline"]) {
        cfg = (PureCfg){.direct = NO, .useSimple = NO, .useShared = YES, .lateAttach = NO};
    } else if ([subcase isEqualToString:@"path_shared_late"] || [subcase isEqualToString:@"pure_shared_late"]) {
        cfg = (PureCfg){.direct = NO, .useSimple = NO, .useShared = YES, .lateAttach = YES};
    } else if ([subcase isEqualToString:@"path_shared_inline_direct"] || [subcase isEqualToString:@"pure_shared_inline_direct"]) {
        cfg = (PureCfg){.direct = YES, .useSimple = NO, .useShared = YES, .lateAttach = NO};
    } else if ([subcase isEqualToString:@"path_shared_late_direct"] || [subcase isEqualToString:@"pure_shared_late_direct"]) {
        cfg = (PureCfg){.direct = YES, .useSimple = NO, .useShared = YES, .lateAttach = YES};
    } else {
        free(out);
        free(input);
        fprintf(stderr, "unknown shared subcase: %s\n", [subcase UTF8String]);
        return 3;
    }
    char detail[256] = {0};
    BOOL ok = run_pure_fenced(ctx, input, cfg, out, detail, sizeof(detail));
    fprintf(stdout, "[SUBCASE] %s ok=%d detail=%s\n", [subcase UTF8String], ok ? 1 : 0, detail[0] ? detail : "-");
    free(out);
    free(input);
    return ok ? 0 : 1;
}

static BOOL run_shared_subprocess(const char *subcase, char *detail, size_t detailLen) {
    if (subcase == NULL || subcase[0] == '\0') {
        snprintf(detail, detailLen, "invalid subcase");
        return NO;
    }
    NSString *exe = NSProcessInfo.processInfo.arguments.firstObject;
    if (exe == nil || exe.length == 0) {
        snprintf(detail, detailLen, "failed to resolve current executable path");
        return NO;
    }
    NSTask *task = [[NSTask alloc] init];
    if (@available(macOS 10.13, *)) {
        task.executableURL = [NSURL fileURLWithPath:exe];
    } else {
        task.launchPath = exe;
    }
    NSMutableDictionary *env = [NSProcessInfo.processInfo.environment mutableCopy];
    env[@"ANE_CHAIN_SUBCASE"] = [NSString stringWithUTF8String:subcase];
    env[@"ANE_CHAIN_ENABLE_SHARED_TESTS"] = @"1";
    task.environment = env;
    NSPipe *pipe = [NSPipe pipe];
    task.standardOutput = pipe;
    task.standardError = pipe;
    @try {
        if (@available(macOS 10.13, *)) {
            [task launchAndReturnError:nil];
        } else {
            [task launch];
        }
    } @catch (NSException *ex) {
        snprintf(detail, detailLen, "subprocess launch exception: %s", [[ex reason] UTF8String]);
        return NO;
    }
    [task waitUntilExit];
    NSData *data = [[pipe fileHandleForReading] readDataToEndOfFile];
    NSString *out = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    if (out == nil) {
        out = @"";
    }
    int code = task.terminationStatus;
    if (code == 0) {
        snprintf(detail, detailLen, "subprocess %s exit=0", subcase);
        return YES;
    }
    snprintf(detail, detailLen, "subprocess %s exit=%d output=%s", subcase, code, out.UTF8String);
    return NO;
}

static CaseResult test_single_procedure_baseline(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *out1 = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *out2 = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || out1 == NULL || out2 == NULL) {
        free(out2);
        free(out1);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    char detail[256] = {0};
    double mapMs = 0;
    double evalMs = 0;
    if (!run_eval_for_proc(ctx, input, 0, out1, &mapMs, &evalMs, detail, sizeof(detail))) {
        free(out2);
        free(out1);
        free(input);
        return make_result(__func__, CaseFail, "baseline_eval failed: %s", detail);
    }

    if (!run_eval_for_proc(ctx, input, 0, out2, NULL, NULL, detail, sizeof(detail))) {
        free(out2);
        free(out1);
        free(input);
        return make_result(__func__, CaseFail, "baseline_eval_repeat failed: %s", detail);
    }

    double diff = max_abs_diff(out1, out2, ctx.outCount);
    free(out2);
    free(out1);
    free(input);
    if (diff > 1e-5) {
        return make_result(__func__, CaseFail, "baseline repeat mismatch maxAbsDiff=%g", diff);
    }
    return make_result(__func__, CasePass, "map=%.3fms eval=%.3fms maxAbsDiff=%g", mapMs, evalMs, diff);
}

typedef struct {
    id modelA;
    id modelB;
    BOOL ownsA;
    BOOL ownsB;
    NSString *pathA;
    NSString *pathB;
} ChainModelPair;

static BOOL setup_chain_models(ANEContext *ctx, ChainModelPair *pair, char *detail, size_t detailLen) {
    if (ctx == nil || pair == NULL) {
        snprintf(detail, detailLen, "invalid setup_chain_models args");
        return NO;
    }
    pair->modelA = nil;
    pair->modelB = nil;
    pair->ownsA = NO;
    pair->ownsB = NO;
    pair->pathA = nil;
    pair->pathB = nil;

    NSString *pathA = env_string("ANE_CHAIN_MODEL_A_PATH", ctx.modelPath);
    NSString *keyA = env_string("ANE_CHAIN_MODEL_A_KEY", ctx.modelKey);
    NSString *pathB = env_string("ANE_CHAIN_MODEL_B_PATH", pathA);
    NSString *keyB = env_string("ANE_CHAIN_MODEL_B_KEY", keyA);

    pair->pathA = pathA;
    pair->pathB = pathB;

    BOOL aIsPrimary = [pathA isEqualToString:ctx.modelPath] && [keyA isEqualToString:ctx.modelKey];
    if (aIsPrimary) {
        pair->modelA = ctx.model;
    } else {
        char loadDetail[320] = {0};
        id model = nil;
        if (!load_model_at_path(ctx.client, pathA, keyA, ctx.qos, &model, loadDetail, sizeof(loadDetail))) {
            snprintf(detail, detailLen, "modelA load failed: %s", loadDetail);
            return NO;
        }
        pair->modelA = model;
        pair->ownsA = YES;
    }

    BOOL bMatchesA = [pathB isEqualToString:pathA] && [keyB isEqualToString:keyA];
    BOOL bIsPrimary = [pathB isEqualToString:ctx.modelPath] && [keyB isEqualToString:ctx.modelKey];
    if (bMatchesA) {
        pair->modelB = pair->modelA;
    } else if (bIsPrimary) {
        pair->modelB = ctx.model;
    } else {
        char loadDetail[320] = {0};
        id model = nil;
        if (!load_model_at_path(ctx.client, pathB, keyB, ctx.qos, &model, loadDetail, sizeof(loadDetail))) {
            if (pair->ownsA) {
                NSError *unloadErr = nil;
                (void)call_unload(ctx.client, pair->modelA, ctx.qos, &unloadErr);
            }
            snprintf(detail, detailLen, "modelB load failed: %s", loadDetail);
            return NO;
        }
        pair->modelB = model;
        pair->ownsB = YES;
    }

    snprintf(
        detail,
        detailLen,
        "models resolved A=%s B=%s distinct=%d",
        pathA.UTF8String,
        pathB.UTF8String,
        pair->modelA != pair->modelB ? 1 : 0
    );
    return YES;
}

static void teardown_chain_models(ANEContext *ctx, ChainModelPair *pair) {
    if (ctx == nil || pair == NULL) {
        return;
    }
    if (pair->ownsB && pair->modelB != nil) {
        NSError *err = nil;
        (void)call_unload(ctx.client, pair->modelB, ctx.qos, &err);
    }
    if (pair->ownsA && pair->modelA != nil && pair->modelA != pair->modelB) {
        NSError *err = nil;
        (void)call_unload(ctx.client, pair->modelA, ctx.qos, &err);
    }
}

static BOOL run_manual_pipeline_models(
    ANEContext *ctx,
    id modelA,
    id modelB,
    const float *input,
    BOOL direct,
    float *out,
    char *detail,
    size_t detailLen
) {
    if (ctx.inCount != ctx.outCount) {
        snprintf(detail, detailLen, "manual pipeline requires inCount==outCount");
        return NO;
    }

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef midSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    if (inSurf == NULL || midSurf == NULL || outSurf == NULL) {
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (midSurf != NULL) {
            CFRelease(midSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        snprintf(detail, detailLen, "failed to create IOSurface(s)");
        return NO;
    }

    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, detailLen)) {
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id midObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), midSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);

    id req1 = make_simple_request(inObj, midObj, 0, 0, 0);
    id req2 = make_simple_request(midObj, outObj, 0, 0, 0);
    if (req1 == nil || req2 == nil || !request_validate(req1) || !request_validate(req2)) {
        snprintf(detail, detailLen, "manual request creation/validate failed");
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }

    NSError *err = nil;
    if (!call_map(ctx.client, modelA, req1, &err)) {
        snprintf(detail, detailLen, "manual map req1 failed: %s", err_text(err));
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }
    err = nil;
    if (!call_map(ctx.client, modelB, req2, &err)) {
        call_unmap(ctx.client, modelA, req1);
        snprintf(detail, detailLen, "manual map req2 failed: %s", err_text(err));
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }

    err = nil;
    BOOL ok = call_eval(ctx.client, modelA, req1, ctx.qos, direct, &err);
    if (!ok) {
        call_unmap(ctx.client, modelB, req2);
        call_unmap(ctx.client, modelA, req1);
        snprintf(detail, detailLen, "manual eval req1 failed: %s", err_text(err));
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }

    err = nil;
    ok = call_eval(ctx.client, modelB, req2, ctx.qos, direct, &err);
    if (!ok) {
        call_unmap(ctx.client, modelB, req2);
        call_unmap(ctx.client, modelA, req1);
        snprintf(detail, detailLen, "manual eval req2 failed: %s", err_text(err));
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }

    if (!read_surface_f32(outSurf, out, ctx.outCount, detail, detailLen)) {
        call_unmap(ctx.client, modelB, req2);
        call_unmap(ctx.client, modelA, req1);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return NO;
    }

    call_unmap(ctx.client, modelB, req2);
    call_unmap(ctx.client, modelA, req1);
    CFRelease(outSurf);
    CFRelease(midSurf);
    CFRelease(inSurf);
    return YES;
}

static BOOL run_manual_pipeline(ANEContext *ctx, const float *input, BOOL direct, float *out, char *detail, size_t detailLen) {
    return run_manual_pipeline_models(ctx, ctx.model, ctx.model, input, direct, out, detail, detailLen);
}

static void fill_input_for_iter(float *dst, int count, int iter) {
    if (dst == NULL || count <= 0) {
        return;
    }
    for (int i = 0; i < count; i++) {
        float base = (float)(((i + (iter * 3)) % 29) - 14) * 0.03125f;
        dst[i] = base + (float)(iter % 7) * 0.015625f;
    }
}

static CaseResult test_multibuffer_rotation_pipeline(ANEContext *ctx) {
    int nFrames = env_int("ANE_CHAIN_MB_FRAMES", 3);
    int iterations = env_int("ANE_CHAIN_MB_ITERS", 100);
    if (nFrames < 2) {
        nFrames = 2;
    }
    if (nFrames > 6) {
        nFrames = 6;
    }
    if (iterations < 8) {
        iterations = 8;
    }

    ChainModelPair models;
    char modelDetail[256] = {0};
    if (!setup_chain_models(ctx, &models, modelDetail, sizeof(modelDetail))) {
        return make_result(__func__, CaseFail, "%s", modelDetail);
    }

    IOSurfaceRef *inSurf = (IOSurfaceRef *)calloc((size_t)nFrames, sizeof(IOSurfaceRef));
    IOSurfaceRef *midSurf = (IOSurfaceRef *)calloc((size_t)nFrames, sizeof(IOSurfaceRef));
    IOSurfaceRef *outSurf = (IOSurfaceRef *)calloc((size_t)nFrames, sizeof(IOSurfaceRef));
    NSMutableArray *reqAList = [NSMutableArray arrayWithCapacity:(NSUInteger)nFrames];
    NSMutableArray *reqBList = [NSMutableArray arrayWithCapacity:(NSUInteger)nFrames];
    if (inSurf == NULL || midSurf == NULL || outSurf == NULL || reqAList == nil || reqBList == nil) {
        free(outSurf);
        free(midSurf);
        free(inSurf);
        teardown_chain_models(ctx, &models);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    BOOL ok = YES;
    char detail[320] = {0};
    for (int f = 0; f < nFrames; f++) {
        inSurf[f] = make_surface((size_t)ctx.inCount * sizeof(float));
        midSurf[f] = make_surface((size_t)ctx.outCount * sizeof(float));
        outSurf[f] = make_surface((size_t)ctx.outCount * sizeof(float));
        if (inSurf[f] == NULL || midSurf[f] == NULL || outSurf[f] == NULL) {
            snprintf(detail, sizeof(detail), "surface allocation failed frame=%d", f);
            ok = NO;
            break;
        }
        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf[f]);
        id midObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), midSurf[f]);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf[f]);
        id reqAObj = make_simple_request(inObj, midObj, 0, 0, 0);
        id reqBObj = make_simple_request(midObj, outObj, 0, 0, 0);
        if (reqAObj == nil || reqBObj == nil || !request_validate(reqAObj) || !request_validate(reqBObj)) {
            snprintf(detail, sizeof(detail), "request create/validate failed frame=%d", f);
            ok = NO;
            break;
        }
        [reqAList addObject:reqAObj];
        [reqBList addObject:reqBObj];
        NSError *err = nil;
        if (!call_map(ctx.client, models.modelA, reqAObj, &err)) {
            snprintf(detail, sizeof(detail), "map A failed frame=%d err=%s", f, err_text(err));
            ok = NO;
            break;
        }
        err = nil;
        if (!call_map(ctx.client, models.modelB, reqBObj, &err)) {
            call_unmap(ctx.client, models.modelA, reqAObj);
            snprintf(detail, sizeof(detail), "map B failed frame=%d err=%s", f, err_text(err));
            ok = NO;
            break;
        }
    }

    float *scratchIn = (float *)calloc((size_t)ctx.inCount, sizeof(float));
    float *baselineOut = (float *)calloc((size_t)iterations * (size_t)ctx.outCount, sizeof(float));
    float *pipelinedOut = (float *)calloc((size_t)iterations * (size_t)ctx.outCount, sizeof(float));
    if (ok && (scratchIn == NULL || baselineOut == NULL || pipelinedOut == NULL)) {
        snprintf(detail, sizeof(detail), "output buffer allocation failed");
        ok = NO;
    }

    double baselineMs = 0.0;
    double pipelinedMs = 0.0;
    if (ok) {
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < iterations; i++) {
            int frame = i % nFrames;
            id reqAObj = reqAList[(NSUInteger)frame];
            id reqBObj = reqBList[(NSUInteger)frame];
            fill_input_for_iter(scratchIn, ctx.inCount, i);
            if (!write_surface_f32(inSurf[frame], scratchIn, ctx.inCount, detail, sizeof(detail))) {
                ok = NO;
                break;
            }
            NSError *err = nil;
            if (!call_eval(ctx.client, models.modelA, reqAObj, ctx.qos, NO, &err)) {
                snprintf(detail, sizeof(detail), "baseline eval A failed iter=%d err=%s", i, err_text(err));
                ok = NO;
                break;
            }
            err = nil;
            if (!call_eval(ctx.client, models.modelB, reqBObj, ctx.qos, NO, &err)) {
                snprintf(detail, sizeof(detail), "baseline eval B failed iter=%d err=%s", i, err_text(err));
                ok = NO;
                break;
            }
            if (!read_surface_f32(outSurf[frame], baselineOut + ((size_t)i * (size_t)ctx.outCount), ctx.outCount, detail, sizeof(detail))) {
                ok = NO;
                break;
            }
        }
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
        baselineMs = (double)(t1 - t0) * 1000.0;
    }

    if (ok) {
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i <= iterations; i++) {
            if (i < iterations) {
                int frameA = i % nFrames;
                id reqAObj = reqAList[(NSUInteger)frameA];
                fill_input_for_iter(scratchIn, ctx.inCount, i);
                if (!write_surface_f32(inSurf[frameA], scratchIn, ctx.inCount, detail, sizeof(detail))) {
                    ok = NO;
                    break;
                }
                NSError *err = nil;
                if (!call_eval(ctx.client, models.modelA, reqAObj, ctx.qos, NO, &err)) {
                    snprintf(detail, sizeof(detail), "pipelined eval A failed iter=%d err=%s", i, err_text(err));
                    ok = NO;
                    break;
                }
            }
            if (i > 0) {
                int frameB = (i - 1) % nFrames;
                id reqBObj = reqBList[(NSUInteger)frameB];
                NSError *err = nil;
                if (!call_eval(ctx.client, models.modelB, reqBObj, ctx.qos, NO, &err)) {
                    snprintf(detail, sizeof(detail), "pipelined eval B failed iter=%d err=%s", i - 1, err_text(err));
                    ok = NO;
                    break;
                }
                if (!read_surface_f32(outSurf[frameB], pipelinedOut + ((size_t)(i - 1) * (size_t)ctx.outCount), ctx.outCount, detail, sizeof(detail))) {
                    ok = NO;
                    break;
                }
            }
        }
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
        pipelinedMs = (double)(t1 - t0) * 1000.0;
    }

    for (int f = 0; f < nFrames; f++) {
        if ((NSUInteger)f < reqBList.count) {
            call_unmap(ctx.client, models.modelB, reqBList[(NSUInteger)f]);
        }
        if ((NSUInteger)f < reqAList.count) {
            call_unmap(ctx.client, models.modelA, reqAList[(NSUInteger)f]);
        }
    }

    for (int f = 0; f < nFrames; f++) {
        if (outSurf[f] != NULL) {
            CFRelease(outSurf[f]);
        }
        if (midSurf[f] != NULL) {
            CFRelease(midSurf[f]);
        }
        if (inSurf[f] != NULL) {
            CFRelease(inSurf[f]);
        }
    }

    double maxDiff = 0.0;
    if (ok) {
        maxDiff = max_abs_diff(baselineOut, pipelinedOut, iterations * ctx.outCount);
    }
    double speedup = (ok && pipelinedMs > 0.0) ? baselineMs / pipelinedMs : 0.0;

    free(pipelinedOut);
    free(baselineOut);
    free(scratchIn);
    free(outSurf);
    free(midSurf);
    free(inSurf);
    teardown_chain_models(ctx, &models);

    if (!ok) {
        return make_result(__func__, CaseXFail, "multi-buffer run blocked: %s (%s)", detail, modelDetail);
    }
    if (maxDiff > 3e-2) {
        return make_result(__func__, CaseFail, "multi-buffer output mismatch maxAbsDiff=%g (%s)", maxDiff, modelDetail);
    }
    if (speedup >= 1.5) {
        return make_result(__func__, CasePass, "multi-buffer speedup=%.3fx baseline=%.3fms pipelined=%.3fms diff=%g (%s)", speedup, baselineMs, pipelinedMs, maxDiff, modelDetail);
    }
    return make_result(__func__, CaseXFail, "multi-buffer speedup below target: %.3fx baseline=%.3fms pipelined=%.3fms diff=%g (%s)", speedup, baselineMs, pipelinedMs, maxDiff, modelDetail);
}

static CaseResult test_manual_iosurface_pipeline(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *stage1 = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *want = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *gotEval = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *gotDirect = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || stage1 == NULL || want == NULL || gotEval == NULL || gotDirect == NULL) {
        free(gotDirect);
        free(gotEval);
        free(want);
        free(stage1);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    char detail[256] = {0};
    if (!run_eval_for_proc(ctx, input, 0, stage1, NULL, NULL, detail, sizeof(detail))) {
        free(gotDirect);
        free(gotEval);
        free(want);
        free(stage1);
        free(input);
        return make_result(__func__, CaseFail, "stage1 baseline failed: %s", detail);
    }
    if (!run_eval_for_proc(ctx, stage1, 0, want, NULL, NULL, detail, sizeof(detail))) {
        free(gotDirect);
        free(gotEval);
        free(want);
        free(stage1);
        free(input);
        return make_result(__func__, CaseFail, "stage2 baseline failed: %s", detail);
    }

    if (!run_manual_pipeline(ctx, input, NO, gotEval, detail, sizeof(detail))) {
        free(gotDirect);
        free(gotEval);
        free(want);
        free(stage1);
        free(input);
        return make_result(__func__, CaseFail, "manual evaluate failed: %s", detail);
    }
    if (!run_manual_pipeline(ctx, input, YES, gotDirect, detail, sizeof(detail))) {
        free(gotDirect);
        free(gotEval);
        free(want);
        free(stage1);
        free(input);
        return make_result(__func__, CaseFail, "manual evaluate_direct failed: %s", detail);
    }

    double dEval = max_abs_diff(gotEval, want, ctx.outCount);
    double dDirect = max_abs_diff(gotDirect, want, ctx.outCount);

    free(gotDirect);
    free(gotEval);
    free(want);
    free(stage1);
    free(input);

    if (dEval > 3e-2 || dDirect > 3e-2) {
        return make_result(__func__, CaseFail, "manual diff too high eval=%g direct=%g", dEval, dDirect);
    }
    return make_result(__func__, CasePass, "manual pipeline diffs eval=%g direct=%g", dEval, dDirect);
}

static CaseResult test_health_gate(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *out = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || out == NULL) {
        free(out);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }
    char detail[256] = {0};
    BOOL ok = run_eval_for_proc(ctx, input, 0, out, NULL, NULL, detail, sizeof(detail));
    free(out);
    free(input);
    if (!ok) {
        return make_result(__func__, CaseFail, "%s", detail);
    }
    return make_result(__func__, CasePass, "ANE map/eval baseline path works");
}

static CaseResult test_single_procedure_path_matrix(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *baseline = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *pure = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || baseline == NULL || pure == NULL) {
        free(pure);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    char detail[320] = {0};
    if (!run_eval_for_proc(ctx, input, 0, baseline, NULL, NULL, detail, sizeof(detail))) {
        free(pure);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "baseline_eval failed: %s", detail);
    }

    PureCfg cfgSimple = {.direct = NO, .useSimple = YES, .useShared = NO, .lateAttach = NO};
    if (!run_pure_fenced(ctx, input, cfgSimple, pure, detail, sizeof(detail))) {
        free(pure);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "pure_simple failed: %s", detail);
    }

    double diff = max_abs_diff(pure, baseline, ctx.outCount);
    BOOL strictShared = env_enabled("ANE_CHAIN_ENABLE_SHARED_TESTS");

    PureCfg cfgSimpleDirect = {.direct = YES, .useSimple = YES, .useShared = NO, .lateAttach = NO};
    BOOL directOK = run_pure_fenced(ctx, input, cfgSimpleDirect, pure, detail, sizeof(detail));
    char directNote[200] = {0};
    if (!directOK) {
        snprintf(directNote, sizeof(directNote), " pure_simple_direct exploratory fail: %s;", detail);
    }

    char sharedNote[220] = {0};
    if (strictShared) {
        char subDetailA[256] = {0};
        char subDetailB[256] = {0};
        BOOL sh1 = run_shared_subprocess("path_shared_inline", subDetailA, sizeof(subDetailA));
        BOOL sh2 = run_shared_subprocess("path_shared_late", subDetailB, sizeof(subDetailB));
        if (!sh1 || !sh2) {
            snprintf(sharedNote, sizeof(sharedNote), " shared exploratory fail inline=%d late=%d (%s | %s);", sh1, sh2, subDetailA, subDetailB);
        }
    } else {
        snprintf(sharedNote, sizeof(sharedNote), " shared cases skipped (ANE_CHAIN_ENABLE_SHARED_TESTS=1 to run);");
    }

    free(pure);
    free(baseline);
    free(input);

    if (diff > kCompareTolerance) {
        return make_result(__func__, CaseFail, "pure_simple diff=%g%s%s", diff, directNote, sharedNote);
    }
    return make_result(__func__, CasePass, "pure_simple diff=%g%s%s", diff, directNote, sharedNote);
}

static CaseResult test_pure_fenced_eval_matrix(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *baseline = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *out = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || baseline == NULL || out == NULL) {
        free(out);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    char detail[320] = {0};
    if (!run_eval_for_proc(ctx, input, 0, baseline, NULL, NULL, detail, sizeof(detail))) {
        free(out);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "baseline_eval failed: %s", detail);
    }

    PureCfg simpleNoShared = {.direct = NO, .useSimple = YES, .useShared = NO, .lateAttach = NO};
    if (!run_pure_fenced(ctx, input, simpleNoShared, out, detail, sizeof(detail))) {
        free(out);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "simple_no_shared failed: %s", detail);
    }
    double dSimple = max_abs_diff(out, baseline, ctx.outCount);

    char notes[320] = {0};
    PureCfg nonsimpleNoSharedLate = {.direct = NO, .useSimple = NO, .useShared = NO, .lateAttach = YES};
    BOOL nsOK = run_pure_fenced(ctx, input, nonsimpleNoSharedLate, out, detail, sizeof(detail));
    if (!nsOK) {
        strlcat(notes, " nonsimple_no_shared_late_attach exploratory fail;", sizeof(notes));
    }

    PureCfg simpleDirect = {.direct = YES, .useSimple = YES, .useShared = NO, .lateAttach = NO};
    BOOL sdOK = run_pure_fenced(ctx, input, simpleDirect, out, detail, sizeof(detail));
    if (!sdOK) {
        strlcat(notes, " simple_no_shared_direct exploratory fail;", sizeof(notes));
    }

    if (env_enabled("ANE_CHAIN_ENABLE_SHARED_TESTS")) {
        char subDetailA[256] = {0};
        char subDetailB[256] = {0};
        BOOL sh1 = run_shared_subprocess("pure_shared_inline", subDetailA, sizeof(subDetailA));
        BOOL sh2 = run_shared_subprocess("pure_shared_late", subDetailB, sizeof(subDetailB));
        if (!sh1 || !sh2) {
            strlcat(notes, " shared exploratory failures observed;", sizeof(notes));
        }
    } else {
        strlcat(notes, " shared cases skipped;", sizeof(notes));
    }

    free(out);
    free(baseline);
    free(input);

    if (dSimple > kCompareTolerance) {
        return make_result(__func__, CaseFail, "simple_no_shared diff=%g%s", dSimple, notes);
    }
    return make_result(__func__, CasePass, "simple_no_shared diff=%g%s", dSimple, notes);
}

static CaseResult test_release_readiness(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *baseline = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    float *pure = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || baseline == NULL || pure == NULL) {
        free(pure);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    char detail[256] = {0};
    if (!run_eval_for_proc(ctx, input, 0, baseline, NULL, NULL, detail, sizeof(detail))) {
        free(pure);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "baseline failed: %s", detail);
    }
    PureCfg cfg = {.direct = NO, .useSimple = YES, .useShared = NO, .lateAttach = NO};
    if (!run_pure_fenced(ctx, input, cfg, pure, detail, sizeof(detail))) {
        free(pure);
        free(baseline);
        free(input);
        return make_result(__func__, CaseFail, "pure fenced failed: %s", detail);
    }
    double diff = max_abs_diff(baseline, pure, ctx.outCount);

    free(pure);
    free(baseline);
    free(input);

    if (diff > kCompareTolerance) {
        return make_result(__func__, CaseFail, "release readiness diff=%g", diff);
    }
    return make_result(__func__, CasePass, "release readiness diff=%g", diff);
}

static CaseResult test_seq_map_threshold_sweep(ANEContext *ctx) {
    if (!env_enabled("ANE_CHAIN_ENABLE_SEQ_SWEEP")) {
        return make_result(__func__, CaseSkip, "set ANE_CHAIN_ENABLE_SEQ_SWEEP=1 to run seq threshold sweep");
    }

    const int channels = env_int("ANE_CHAIN_SWEEP_CHANNELS", 64);
    const int seqMin = env_int("ANE_CHAIN_SWEEP_MIN", 1);
    const int seqMax = env_int("ANE_CHAIN_SWEEP_MAX", 32);
    const int expectFloor = env_int("ANE_CHAIN_EXPECTED_MIN_SEQ", 16);
    NSString *csvPath = env_string("ANE_CHAIN_SWEEP_CSV", @"");
    if (channels <= 0 || seqMin <= 0 || seqMax < seqMin) {
        return make_result(__func__, CaseFail, "invalid sweep config channels=%d seqMin=%d seqMax=%d", channels, seqMin, seqMax);
    }

    int firstSuccess = -1;
    int mapFailureCount = 0;
    char summary[1024] = {0};
    FILE *csv = NULL;
    if (csvPath.length > 0) {
        csv = fopen(csvPath.UTF8String, "w");
        if (csv == NULL) {
            return make_result(__func__, CaseFail, "failed to open ANE_CHAIN_SWEEP_CSV at %s", csvPath.UTF8String);
        }
        fprintf(csv, "seq,count,map_ok,is_0x1d,first_success_so_far,error_code,error_domain\n");
    }

#define CLOSE_SWEEP_CSV() \
    do { \
        if (csv != NULL) { \
            fclose(csv); \
            csv = NULL; \
        } \
    } while (0)

    for (int seq = seqMin; seq <= seqMax; seq++) {
        int count = channels * seq;
        float *input = make_input(count);
        if (input == NULL) {
            CLOSE_SWEEP_CSV();
            return make_result(__func__, CaseFail, "allocation failed for seq=%d count=%d", seq, count);
        }

        IOSurfaceRef inSurf = make_surface((size_t)count * sizeof(float));
        IOSurfaceRef outSurf = make_surface((size_t)count * sizeof(float));
        if (inSurf == NULL || outSurf == NULL) {
            free(input);
            if (outSurf != NULL) {
                CFRelease(outSurf);
            }
            if (inSurf != NULL) {
                CFRelease(inSurf);
            }
            CLOSE_SWEEP_CSV();
            return make_result(__func__, CaseFail, "surface creation failed for seq=%d count=%d", seq, count);
        }

        char detail[256] = {0};
        if (!write_surface_f32(inSurf, input, count, detail, sizeof(detail))) {
            free(input);
            CFRelease(outSurf);
            CFRelease(inSurf);
            CLOSE_SWEEP_CSV();
            return make_result(__func__, CaseFail, "write input failed for seq=%d: %s", seq, detail);
        }

        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
        id req = make_simple_request(inObj, outObj, 0, 0, 0);
        if (req == nil || !request_validate(req)) {
            free(input);
            CFRelease(outSurf);
            CFRelease(inSurf);
            CLOSE_SWEEP_CSV();
            return make_result(__func__, CaseFail, "request create/validate failed for seq=%d", seq);
        }

        NSError *err = nil;
        BOOL ok = call_map(ctx.client, ctx.model, req, &err);
        BOOL mapFail1D = NO;
        if (ok) {
            call_unmap(ctx.client, ctx.model, req);
            if (firstSuccess < 0) {
                firstSuccess = seq;
            }
            char line[64] = {0};
            snprintf(line, sizeof(line), " [s=%d ok]", seq);
            strlcat(summary, line, sizeof(summary));
        } else if (is_program_iosurface_map_failure(err)) {
            mapFail1D = YES;
            mapFailureCount++;
            char line[64] = {0};
            snprintf(line, sizeof(line), " [s=%d 0x1D]", seq);
            strlcat(summary, line, sizeof(summary));
        } else {
            free(input);
            CFRelease(outSurf);
            CFRelease(inSurf);
            if (csv != NULL) {
                long code = err != nil ? (long)err.code : 0;
                const char *domain = (err != nil && err.domain != nil) ? err.domain.UTF8String : "";
                fprintf(csv, "%d,%d,%d,%d,%d,%ld,%s\n", seq, count, ok ? 1 : 0, 0, firstSuccess, code, domain);
            }
            CLOSE_SWEEP_CSV();
            return make_result(__func__, CaseFail, "unexpected map failure at seq=%d err=%s", seq, err_text(err));
        }

        if (csv != NULL) {
            long code = err != nil ? (long)err.code : 0;
            const char *domain = (err != nil && err.domain != nil) ? err.domain.UTF8String : "";
            fprintf(csv, "%d,%d,%d,%d,%d,%ld,%s\n", seq, count, ok ? 1 : 0, mapFail1D ? 1 : 0, firstSuccess, code, domain);
        }

        free(input);
        CFRelease(outSurf);
        CFRelease(inSurf);
    }

    if (firstSuccess < 0) {
        CLOSE_SWEEP_CSV();
        return make_result(__func__, CaseXFail, "no successful map in seq range [%d,%d]; mapFailures=%d%s", seqMin, seqMax, mapFailureCount, summary);
    }

    if (expectFloor > 0 && firstSuccess != expectFloor) {
        CLOSE_SWEEP_CSV();
        return make_result(__func__, CaseFail, "seq floor changed: expected=%d got=%d; channels=%d%s", expectFloor, firstSuccess, channels, summary);
    }

    CLOSE_SWEEP_CSV();
    return make_result(__func__, CasePass, "seq floor=%d channels=%d mapFailures=%d%s", firstSuccess, channels, mapFailureCount, summary);

#undef CLOSE_SWEEP_CSV
}

static CaseResult test_inmemory_model_chaining_selector_probe(ANEContext *ctx) {
    (void)ctx;
    Class inMemoryModel = NSClassFromString(@"_ANEInMemoryModel");
    if (inMemoryModel == Nil) {
        return make_result(__func__, CaseXFail, "_ANEInMemoryModel class missing");
    }

    const char *candidates[] = {
        "prepareChainingWithModel:options:chainingReq:qos:error:",
        "doPrepareChainingWithModel:options:chainingReq:qos:error:",
        "enqueueSetsWithModel:outputSet:options:qos:error:",
        "doEnqueueSetsWithModel:outputSet:options:qos:error:",
        "buffersReadyWithModel:inputBuffers:options:qos:error:",
        "doBuffersReadyWithModel:inputBuffers:options:qos:error:",
        "mapIOSurfacesWithModel:chainingReq:cacheInference:error:",
    };

    int present = 0;
    char presentList[512] = {0};
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        SEL sel = sel_registerName(candidates[i]);
        if ([inMemoryModel instancesRespondToSelector:sel]) {
            if (present > 0) {
                strlcat(presentList, ",", sizeof(presentList));
            }
            strlcat(presentList, candidates[i], sizeof(presentList));
            present++;
        }
    }

    if (present == 0) {
        return make_result(__func__, CaseXFail, "_ANEInMemoryModel exposes no chaining selectors from probe set");
    }
    return make_result(__func__, CasePass, "in-memory chaining selectors found=%d [%s]", present, presentList);
}

static CaseResult test_chaining_map_selector_probe(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef statsSurf = make_surface(1024);
    if (inSurf == NULL || outSurf == NULL || statsSurf == NULL) {
        free(input);
        if (statsSurf != NULL) {
            CFRelease(statsSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        return make_result(__func__, CaseFail, "surface creation failed");
    }

    char detail[320] = {0};
    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, sizeof(detail))) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "%s", detail);
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id req = make_simple_request(inObj, outObj, 0, 0, 0);
    if (req == nil || !request_validate(req)) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "request create/validate failed");
    }

    NSError *err = nil;
    if (!call_map(ctx.client, ctx.model, req, &err)) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "base map failed: %s", err_text(err));
    }

    id inBuf = make_buffer(inObj, 0, 0);
    id outBuf = make_buffer(outObj, 0, 0);
    id outSet = make_output_set(statsSurf, @[outBuf]);
    id sharedReady = make_shared_event();
    id sharedFree = make_shared_event();
    id evReady = make_signal_event(1, 0, 5, sharedReady, 1);
    id evFree = make_signal_event(2, 0, 4, sharedFree, 1);
    id chainReq = make_chaining_request(@[inBuf], @[outSet], @[@0], @[@0], 0, @[evReady, evFree], 1, 0, 0);
    if (chainReq == nil || !request_validate(chainReq)) {
        call_unmap(ctx.client, ctx.model, req);
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "chaining request create/validate failed");
    }

    const char *mapCandidates[] = {
        "mapIOSurfacesWithModel:chainingReq:cacheInference:error:",
        "mapIOSurfacesWithModel:chainingRequest:cacheInference:error:",
        "mapChainingIOSurfacesWithModel:chainingReq:cacheInference:error:",
        "mapChainingBuffersWithModel:chainingReq:cacheInference:error:",
    };

    BOOL sawCandidate = NO;
    BOOL mapWorked = NO;
    BOOL chainWorked = NO;
    char notes[512] = {0};

    id options = make_shared_event_options();
    id enqueueObj = make_enqueue(0, 0, 1, NO, NO);
    id readyObjA = make_ready(0, @[@0], @[@2], 0);
    id readyObjB = make_ready(0, @[@0, @(UINT32_MAX)], @[@2, @(UINT64_MAX)], 0);
    NSArray *readyCandidates = @[readyObjA ?: [NSNull null], readyObjB ?: [NSNull null]];

    for (size_t i = 0; i < sizeof(mapCandidates) / sizeof(mapCandidates[0]); i++) {
        SEL sel = sel_registerName(mapCandidates[i]);
        if (![ctx.client respondsToSelector:sel]) {
            continue;
        }
        sawCandidate = YES;
        err = nil;
        BOOL altMapOK = call_map_alt_sel(ctx.client, sel, ctx.model, chainReq, &err);
        char line[180] = {0};
        snprintf(line, sizeof(line), " [%s ok=%d err=%s]", mapCandidates[i], altMapOK ? 1 : 0, err_text(err));
        strlcat(notes, line, sizeof(notes));
        if (!altMapOK) {
            continue;
        }
        mapWorked = YES;
        char prepDetail[200] = {0};
        if (!first_working_prepare(ctx, chainReq, options, NO, prepDetail, sizeof(prepDetail))) {
            continue;
        }
        char stageDetail[200] = {0};
        BOOL enqueueOK = try_enqueue(ctx, enqueueObj, options, NO, stageDetail, sizeof(stageDetail));
        BOOL readyOK = NO;
        if (enqueueOK) {
            NSMutableArray *validReady = [NSMutableArray arrayWithCapacity:2];
            for (id obj in readyCandidates) {
                if (obj != (id)[NSNull null]) {
                    [validReady addObject:obj];
                }
            }
            readyOK = try_ready(ctx, validReady, options, NO, stageDetail, sizeof(stageDetail));
        }
        if (enqueueOK && readyOK) {
            chainWorked = YES;
            break;
        }
    }

    call_unmap(ctx.client, ctx.model, req);
    free(input);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(inSurf);

    if (chainWorked) {
        return make_result(__func__, CasePass, "chaining map selector succeeded;%s", notes);
    }
    if (!sawCandidate) {
        return make_result(__func__, CaseXFail, "no chaining map selector found on _ANEClient");
    }
    if (!mapWorked) {
        return make_result(__func__, CaseXFail, "chaining map selectors present but call failed;%s", notes);
    }
    return make_result(__func__, CaseXFail, "chaining map selector call succeeded but prepare/enqueue/ready still blocked;%s", notes);
}

static CaseResult test_two_model_chaining_probe(ANEContext *ctx) {
    ChainModelPair models;
    char modelDetail[256] = {0};
    if (!setup_chain_models(ctx, &models, modelDetail, sizeof(modelDetail))) {
        return make_result(__func__, CaseFail, "%s", modelDetail);
    }

    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        teardown_chain_models(ctx, &models);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef midSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef statsSurf = make_surface(1024);
    if (inSurf == NULL || midSurf == NULL || outSurf == NULL || statsSurf == NULL) {
        free(input);
        teardown_chain_models(ctx, &models);
        if (statsSurf != NULL) {
            CFRelease(statsSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (midSurf != NULL) {
            CFRelease(midSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        return make_result(__func__, CaseFail, "surface creation failed");
    }

    char detail[320] = {0};
    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, sizeof(detail))) {
        free(input);
        teardown_chain_models(ctx, &models);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "%s", detail);
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id midObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), midSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id reqA = make_simple_request(inObj, midObj, 0, 0, 0);
    id reqB = make_simple_request(midObj, outObj, 0, 0, 0);
    if (reqA == nil || reqB == nil || !request_validate(reqA) || !request_validate(reqB)) {
        free(input);
        teardown_chain_models(ctx, &models);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "reqA/reqB create/validate failed");
    }

    NSError *err = nil;
    if (!call_map(ctx.client, models.modelA, reqA, &err)) {
        free(input);
        teardown_chain_models(ctx, &models);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "map modelA failed: %s", err_text(err));
    }
    err = nil;
    if (!call_map(ctx.client, models.modelB, reqB, &err)) {
        call_unmap(ctx.client, models.modelA, reqA);
        free(input);
        teardown_chain_models(ctx, &models);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "map modelB failed: %s", err_text(err));
    }

    id inBuf = make_buffer(inObj, 0, 0);
    id outBuf = make_buffer(outObj, 0, 1);
    id outSet = make_output_set(statsSurf, @[outBuf]);
    id sharedReady = make_shared_event();
    id sharedFree = make_shared_event();
    id evReady = make_signal_event(1, 0, 5, sharedReady, 1);
    id evFree = make_signal_event(2, 0, 4, sharedFree, 1);
    id chainReq = make_chaining_request(@[inBuf], @[outSet], @[@0], @[@0], 0, @[evReady, evFree], 1, 0, 0);
    if (chainReq == nil || !request_validate(chainReq)) {
        call_unmap(ctx.client, models.modelB, reqB);
        call_unmap(ctx.client, models.modelA, reqA);
        free(input);
        teardown_chain_models(ctx, &models);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(midSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "two-model chain request create/validate failed");
    }

    id options = make_shared_event_options();
    id enqueueObj = make_enqueue(0, 0, 1, NO, NO);
    id readyObj = make_ready(0, @[@0], @[@2], 0);
    id readyObjWide = make_ready(0, @[@0, @(UINT32_MAX)], @[@2, @(UINT64_MAX)], 0);
    NSArray *readyCandidates = @[readyObj ?: [NSNull null], readyObjWide ?: [NSNull null]];

    struct {
        const char *name;
        BOOL prepareA;
        BOOL prepareB;
    } variants[] = {
        {"prepare_A", YES, NO},
        {"prepare_B", NO, YES},
        {"prepare_A_then_B", YES, YES},
    };

    BOOL fullSuccess = NO;
    char summary[640] = {0};
    for (size_t i = 0; i < sizeof(variants) / sizeof(variants[0]); i++) {
        char prepA[160] = {0};
        char prepB[160] = {0};
        BOOL pA = !variants[i].prepareA || first_working_prepare_for_model(ctx, models.modelA, chainReq, options, NO, prepA, sizeof(prepA));
        BOOL pB = !variants[i].prepareB || first_working_prepare_for_model(ctx, models.modelB, chainReq, options, NO, prepB, sizeof(prepB));

        BOOL enqueueOK = NO;
        BOOL readyOK = NO;
        if (pA && pB) {
            char stage[160] = {0};
            enqueueOK = try_enqueue_for_model(ctx, models.modelB, enqueueObj, options, NO, stage, sizeof(stage));
            if (enqueueOK) {
                NSMutableArray *validReady = [NSMutableArray arrayWithCapacity:2];
                for (id obj in readyCandidates) {
                    if (obj != (id)[NSNull null]) {
                        [validReady addObject:obj];
                    }
                }
                readyOK = try_ready_for_model(ctx, models.modelA, validReady, options, NO, stage, sizeof(stage));
            }
        }

        char line[220] = {0};
        snprintf(line, sizeof(line), " [%s pA=%d pB=%d enq=%d ready=%d]", variants[i].name, pA ? 1 : 0, pB ? 1 : 0, enqueueOK ? 1 : 0, readyOK ? 1 : 0);
        strlcat(summary, line, sizeof(summary));
        if (pA && pB && enqueueOK && readyOK) {
            fullSuccess = YES;
            break;
        }
    }

    call_unmap(ctx.client, models.modelB, reqB);
    call_unmap(ctx.client, models.modelA, reqA);
    free(input);
    teardown_chain_models(ctx, &models);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(midSurf);
    CFRelease(inSurf);

    if (fullSuccess) {
        return make_result(__func__, CasePass, "two-model chaining path succeeded;%s (%s)", summary, modelDetail);
    }
    return make_result(__func__, CaseXFail, "two-model prepare variants still blocked;%s (%s)", summary, modelDetail);
}

static CaseResult test_single_procedure_chaining_matrix(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }

    ChainCfg cases[] = {
        {.doCalls = NO, .enqueueFirst = NO, .proc = 0, .freeSeq = {2}, .freeSeqLen = 1},
        {.doCalls = NO, .enqueueFirst = YES, .proc = 0, .freeSeq = {2}, .freeSeqLen = 1},
        {.doCalls = YES, .enqueueFirst = NO, .proc = 0, .freeSeq = {2}, .freeSeqLen = 1},
    };
    const char *labels[] = {"default_sequence", "enqueue_first", "do_calls"};

    int success = 0;
    char summary[512] = {0};
    for (int i = 0; i < 3; i++) {
        ChainOutcome o = run_single_chaining(ctx, input, cases[i]);
        char line[180] = {0};
        snprintf(line, sizeof(line), " [%s p=%d r=%d e=%d]", labels[i], o.prepareOK, o.readyOK, o.enqueueOK);
        strlcat(summary, line, sizeof(summary));
        if (o.prepareOK && o.readyOK && o.enqueueOK) {
            success++;
        }
    }

    free(input);
    if (success == 0) {
        return make_result(__func__, CaseXFail, "no full chaining success in exploratory matrix;%s", summary);
    }
    return make_result(__func__, CasePass, "full chaining success=%d;%s", success, summary);
}

static CaseResult test_single_procedure_buffers_ready_matrix(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }

    struct {
        const char *name;
        BOOL enqueueFirst;
        int seq[4];
        int seqLen;
    } cases[] = {
        {"enqueue_first_free0", YES, {0}, 1},
        {"enqueue_first_free1", YES, {1}, 1},
        {"enqueue_first_free2", YES, {2}, 1},
        {"enqueue_first_free_seq_1_2", YES, {1, 2}, 2},
        {"default_order_free1", NO, {1}, 1},
    };

    int readySuccess = 0;
    char summary[512] = {0};
    for (int i = 0; i < 5; i++) {
        ChainCfg cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.doCalls = NO;
        cfg.enqueueFirst = cases[i].enqueueFirst;
        cfg.proc = 0;
        cfg.freeSeqLen = cases[i].seqLen;
        for (int j = 0; j < cfg.freeSeqLen; j++) {
            cfg.freeSeq[j] = cases[i].seq[j];
        }
        ChainOutcome o = run_single_chaining(ctx, input, cfg);
        char line[180] = {0};
        snprintf(line, sizeof(line), " [%s p=%d r=%d e=%d]", cases[i].name, o.prepareOK, o.readyOK, o.enqueueOK);
        strlcat(summary, line, sizeof(summary));
        if (o.readyOK) {
            readySuccess++;
        }
    }

    free(input);
    if (readySuccess == 0) {
        return make_result(__func__, CaseXFail, "no buffersReady success in exploratory matrix;%s", summary);
    }
    return make_result(__func__, CasePass, "buffersReady successes=%d;%s", readySuccess, summary);
}

static CaseResult test_chaining_spike_single_procedure(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }
    ChainCfg cfg = {.doCalls = NO, .enqueueFirst = NO, .proc = 0, .freeSeq = {2}, .freeSeqLen = 1};
    ChainOutcome o = run_single_chaining(ctx, input, cfg);
    free(input);

    if (!(o.prepareOK && o.readyOK && o.enqueueOK)) {
        return make_result(__func__, CaseXFail, "exploratory chaining spike did not complete: %s", o.detail);
    }
    return make_result(__func__, CasePass, "chaining spike completed: %s", o.detail);
}

static CaseResult test_chaining_spike_multi_procedure(ANEContext *ctx) {
    (void)ctx;
    if (!env_enabled("ANE_CHAIN_ENABLE_MULTIPROC")) {
        return make_result(__func__, CaseSkip, "set ANE_CHAIN_ENABLE_MULTIPROC=1 to enable multi-procedure attempt");
    }
    return make_result(__func__, CaseSkip, "multi-procedure pure-objc builder not implemented in this revision");
}

static CaseResult experiment1_strict_native_3step(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }
    ChainCfg cfg = {.doCalls = NO, .enqueueFirst = NO, .proc = 0, .freeSeq = {2}, .freeSeqLen = 1};
    ChainOutcome o = run_single_chaining(ctx, input, cfg);
    free(input);
    if (!o.prepareOK) {
        return make_result(__func__, CaseXFail, "prepare failed under strict map->prepare path: %s", o.detail);
    }
    if (o.readyOK && o.enqueueOK) {
        return make_result(__func__, CasePass, "native 3-step succeeded: %s", o.detail);
    }
    return make_result(__func__, CaseXFail, "prepare passed but execution still blocked (likely driver invariant): %s", o.detail);
}

static CaseResult experiment2_hybrid_program_path(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef statsSurf = make_surface(1024);
    if (inSurf == NULL || outSurf == NULL || statsSurf == NULL) {
        free(input);
        if (statsSurf != NULL) {
            CFRelease(statsSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        return make_result(__func__, CaseFail, "surface creation failed");
    }

    char detail[320] = {0};
    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, sizeof(detail))) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "%s", detail);
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id req = make_simple_request(inObj, outObj, 0, 0, 0);
    if (req == nil || !request_validate(req)) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "request creation/validate failed");
    }

    NSError *err = nil;
    if (!call_map(ctx.client, ctx.model, req, &err)) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "map failed: %s", err_text(err));
    }

    id inBuf = make_buffer(inObj, 0, 0);
    id outBuf = make_buffer(outObj, 0, 0);
    id outSet = make_output_set(statsSurf, @[outBuf]);
    id sharedReady = make_shared_event();
    id sharedFree = make_shared_event();
    id evReady = make_signal_event(1, 0, 5, sharedReady, 1);
    id evFree = make_signal_event(2, 0, 4, sharedFree, 1);
    id chainReq = make_chaining_request(@[inBuf], @[outSet], nil, nil, 0, @[evReady, evFree], 1, 0, 0);
    if (chainReq == nil || !request_validate(chainReq)) {
        call_unmap(ctx.client, ctx.model, req);
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "chaining request creation/validate failed");
    }

    id options = make_shared_event_options();
    char prepDetail[256] = {0};
    if (!first_working_prepare(ctx, chainReq, options, NO, prepDetail, sizeof(prepDetail))) {
        call_unmap(ctx.client, ctx.model, req);
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "prepare failed before program-path test: %s", prepDetail);
    }

    id enqueueObj = make_enqueue(0, 0, 1, NO, NO);
    id readyObj = make_ready(0, @[@0], @[@2], 0);
    if (enqueueObj == nil || readyObj == nil) {
        call_unmap(ctx.client, ctx.model, req);
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "enqueue/ready object creation failed");
    }

    err = nil;
    BOOL pOut = call_program_process_output(ctx.model, enqueueObj, options, &err);
    const char *pOutErr = err_text(err);
    err = nil;
    BOOL pIn = call_program_process_input(ctx.model, readyObj, options, &err);
    const char *pInErr = err_text(err);

    call_unmap(ctx.client, ctx.model, req);
    free(input);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(inSurf);

    if (pOut && pIn) {
        return make_result(__func__, CasePass, "program-path output/input succeeded");
    }
    return make_result(__func__, CaseXFail, "program-path blocked: processOutput=%d err=%s processInput=%d err=%s", pOut, pOutErr, pIn, pInErr);
}

static CaseResult experiment3_direct_evaluation_hijack(ANEContext *ctx) {
    float *input = make_input(ctx.inCount);
    float *out = (float *)calloc((size_t)ctx.outCount, sizeof(float));
    if (input == NULL || out == NULL) {
        free(out);
        free(input);
        return make_result(__func__, CaseFail, "allocation failed");
    }

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    if (inSurf == NULL || outSurf == NULL) {
        free(out);
        free(input);
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        return make_result(__func__, CaseFail, "surface creation failed");
    }

    char detail[256] = {0};
    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, sizeof(detail))) {
        free(out);
        free(input);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "%s", detail);
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id shared = make_shared_event();
    id evReady = make_signal_event(1, 0, 5, shared, 1);
    id evFree = make_signal_event(2, 0, 4, shared, 1);
    id wrapper = make_shared_events_wrapper(@[evReady, evFree]);
    id req = make_full_request(inObj, outObj, 0, 0, 0, wrapper, 1);
    if (req == nil || !request_validate(req)) {
        free(out);
        free(input);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "full request creation/validate failed");
    }

    NSError *err = nil;
    if (!call_map(ctx.client, ctx.model, req, &err)) {
        free(out);
        free(input);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "map failed before direct hijack: %s", err_text(err));
    }

    id options = make_shared_event_options();
    err = nil;
    BOOL ok = call_eval_opt(ctx.client, ctx.model, options, req, ctx.qos, YES, &err);
    const char *evalErr = err_text(err);
    if (ok) {
        (void)read_surface_f32(outSurf, out, ctx.outCount, detail, sizeof(detail));
    }

    call_unmap(ctx.client, ctx.model, req);
    free(out);
    free(input);
    CFRelease(outSurf);
    CFRelease(inSurf);

    if (ok) {
        return make_result(__func__, CasePass, "direct evaluate hijack succeeded with shared-event options");
    }
    return make_result(__func__, CaseXFail, "direct evaluate hijack blocked: %s", evalErr);
}

static CaseResult experiment4_selector8_mutagenesis(ANEContext *ctx) {
    if (!env_enabled("ANE_CHAIN_ENABLE_RAW_VTABLE")) {
        return make_result(__func__, CaseSkip, "set ANE_CHAIN_ENABLE_RAW_VTABLE=1 to run selector-8 mutagenesis (crash risk)");
    }

    float *input = make_input(ctx.inCount);
    if (input == NULL) {
        return make_result(__func__, CaseFail, "allocation failed");
    }

    IOSurfaceRef inSurf = make_surface((size_t)ctx.inCount * sizeof(float));
    IOSurfaceRef outSurf = make_surface((size_t)ctx.outCount * sizeof(float));
    IOSurfaceRef statsSurf = make_surface(1024);
    if (inSurf == NULL || outSurf == NULL || statsSurf == NULL) {
        free(input);
        if (statsSurf != NULL) {
            CFRelease(statsSurf);
        }
        if (outSurf != NULL) {
            CFRelease(outSurf);
        }
        if (inSurf != NULL) {
            CFRelease(inSurf);
        }
        return make_result(__func__, CaseFail, "surface creation failed");
    }

    char detail[320] = {0};
    if (!write_surface_f32(inSurf, input, ctx.inCount, detail, sizeof(detail))) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "%s", detail);
    }

    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIOSurfaceObject, @selector(objectWithIOSurface:), outSurf);
    id req = make_simple_request(inObj, outObj, 0, 0, 0);
    if (req == nil || !request_validate(req)) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "request creation/validate failed");
    }
    NSError *err = nil;
    if (!call_map(ctx.client, ctx.model, req, &err)) {
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "map failed: %s", err_text(err));
    }

    id inBuf = make_buffer(inObj, 0, 0);
    id outBuf = make_buffer(outObj, 0, 0);
    id outSet = make_output_set(statsSurf, @[outBuf]);
    id sharedReady = make_shared_event();
    id sharedFree = make_shared_event();
    id evReady = make_signal_event(1, 0, 5, sharedReady, 1);
    id evFree = make_signal_event(2, 0, 4, sharedFree, 1);
    id chainReq = make_chaining_request(@[inBuf], @[outSet], nil, nil, 0, @[evReady, evFree], 1, 0, 0);
    id options = make_shared_event_options();
    char prepDetail[256] = {0};
    if (!first_working_prepare(ctx, chainReq, options, NO, prepDetail, sizeof(prepDetail))) {
        call_unmap(ctx.client, ctx.model, req);
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseXFail, "prepare failed before selector-8 test: %s", prepDetail);
    }

    uint64_t programHandle = 0;
    void *device = NULL;
    if (!model_program_handle_and_device(ctx.model, &programHandle, &device, detail, sizeof(detail))) {
        call_unmap(ctx.client, ctx.model, req);
        free(input);
        CFRelease(statsSurf);
        CFRelease(outSurf);
        CFRelease(inSurf);
        return make_result(__func__, CaseFail, "model program/device unavailable: %s", detail);
    }

    uint8_t payloadBuf[0x1000];
    memset(payloadBuf, 0, sizeof(payloadBuf));
    ANESetActiveProcedurePayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.programHandle = programHandle;
    payload.procedureIndex = 0;
    memcpy(payloadBuf, &payload, sizeof(payload));

    int32_t rc = 0;
    BOOL callOK = call_device_method(
        device,
        kANEDeviceMethodOffsetProgramChainingSetActiveProc,
        payloadBuf,
        &rc,
        detail,
        sizeof(detail)
    );

    call_unmap(ctx.client, ctx.model, req);
    free(input);
    CFRelease(statsSurf);
    CFRelease(outSurf);
    CFRelease(inSurf);

    if (!callOK) {
        return make_result(__func__, CaseFail, "selector-8 call failed: %s", detail);
    }
    if (rc == 0) {
        return make_result(__func__, CasePass, "selector-8 returned rc=0 with padded payload");
    }
    return make_result(__func__, CaseXFail, "selector-8 returned rc=%d", rc);
}

typedef CaseResult (*CaseFn)(ANEContext *);

typedef struct {
    const char *name;
    CaseFn fn;
} CaseDef;

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        ANEContext *ctx = [[ANEContext alloc] init];
        char setupDetail[320] = {0};
        if (!setup_context(ctx, setupDetail, sizeof(setupDetail))) {
            fprintf(stderr, "setup failed: %s\n", setupDetail);
            return 2;
        }
        printf("setup: %s\n", setupDetail);

        NSString *subcase = env_string("ANE_CHAIN_SUBCASE", @"");
        if (subcase.length > 0) {
            int rc = run_shared_subcase(ctx, subcase);
            teardown_context(ctx);
            return rc;
        }

        const CaseDef cases[] = {
            {"TestANEClientSingleProcedureBaseline", test_single_procedure_baseline},
            {"TestANEClientManualIOSurfacePipeline", test_manual_iosurface_pipeline},
            {"TestANEClientMultiBufferRotationPipeline", test_multibuffer_rotation_pipeline},
            {"TestANEClientSingleProcedurePathMatrix", test_single_procedure_path_matrix},
            {"TestANEClientHealthGate", test_health_gate},
            {"TestANEClientPureFencedEvalMatrix", test_pure_fenced_eval_matrix},
            {"TestANEClientReleaseReadiness", test_release_readiness},
            {"TestANEClientSeqMapThresholdSweep", test_seq_map_threshold_sweep},
            {"TestANEInMemoryModelChainingSelectorProbe", test_inmemory_model_chaining_selector_probe},
            {"TestANEClientChainingMapSelectorProbe", test_chaining_map_selector_probe},
            {"TestANEClientTwoModelChainingProbe", test_two_model_chaining_probe},
            {"TestANEClientSingleProcedureChainingMatrix", test_single_procedure_chaining_matrix},
            {"TestANEClientSingleProcedureBuffersReadyMatrix", test_single_procedure_buffers_ready_matrix},
            {"TestANEClientChainingSpikeSingleProcedure", test_chaining_spike_single_procedure},
            {"TestANEClientChainingSpikeMultiProcedureModel", test_chaining_spike_multi_procedure},
            {"Experiment1StrictNative3Step", experiment1_strict_native_3step},
            {"Experiment2HybridProgramPath", experiment2_hybrid_program_path},
            {"Experiment3DirectEvaluationHijack", experiment3_direct_evaluation_hijack},
            {"Experiment4Selector8Mutagenesis", experiment4_selector8_mutagenesis},
        };

        int pass = 0;
        int fail = 0;
        int xfail = 0;
        int skip = 0;
        BOOL matchedCaseOnly = NO;

        BOOL strict = env_enabled("ANE_CHAIN_STRICT");
        NSString *caseOnly = env_string("ANE_CHAIN_CASE_ONLY", @"");

        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            if (caseOnly.length > 0 && ![caseOnly isEqualToString:[NSString stringWithUTF8String:cases[i].name]]) {
                continue;
            }
            matchedCaseOnly = YES;
            CaseResult r = cases[i].fn(ctx);
            const char *status = "PASS";
            switch (r.status) {
                case CasePass:
                    pass++;
                    status = "PASS";
                    break;
                case CaseFail:
                    fail++;
                    status = "FAIL";
                    break;
                case CaseXFail:
                    xfail++;
                    status = "XFAIL";
                    break;
                case CaseSkip:
                    skip++;
                    status = "SKIP";
                    break;
            }
            printf("[%s] %s: %s\n", status, cases[i].name, r.detail[0] ? r.detail : "-");
        }

        if (caseOnly.length > 0 && !matchedCaseOnly) {
            teardown_context(ctx);
            fprintf(stderr, "unknown case filter: %s\n", caseOnly.UTF8String);
            return 2;
        }

        teardown_context(ctx);

        printf("summary: pass=%d fail=%d xfail=%d skip=%d strict=%d\n", pass, fail, xfail, skip, strict ? 1 : 0);
        if (strict && xfail > 0) {
            fail += xfail;
        }
        return fail == 0 ? 0 : 1;
    }
}
