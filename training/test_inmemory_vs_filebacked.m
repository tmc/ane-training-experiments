// test_inmemory_vs_filebacked.m
// Compare _ANEInMemoryModel vs file-backed _ANEModel/_ANEClient execution paths.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned int kQoS = 21;

static Class CInMemDesc;
static Class CInMem;
static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIO;
static Class CNSURL;

static const char *err_desc(NSError *err) {
    return err ? [[err description] UTF8String] : "nil";
}

static double now_ms(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000.0;
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

static BOOL setup_classes(void) {
    void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!ane) {
        fprintf(stderr, "FAIL: dlopen AppleNeuralEngine\n");
        return NO;
    }

    CInMemDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    CInMem = NSClassFromString(@"_ANEInMemoryModel");
    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");
    CNSURL = NSClassFromString(@"NSURL");

    return CInMemDesc && CInMem && CClient && CModel && CReq && CAIO && CNSURL;
}

static id make_basic_request(IOSurfaceRef inSurf, IOSurfaceRef outSurf) {
    id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), outSurf);
    if (!inObj || !outObj) {
        return nil;
    }
    SEL reqSel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    if (![CReq respondsToSelector:reqSel]) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        CReq, reqSel, @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0);
}

static BOOL stage_inmem_model_files(id mdl, NSData *milData, NSData *weightsData, NSError **err) {
    if (!mdl || !milData || !weightsData) {
        return NO;
    }
    NSString *hx = nil;
    if ([mdl respondsToSelector:@selector(hexStringIdentifier)]) {
        hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    }
    if (!hx) {
        hx = [[NSUUID UUID] UUIDString];
    }

    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *wd = [td stringByAppendingPathComponent:@"weights"];
    if (![fm createDirectoryAtPath:wd withIntermediateDirectories:YES attributes:nil error:err]) {
        return NO;
    }
    if (![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:err]) {
        return NO;
    }
    if (![weightsData writeToFile:[wd stringByAppendingPathComponent:@"weight.bin"] options:NSDataWritingAtomic error:err]) {
        return NO;
    }
    return YES;
}

static id create_inmem_model_from_mlmodelc(NSString *modelDir, NSError **err) {
    NSString *milPath = [modelDir stringByAppendingPathComponent:@"model.mil"];
    NSString *wPath = [modelDir stringByAppendingPathComponent:@"weights/weight.bin"];
    NSData *milData = [NSData dataWithContentsOfFile:milPath options:0 error:err];
    if (!milData) {
        return nil;
    }
    NSData *wData = [NSData dataWithContentsOfFile:wPath options:0 error:err];
    if (!wData) {
        return nil;
    }

    NSDictionary *weights = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wData}};
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        CInMemDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
    if (!desc) {
        return nil;
    }

    id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(CInMem, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) {
        return nil;
    }

    NSError *stageErr = nil;
    if (!stage_inmem_model_files(mdl, milData, wData, &stageErr)) {
        if (err) *err = stageErr;
        return nil;
    }

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), kQoS, @{}, &e);
    if (!ok) {
        if (err) *err = e;
        return nil;
    }

    e = nil;
    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), kQoS, @{}, &e);
    if (!ok) {
        usleep(100000);
        e = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), kQoS, @{}, &e);
    }
    if (!ok) {
        if (err) *err = e;
        return nil;
    }
    return mdl;
}

static id get_inmem_model_handle(id inMemModel) {
    if (!inMemModel) {
        return nil;
    }
    SEL modelSel = NSSelectorFromString(@"model");
    if ([inMemModel respondsToSelector:modelSel]) {
        @try {
            return ((id(*)(id, SEL))objc_msgSend)(inMemModel, modelSel);
        } @catch (__unused NSException *e) {}
    }
    @try {
        return [inMemModel valueForKey:@"model"];
    } @catch (__unused NSException *e) {
        return nil;
    }
}

static BOOL eval_inmem(id inMemModel, id req, double *latMs, NSError **err) {
    double t0 = now_ms();
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        inMemModel, @selector(evaluateWithQoS:options:request:error:), kQoS, @{}, req, err);
    double t1 = now_ms();
    if (latMs) {
        *latMs = t1 - t0;
    }
    return ok;
}

static BOOL eval_filebacked(id client, id model, id req, double *latMs, NSError **err) {
    SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    if (![client respondsToSelector:evalSel]) {
        evalSel = @selector(evaluateWithModel:options:request:qos:error:);
    }
    double t0 = now_ms();
    BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        client, evalSel, model, @{}, req, kQoS, err);
    double t1 = now_ms();
    if (latMs) {
        *latMs = t1 - t0;
    }
    return ok;
}

static id safe_kvc(id obj, NSString *key) {
    @try {
        return [obj valueForKey:key];
    } @catch (__unused NSException *e) {
        return nil;
    }
}

static void print_backend_probe(id inMemModel) {
    id espresso = nil;
    id program = nil;
    id isLoaded = nil;
    id queueDepth = nil;

    @try { espresso = [inMemModel valueForKey:@"_espressoModel"]; } @catch (__unused NSException *e) {}
    @try { program = [inMemModel valueForKey:@"_program"]; } @catch (__unused NSException *e) {}
    if (program) {
        @try { isLoaded = [program valueForKey:@"_isLoaded"]; } @catch (__unused NSException *e) {}
        @try { queueDepth = [program valueForKey:@"queueDepth"]; } @catch (__unused NSException *e) {}
    }

    printf("backend_probe espressoModel=%s program=%s program._isLoaded=%s program.queueDepth=%s\n",
           espresso ? "non-nil" : "nil",
           program ? "non-nil" : "nil",
           isLoaded ? [[isLoaded description] UTF8String] : "nil",
           queueDepth ? [[queueDepth description] UTF8String] : "nil");
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (!setup_classes()) {
            printf("FAIL: class setup\n");
            return 1;
        }

        NSString *modelDir = @"/Volumes/tmc/go/src/github.com/maderix/ANE/testdata/ffn/draft125m_ffn.mlmodelc";
        const char *rawPath = getenv("ANE_MODEL_PATH");
        if (rawPath && rawPath[0]) {
            modelDir = [NSString stringWithUTF8String:rawPath];
        }

        const int count = 768;
        const size_t bytes = (size_t)count * sizeof(float);

        float inVals[count];
        for (int i = 0; i < count; i++) {
            inVals[i] = (float)(i + 1) * 0.001f;
        }

        // --- Experiment 1: map instrumentation on in-memory model ---
        NSError *err = nil;
        id inMemModel = create_inmem_model_from_mlmodelc(modelDir, &err);
        if (!inMemModel) {
            printf("FAIL: in-memory model create/compile/load: %s\n", err_desc(err));
            return 1;
        }

        IOSurfaceRef inSurf = make_surface(bytes);
        IOSurfaceRef outSurf = make_surface(bytes);
        write_surface_f32(inSurf, inVals, count);
        id reqInMem = make_basic_request(inSurf, outSurf);
        if (!reqInMem) {
            printf("FAIL: create request for in-memory path\n");
            return 1;
        }

        id sharedConnection = nil;
        if ([inMemModel respondsToSelector:@selector(sharedConnection)]) {
            @try {
                sharedConnection = ((id(*)(id, SEL))objc_msgSend)(inMemModel, @selector(sharedConnection));
            } @catch (__unused NSException *e) {}
        }

        id inMemModelHandle = get_inmem_model_handle(inMemModel);
        BOOL mapClientOK = NO;
        BOOL mapLocalOK = NO;
        NSError *mapClientErr = nil;
        NSError *mapLocalErr = nil;
        const char *usedMap = "none";

        if (sharedConnection && inMemModelHandle && [sharedConnection respondsToSelector:@selector(mapIOSurfacesWithModel:request:cacheInference:error:)]) {
            mapClientOK = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
                sharedConnection,
                @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
                inMemModelHandle,
                reqInMem,
                YES,
                &mapClientErr);
        }

        SEL localMapSel = NSSelectorFromString(@"mapIOSurfacesWithRequest:cacheInference:error:");
        if ([inMemModel respondsToSelector:localMapSel]) {
            mapLocalOK = ((BOOL(*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                inMemModel, localMapSel, reqInMem, YES, &mapLocalErr);
        }

        double inMemEvalMs = -1.0;
        NSError *inMemEvalErr = nil;
        BOOL inMemEvalOK = NO;
        if (mapClientOK) {
            usedMap = "client map";
            inMemEvalOK = eval_inmem(inMemModel, reqInMem, &inMemEvalMs, &inMemEvalErr);
        } else if (mapLocalOK) {
            usedMap = "local map";
            inMemEvalOK = eval_inmem(inMemModel, reqInMem, &inMemEvalMs, &inMemEvalErr);
        }

        printf("exp1 sharedConnection=%s map_client=%d err=%s map_local=%d err=%s used_map=%s eval_ok=%d eval_ms=%.3f eval_err=%s\n",
               sharedConnection ? "non-nil" : "nil",
               mapClientOK ? 1 : 0,
               err_desc(mapClientErr),
               mapLocalOK ? 1 : 0,
               err_desc(mapLocalErr),
               usedMap,
               inMemEvalOK ? 1 : 0,
               inMemEvalMs,
               err_desc(inMemEvalErr));

        print_backend_probe(inMemModel);
        id espressoObj = safe_kvc(inMemModel, @"_espressoModel");
        id programObj = safe_kvc(inMemModel, @"_program");

        // --- Experiment 2: explicit compare with file-backed _ANEModel + _ANEClient ---
        id client = ((id(*)(Class, SEL))objc_msgSend)(CClient, @selector(sharedConnection));
        if (!client) {
            printf("FAIL: _ANEClient sharedConnection nil\n");
            return 1;
        }

        id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), modelDir);
        id fbModel = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, @"main");
        if (!fbModel) {
            printf("FAIL: file-backed modelAtURL:key: main\n");
            return 1;
        }

        err = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(compileModel:options:qos:error:), fbModel, @{}, kQoS, &err);
        if (!ok) {
            printf("FAIL: file-backed compileModel: %s\n", err_desc(err));
            return 1;
        }

        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(loadModel:options:qos:error:), fbModel, @{}, kQoS, &err);
        if (!ok) {
            usleep(100000);
            err = nil;
            ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
                client, @selector(loadModel:options:qos:error:), fbModel, @{}, kQoS, &err);
        }
        if (!ok) {
            printf("FAIL: file-backed loadModel: %s\n", err_desc(err));
            return 1;
        }

        IOSurfaceRef inSurf2 = make_surface(bytes);
        IOSurfaceRef outSurf2 = make_surface(bytes);
        write_surface_f32(inSurf2, inVals, count);
        id reqFile = make_basic_request(inSurf2, outSurf2);
        if (!reqFile) {
            printf("FAIL: create request file-backed\n");
            return 1;
        }

        NSError *fbMapErr = nil;
        BOOL fbMapOK = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), fbModel, reqFile, YES, &fbMapErr);

        double fbEvalMs = -1.0;
        NSError *fbEvalErr = nil;
        BOOL fbEvalOK = NO;
        if (fbMapOK) {
            fbEvalOK = eval_filebacked(client, fbModel, reqFile, &fbEvalMs, &fbEvalErr);
        }

        printf("exp2 filebacked map_ok=%d map_err=%s eval_ok=%d eval_ms=%.3f eval_err=%s\n",
               fbMapOK ? 1 : 0,
               err_desc(fbMapErr),
               fbEvalOK ? 1 : 0,
               fbEvalMs,
               err_desc(fbEvalErr));

        // Experiment 3: backend probe output already printed above.

        printf("\n| Path | Map method | Map result | Eval latency | Backend |\n");
        printf("|---|---|---|---|---|\n");
        printf("| _ANEInMemoryModel | client map | %s | %s | espresso=%s program=%s |\n",
               mapClientOK ? "OK" : err_desc(mapClientErr),
               inMemEvalOK ? [[NSString stringWithFormat:@"%.3f ms", inMemEvalMs] UTF8String] : "n/a",
               (espressoObj ? "non-nil" : "nil"),
               (programObj ? "non-nil" : "nil"));
        printf("| _ANEInMemoryModel | local map | %s | %s | same model probe |\n",
               mapLocalOK ? "OK" : err_desc(mapLocalErr),
               (mapLocalOK && inMemEvalOK && strcmp(usedMap, "local map") == 0)
                   ? [[NSString stringWithFormat:@"%.3f ms", inMemEvalMs] UTF8String]
                   : "n/a");
        printf("| _ANEModel + _ANEClient | client map | %s | %s | ANE client path |\n",
               fbMapOK ? "OK" : err_desc(fbMapErr),
               fbEvalOK ? [[NSString stringWithFormat:@"%.3f ms", fbEvalMs] UTF8String] : "n/a");

        return 0;
    }
}
