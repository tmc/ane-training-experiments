// test_ane_metal_zero_copy.m
// End-to-end zero-copy probe: ANE output IOSurface -> Metal compute -> CPU verify.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <dlfcn.h>
#import <objc/message.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const unsigned int kQoS = 21;

static Class CClient;
static Class CModel;
static Class CReq;
static Class CAIO;
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

static BOOL setup_ane_classes(void) {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "failed to dlopen AppleNeuralEngine: %s\n", dlerror());
        return NO;
    }
    CClient = NSClassFromString(@"_ANEClient");
    CModel = NSClassFromString(@"_ANEModel");
    CReq = NSClassFromString(@"_ANERequest");
    CAIO = NSClassFromString(@"_ANEIOSurfaceObject");
    CNSURL = NSClassFromString(@"NSURL");
    if (!CClient || !CModel || !CReq || !CAIO || !CNSURL) {
        fprintf(stderr, "failed to resolve ANE classes\n");
        return NO;
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (!setup_ane_classes()) {
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
            fprintf(stderr, "_ANEClient sharedConnection nil\n");
            return 2;
        }

        id url = ((id(*)(Class, SEL, id))objc_msgSend)(CNSURL, @selector(fileURLWithPath:), modelPath);
        id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(CModel, @selector(modelAtURL:key:), url, modelKey);
        if (!model) {
            fprintf(stderr, "modelAtURL:key: nil path=%s\n", modelPath.UTF8String);
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
        IOSurfaceRef aneOutSurf = make_surface(bytes);
        if (!inSurf || !aneOutSurf) {
            fprintf(stderr, "surface allocation failed\n");
            return 2;
        }

        float *inVals = (float *)calloc((size_t)count, sizeof(float));
        float *aneVals = (float *)calloc((size_t)count, sizeof(float));
        if (!inVals || !aneVals) {
            fprintf(stderr, "alloc failed\n");
            return 2;
        }
        for (int i = 0; i < count; i++) {
            inVals[i] = (float)(i + 1);
        }
        write_surface_f32(inSurf, inVals, count);

        id inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), inSurf);
        id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(CAIO, @selector(objectWithIOSurface:), aneOutSurf);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            CReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0
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
            fprintf(stderr, "map failed: %s\n", err_desc(err));
            return 2;
        }

        err = nil;
        ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client,
            @selector(evaluateWithModel:options:request:qos:error:),
            model,
            @{},
            req,
            kQoS,
            &err
        );
        if (!ok) {
            fprintf(stderr, "evaluate failed: %s\n", err_desc(err));
            return 2;
        }

        read_surface_f32(aneOutSurf, aneVals, count);
        printf("ane_output[0..2]=[%.6f, %.6f, %.6f]\n", aneVals[0], aneVals[1], aneVals[2]);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }

        IOSurfaceLock(aneOutSurf, kIOSurfaceLockReadOnly, NULL);
        void *base = IOSurfaceGetBaseAddress(aneOutSurf);
        if (!base) {
            IOSurfaceUnlock(aneOutSurf, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "IOSurface base address nil\n");
            return 2;
        }

        id<MTLBuffer> inBuf = [dev newBufferWithBytesNoCopy:base length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> outBuf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!inBuf || !outBuf) {
            IOSurfaceUnlock(aneOutSurf, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "Metal buffer creation failed\n");
            return 2;
        }

        NSString *src = @"using namespace metal;\n"
                        "kernel void scale_half(device const float *inv [[buffer(0)]], device float *outv [[buffer(1)]], uint id [[thread_position_in_grid]]) {\n"
                        "  outv[id] = inv[id] * 0.5f;\n"
                        "}\n";
        NSError *merr = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&merr];
        if (!lib) {
            IOSurfaceUnlock(aneOutSurf, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "newLibraryWithSource failed: %s\n", err_desc(merr));
            return 2;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"scale_half"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&merr];
        if (!pso) {
            IOSurfaceUnlock(aneOutSurf, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "newComputePipelineState failed: %s\n", err_desc(merr));
            return 2;
        }

        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:inBuf offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];
        MTLSize grid = MTLSizeMake((NSUInteger)count, 1, 1);
        NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
        if (w > (NSUInteger)count) w = (NSUInteger)count;
        MTLSize tg = MTLSizeMake(w, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        IOSurfaceUnlock(aneOutSurf, kIOSurfaceLockReadOnly, NULL);

        float *metalVals = (float *)outBuf.contents;
        if (!metalVals) {
            fprintf(stderr, "outBuf contents nil\n");
            return 2;
        }

        double maxDiff = 0.0;
        for (int i = 0; i < count; i++) {
            float want = aneVals[i] * 0.5f;
            double d = fabs((double)metalVals[i] - (double)want);
            if (d > maxDiff) maxDiff = d;
        }
        printf("metal_output[0..2]=[%.6f, %.6f, %.6f] maxDiff=%g\n", metalVals[0], metalVals[1], metalVals[2], maxDiff);

        ((void(*)(id, SEL, id, id))objc_msgSend)(client, @selector(unmapIOSurfacesWithModel:request:), model, req);
        NSError *uerr = nil;
        ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &uerr);

        CFRelease(aneOutSurf);
        CFRelease(inSurf);
        free(aneVals);
        free(inVals);

        if (maxDiff > 1e-4) {
            printf("FAIL: Metal compute output mismatch\n");
            return 1;
        }

        printf("PASS: zero-copy ANE->Metal->CPU path verified\n");
        return 0;
    }
}
