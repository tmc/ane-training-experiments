// test_espresso_bindings.m - Runtime probe for Espresso multi-buffer IOSurface bindings.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static bool selector_exists(Class cls, const char *name) {
    if (cls == Nil || name == NULL) {
        return false;
    }
    SEL sel = sel_registerName(name);
    return [cls instancesRespondToSelector:sel];
}

static IOSurfaceRef make_probe_surface(size_t bytes) {
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

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        void *espresso = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW | RTLD_GLOBAL);
        if (espresso == NULL) {
            fprintf(stderr, "FAIL: dlopen Espresso failed: %s\n", dlerror());
            return 1;
        }

        Class surfCls = NSClassFromString(@"EspressoANEIOSurface");
        Class execCls = NSClassFromString(@"EspressoDataFrameExecutor");
        if (surfCls == Nil) {
            fprintf(stderr, "FAIL: EspressoANEIOSurface class not found\n");
            return 1;
        }

        const char *surfaceSelectors[] = {
            "initWithIOSurfaceProperties:andPixelFormats:",
            "resizeForMultipleAsyncBuffers:",
            "nFrames",
            "ioSurfaceForMultiBufferFrame:",
            "ane_io_surfaceForMultiBufferFrame:",
            "setExternalStorage:ioSurface:",
            "restoreInternalStorageForAllMultiBufferFrames",
            "metalBufferWithDevice:multiBufferFrame:",
            "cleanup",
        };

        int present = 0;
        for (size_t i = 0; i < sizeof(surfaceSelectors) / sizeof(surfaceSelectors[0]); i++) {
            bool ok = selector_exists(surfCls, surfaceSelectors[i]);
            printf("selector EspressoANEIOSurface::%s present=%d\n", surfaceSelectors[i], ok ? 1 : 0);
            if (ok) {
                present++;
            }
        }

        if (execCls != Nil) {
            const char *executorSelectors[] = {
                "bindInputsFromFrame:toNetwork:",
                "bindOutputsFromFrame:toNetwork:",
                "bindOutputsFromFrame:toNetwork:executionStatus:",
                "freeTemporaryResources",
                "useCVPixelBuffersForOutputs:",
            };
            for (size_t i = 0; i < sizeof(executorSelectors) / sizeof(executorSelectors[0]); i++) {
                bool ok = selector_exists(execCls, executorSelectors[i]);
                printf("selector EspressoDataFrameExecutor::%s present=%d\n", executorSelectors[i], ok ? 1 : 0);
            }
        } else {
            printf("note: EspressoDataFrameExecutor class not found\n");
        }

        SEL initSel = sel_registerName("initWithIOSurfaceProperties:andPixelFormats:");
        if (![surfCls instancesRespondToSelector:initSel]) {
            fprintf(stderr, "FAIL: initWithIOSurfaceProperties:andPixelFormats: missing\n");
            return 1;
        }

        NSDictionary *props = @{
            @"IOSurfaceWidth": @4096,
            @"IOSurfaceHeight": @1,
            @"IOSurfaceBytesPerElement": @1,
            @"IOSurfaceBytesPerRow": @4096,
            @"IOSurfaceAllocSize": @4096,
            @"IOSurfacePixelFormat": @0,
        };
        NSSet *formats = [NSSet setWithObject:@0u];

        id obj = ((id(*)(Class, SEL))objc_msgSend)(surfCls, @selector(alloc));
        obj = ((id(*)(id, SEL, id, id))objc_msgSend)(obj, initSel, props, formats);
        if (obj == nil) {
            fprintf(stderr, "FAIL: EspressoANEIOSurface init returned nil\n");
            return 1;
        }

        if ([obj respondsToSelector:sel_registerName("resizeForMultipleAsyncBuffers:")]) {
            ((void(*)(id, SEL, unsigned long long))objc_msgSend)(obj, sel_registerName("resizeForMultipleAsyncBuffers:"), 3ULL);
        }

        unsigned long long frames = 0;
        if ([obj respondsToSelector:sel_registerName("nFrames")]) {
            frames = ((unsigned long long(*)(id, SEL))objc_msgSend)(obj, sel_registerName("nFrames"));
        }
        printf("runtime EspressoANEIOSurface nFrames=%llu\n", frames);

        SEL frameSel = sel_registerName("ioSurfaceForMultiBufferFrame:");
        int surfaces = 0;
        for (unsigned long long i = 0; i < 3; i++) {
            IOSurfaceRef surf = NULL;
            if ([obj respondsToSelector:frameSel]) {
                surf = ((IOSurfaceRef(*)(id, SEL, unsigned long long))objc_msgSend)(obj, frameSel, i);
            }
            if (surf != NULL) {
                uint32_t sid = IOSurfaceGetID(surf);
                size_t alloc = IOSurfaceGetAllocSize(surf);
                printf("frame=%llu surface_id=%u alloc=%zu\n", i, sid, alloc);
                surfaces++;
            } else {
                printf("frame=%llu surface=nil\n", i);
            }
        }

        if ([obj respondsToSelector:sel_registerName("setExternalStorage:ioSurface:")]) {
            IOSurfaceRef external = make_probe_surface(4096);
            if (external != NULL) {
                ((void(*)(id, SEL, unsigned long long, IOSurfaceRef))objc_msgSend)(obj, sel_registerName("setExternalStorage:ioSurface:"), 1ULL, external);
                printf("setExternalStorage:ioSurface: invoked for slot=1\n");
                CFRelease(external);
            }
        }

        if ([obj respondsToSelector:sel_registerName("restoreInternalStorageForAllMultiBufferFrames")]) {
            ((void(*)(id, SEL))objc_msgSend)(obj, sel_registerName("restoreInternalStorageForAllMultiBufferFrames"));
            printf("restoreInternalStorageForAllMultiBufferFrames invoked\n");
        }

        if ([obj respondsToSelector:sel_registerName("cleanup")]) {
            ((void(*)(id, SEL))objc_msgSend)(obj, sel_registerName("cleanup"));
            printf("cleanup invoked\n");
        }

        if (present < 6) {
            fprintf(stderr, "FAIL: missing key EspressoANEIOSurface selectors present=%d\n", present);
            return 1;
        }
        if (surfaces == 0) {
            fprintf(stderr, "FAIL: ioSurfaceForMultiBufferFrame returned no surfaces\n");
            return 1;
        }

        printf("PASS: Espresso bindings probe complete selectors=%d frames=%llu nonNilSurfaces=%d\n", present, frames, surfaces);
        return 0;
    }
}
