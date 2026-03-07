// ane_bridge.m — Objective-C implementation of ANE bridge for Python ctypes
// Wraps _ANEInMemoryModel private APIs into C-callable functions

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include <unistd.h>
#include "ane_bridge.h"

// --- Private class references ---
static Class g_ANEDesc = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq = nil;
static Class g_ANEIO = nil;
static Class g_ANEClient = nil;
static Class g_ANEModel = nil;
static Class g_ANEWaitEvent = nil;
static Class g_ANESignalEvent = nil;
static Class g_ANESharedEvents = nil;
static Class g_NSURL = nil;
static void *g_ane_framework_handle = NULL;
static bool g_initialized = false;
static int g_compile_count = 0;
static const void *g_completion_handler_assoc_key = &g_completion_handler_assoc_key;

// --- Kernel handle struct ---
struct ANEKernelHandle {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
};

struct ANEClientHandle {
    id client;      // _ANEClient.sharedConnection
    id model;       // _ANEModel
    IOSurfaceRef inSurf;
    IOSurfaceRef outSurf;
    id inObj;       // _ANEIOSurfaceObject
    id outObj;      // _ANEIOSurfaceObject
    size_t inBytes;
    size_t outBytes;
};

// --- Public API ---

int ane_bridge_init(void) {
    if (g_initialized) return 0;

    g_ane_framework_handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!g_ane_framework_handle) {
        fprintf(stderr, "ane_bridge: Failed to load AppleNeuralEngine.framework\n");
        return -1;
    }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ANEClient = NSClassFromString(@"_ANEClient");
    g_ANEModel = NSClassFromString(@"_ANEModel");
    g_ANEWaitEvent = NSClassFromString(@"_ANESharedWaitEvent");
    g_ANESignalEvent = NSClassFromString(@"_ANESharedSignalEvent");
    g_ANESharedEvents = NSClassFromString(@"_ANESharedEvents");
    g_NSURL = NSClassFromString(@"NSURL");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_bridge: Failed to resolve ANE private classes\n");
        return -1;
    }

    g_initialized = true;
    g_compile_count = 0;
    return 0;
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    @autoreleasepool {
        if (!g_initialized) {
            fprintf(stderr, "ane_bridge: Not initialized\n");
            return NULL;
        }

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSError *e = nil;

        // Build weight dictionary
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict.count > 0 ? wdict : @{}, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: modelWithMILText failed\n");
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: inMemoryModelWithDescriptor failed\n");
            return NULL;
        }

        // Pre-populate temp dir
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            // Extract filename from path like "@model_path/weights/wq.bin" -> "weights/wq.bin"
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) {
                relPath = [name substringFromIndex:12];
            }
            NSString *fullPath = [td stringByAppendingPathComponent:relPath];
            NSString *dir = [fullPath stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:YES];
        }

        // Compile
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_bridge: ANE compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Load (with one retry after a brief pause for ANE slot reclamation)
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed (retrying in 100ms): %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            usleep(100000); // 100ms
            e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed after retry: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        g_compile_count++;

        // Create kernel handle
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;
        k->tmpDir = td;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        // Create IOSurfaces
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            k->ioInputs[i] = create_surface(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_sizes[i]);

        // Build request
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return k;
    }
}

ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes) {
    if (weight_data && weight_len > 0) {
        const char *name = "@model_path/weights/weight.bin";
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            &name, &weight_data, &weight_len, 1,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    } else {
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            NULL, NULL, NULL, 0,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    }
}

bool ane_bridge_eval(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel || !kernel->model) return false;
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            kernel->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, kernel->request, &e);
    }
}

void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return;
    IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(kernel->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
}

void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs) return;
    IOSurfaceLock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

void ane_bridge_free(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) return;
        NSError *e = nil;
        if (kernel->model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &e);
        }
        for (int i = 0; i < kernel->nInputs; i++)
            if (kernel->ioInputs[i]) CFRelease(kernel->ioInputs[i]);
        for (int i = 0; i < kernel->nOutputs; i++)
            if (kernel->ioOutputs[i]) CFRelease(kernel->ioOutputs[i]);
        if (kernel->tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:kernel->tmpDir error:nil];
        }
        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->inputBytes);
        free(kernel->outputBytes);
        
        // Explicitly nil Objective-C objects to trigger ARC release before freeing struct
        kernel->model = nil;
        kernel->request = nil;
        kernel->tmpDir = nil;
        
        free(kernel);
    }
}

int ane_bridge_get_compile_count(void) {
    return g_compile_count;
}

void ane_bridge_reset_compile_count(void) {
    g_compile_count = 0;
}

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len) {
    int wsize = rows * cols * 2; // fp16
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE blob header
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    // Convert float32 -> float16
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)src[i];
    }

    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j * rows + i] = (_Float16)src[i * cols + j];

    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
                                            size_t *out_len) {
    int wsize = rows * cols;  // 1 byte per int8 element
    int total = 64 + wsize;   // 64-byte header + data
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE int8 blob header
    buf[0] = 0xEF; buf[1] = 0xBE; buf[2] = 0xAD; buf[3] = 0xDE;
    buf[4] = 0x01;
    buf[10] = 0x08;  // 8-bit element marker

    memcpy(buf + 64, src, wsize);
    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
                                                 float *out_scale, size_t *out_len) {
    // Find global max abs for symmetric quantization
    float max_abs = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;

    // Quantize to int8
    int wsize = rows * cols;
    int8_t *qdata = (int8_t *)malloc(wsize);
    for (int i = 0; i < wsize; i++) {
        float v = src[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qdata[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }

    uint8_t *blob = ane_bridge_build_weight_blob_int8(qdata, rows, cols, out_len);
    free(qdata);
    *out_scale = scale;
    return blob;
}

void ane_bridge_free_blob(void *ptr) {
    free(ptr);
}

static id ane_const_obj(const char *symbol) {
    if (!g_ane_framework_handle || !symbol || !symbol[0]) {
        return nil;
    }
    void *sym = dlsym(g_ane_framework_handle, symbol);
    if (!sym && symbol[0] != '_') {
        char buf[128] = {0};
        snprintf(buf, sizeof(buf), "_%s", symbol);
        sym = dlsym(g_ane_framework_handle, buf);
    }
    return sym ? *((__unsafe_unretained id *)sym) : nil;
}

static void ane_attach_request_completion_handler(id request) {
    if (!request || ![request respondsToSelector:@selector(setCompletionHandler:)]) {
        return;
    }
    id completionHandler = ^(BOOL success, NSError *error) {
        (void)success;
        (void)error;
    };
    ((void(*)(id, SEL, id))objc_msgSend)(request, @selector(setCompletionHandler:), completionHandler);
    objc_setAssociatedObject(request, g_completion_handler_assoc_key, completionHandler, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

ANEClientHandle *ane_bridge_client_open(const char *model_path, const char *model_key,
                                        size_t input_bytes, size_t output_bytes) {
    @autoreleasepool {
        if (ane_bridge_init() != 0) {
            return NULL;
        }
        if (!g_ANEClient || !g_ANEModel || !g_NSURL || !g_ANEReq || !g_ANEIO) {
            fprintf(stderr, "ane_bridge: _ANEClient path unavailable\n");
            return NULL;
        }
        if (!model_path || model_path[0] == '\0') {
            fprintf(stderr, "ane_bridge: model_path is empty\n");
            return NULL;
        }

        id client = ((id(*)(Class, SEL))objc_msgSend)(g_ANEClient, @selector(sharedConnection));
        if (!client) {
            fprintf(stderr, "ane_bridge: sharedConnection returned nil\n");
            return NULL;
        }
        NSString *path = [NSString stringWithUTF8String:model_path];
        NSString *key = (model_key && model_key[0]) ? [NSString stringWithUTF8String:model_key] : @"s";
        id url = ((id(*)(Class, SEL, id))objc_msgSend)(g_NSURL, @selector(fileURLWithPath:), path);
        id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(g_ANEModel, @selector(modelAtURL:key:), url, key);
        if (!model) {
            fprintf(stderr, "ane_bridge: modelAtURL:key: failed\n");
            return NULL;
        }

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(compileModel:options:qos:error:), model, @{}, 21, &e);
        if (!ok) {
            fprintf(stderr, "ane_bridge: compileModel failed: %s\n", e ? [[e description] UTF8String] : "unknown");
            return NULL;
        }
        e = nil;
        ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(loadModel:options:qos:error:), model, @{}, 21, &e);
        if (!ok) {
            usleep(100000);
            e = nil;
            ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
                client, @selector(loadModel:options:qos:error:), model, @{}, 21, &e);
        }
        if (!ok) {
            fprintf(stderr, "ane_bridge: loadModel failed: %s\n", e ? [[e description] UTF8String] : "unknown");
            return NULL;
        }

        ANEClientHandle *h = (ANEClientHandle *)calloc(1, sizeof(ANEClientHandle));
        h->client = client;
        h->model = model;
        h->inBytes = input_bytes;
        h->outBytes = output_bytes;
        h->inSurf = create_surface(input_bytes);
        h->outSurf = create_surface(output_bytes);
        if (!h->inSurf || !h->outSurf) {
            ane_bridge_client_close(h);
            return NULL;
        }
        h->inObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), h->inSurf);
        h->outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), h->outSurf);
        if (!h->inObj || !h->outObj) {
            ane_bridge_client_close(h);
            return NULL;
        }
        return h;
    }
}

void ane_bridge_client_close(ANEClientHandle *h) {
    @autoreleasepool {
        if (!h) {
            return;
        }
        NSError *e = nil;
        if (h->client && h->model) {
            ((BOOL(*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
                h->client, @selector(unloadModel:options:qos:error:), h->model, @{}, 21, &e);
        }
        if (h->inSurf) {
            CFRelease(h->inSurf);
        }
        if (h->outSurf) {
            CFRelease(h->outSurf);
        }
        h->client = nil;
        h->model = nil;
        h->inObj = nil;
        h->outObj = nil;
        free(h);
    }
}

IOSurfaceRef ane_bridge_client_input_surface(ANEClientHandle *h) {
    return h ? h->inSurf : NULL;
}

IOSurfaceRef ane_bridge_client_output_surface(ANEClientHandle *h) {
    return h ? h->outSurf : NULL;
}

void ane_bridge_client_write_input(ANEClientHandle *h, const float *data, int count) {
    if (!h || !h->inSurf || !data || count <= 0) {
        return;
    }
    size_t want = (size_t)count * sizeof(float);
    size_t n = want < h->inBytes ? want : h->inBytes;
    IOSurfaceLock(h->inSurf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(h->inSurf), data, n);
    IOSurfaceUnlock(h->inSurf, 0, NULL);
}

void ane_bridge_client_read_output(ANEClientHandle *h, float *data, int count) {
    if (!h || !h->outSurf || !data || count <= 0) {
        return;
    }
    size_t want = (size_t)count * sizeof(float);
    size_t n = want < h->outBytes ? want : h->outBytes;
    IOSurfaceLock(h->outSurf, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(h->outSurf), n);
    IOSurfaceUnlock(h->outSurf, kIOSurfaceLockReadOnly, NULL);
}

bool ane_bridge_client_eval(ANEClientHandle *h) {
    @autoreleasepool {
        if (!h || !h->client || !h->model || !h->inObj || !h->outObj || !g_ANEReq) {
            return false;
        }
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[h->inObj], @[@0], @[h->outObj], @[@0], nil, nil, @0);
        if (!req) {
            return false;
        }
        NSError *e = nil;
        BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            h->client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), h->model, req, YES, &e);
        if (!mapped) {
            return false;
        }
        SEL evalSel = @selector(evaluateWithModel:options:request:qos:error:);
        if (![h->client respondsToSelector:evalSel]) {
            return false;
        }
        e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            h->client, evalSel, h->model, @{}, req, 21, &e);
        ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
        return ok ? true : false;
    }
}

int ane_bridge_eval_loopback(ane_bridge_client_t client,
                             const float *initial_input, uint32_t input_count,
                             float *output_logits, uint32_t output_count,
                             int num_tokens,
                             ane_bridge_token_callback_t token_callback,
                             void *callback_ctx) {
    @autoreleasepool {
        ANEClientHandle *h = client;
        if (!h || !h->client || !h->model || !h->inObj || !h->outObj || !g_ANEReq) {
            return -1;
        }
        if (num_tokens <= 0) {
            return -1;
        }
        if (!initial_input || input_count == 0 || !output_logits || output_count == 0) {
            return -1;
        }
        if ((size_t)input_count * sizeof(float) > h->inBytes || (size_t)output_count * sizeof(float) > h->outBytes) {
            return -1;
        }

        ane_bridge_client_write_input(h, initial_input, (int)input_count);

        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[h->inObj], @[@0], @[h->outObj], @[@0], nil, nil, @0);
        if (!req) {
            return -2;
        }

        NSError *e = nil;
        BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            h->client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), h->model, req, YES, &e);
        if (!mapped) {
            return -3;
        }

        SEL evalSel = @selector(evaluateWithModel:options:request:qos:error:);
        if (![h->client respondsToSelector:evalSel]) {
            ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
            return -4;
        }

        float *nextInput = (float *)calloc((size_t)input_count, sizeof(float));
        if (!nextInput) {
            ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
            return -5;
        }

        int rc = 0;
        for (int i = 0; i < num_tokens; i++) {
            e = nil;
            BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
                h->client, evalSel, h->model, @{}, req, 21, &e);
            if (!ok) {
                rc = -6;
                break;
            }

            ane_bridge_client_read_output(h, output_logits, (int)output_count);
            if (i + 1 >= num_tokens) {
                continue;
            }

            if (token_callback) {
                token_callback(output_logits, output_count, nextInput, input_count, callback_ctx);
            } else {
                size_t copyCount = input_count < output_count ? input_count : output_count;
                memcpy(nextInput, output_logits, copyCount * sizeof(float));
            }
            ane_bridge_client_write_input(h, nextInput, (int)input_count);
        }

        free(nextInput);
        ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
        return rc;
    }
}

void *ane_bridge_create_shared_event(void) {
    @autoreleasepool {
        Class cls = NSClassFromString(@"IOSurfaceSharedEvent");
        if (!cls) {
            return NULL;
        }
        id ev = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
        if ([ev respondsToSelector:@selector(initWithOptions:)]) {
            ev = ((id(*)(id, SEL, unsigned long long))objc_msgSend)(ev, @selector(initWithOptions:), 0ULL);
        } else {
            ev = ((id(*)(id, SEL))objc_msgSend)(ev, @selector(init));
        }
        if (!ev) {
            return NULL;
        }
        return (__bridge_retained void *)ev;
    }
}

void ane_bridge_release_objc(void *obj) {
    if (!obj) {
        return;
    }
    CFBridgingRelease(obj);
}

unsigned int ane_bridge_shared_event_port(void *shared_event_obj) {
    id ev = (__bridge id)shared_event_obj;
    if (!ev || ![ev respondsToSelector:@selector(eventPort)]) {
        return 0;
    }
    return ((unsigned int(*)(id, SEL))objc_msgSend)(ev, @selector(eventPort));
}

bool ane_bridge_eval_with_wait_event(ANEClientHandle *h,
                                     void *wait_shared_event_obj,
                                     uint64_t wait_value,
                                     bool disable_fences_use_shared_events,
                                     bool enable_fw_to_fw_signal) {
    @autoreleasepool {
        if (!h || !h->client || !h->model || !h->inObj || !h->outObj || !wait_shared_event_obj) {
            return false;
        }
        if (!g_ANEWaitEvent || !g_ANESharedEvents || !g_ANEReq) {
            return false;
        }
        id waitShared = (__bridge id)wait_shared_event_obj;
        id waitEvent = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(
            g_ANEWaitEvent, @selector(waitEventWithValue:sharedEvent:), wait_value, waitShared);
        if (!waitEvent) {
            return false;
        }
        id sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            g_ANESharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[], @[waitEvent]);
        if (!sharedEvents) {
            return false;
        }
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:),
            @[h->inObj], @[@0], @[h->outObj], @[@0], nil, nil, @0, sharedEvents, @1);
        if (!req) {
            return false;
        }
        ane_attach_request_completion_handler(req);

        NSError *e = nil;
        BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            h->client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), h->model, req, YES, &e);
        if (!mapped) {
            return false;
        }

        NSMutableDictionary *opts = [NSMutableDictionary dictionary];
        if (disable_fences_use_shared_events) {
            id k = ane_const_obj("kANEFDisableIOFencesUseSharedEventsKey");
            if (!k) k = @"kANEFDisableIOFencesUseSharedEventsKey";
            opts[k] = @YES;
        }
        if (enable_fw_to_fw_signal) {
            id k = ane_const_obj("kANEFEnableFWToFWSignal");
            if (!k) k = @"kANEFEnableFWToFWSignal";
            opts[k] = @YES;
        }

        SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
        if (![h->client respondsToSelector:evalSel]) {
            evalSel = @selector(evaluateWithModel:options:request:qos:error:);
        }
        e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            h->client, evalSel, h->model, opts, req, 21, &e);
        bool doUnmap = false;
        const char *ru = getenv("ANE_BRIDGE_UNMAP");
        if (ru && ru[0] == '1') {
            doUnmap = true;
        }
        if (doUnmap) {
            ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
        }
        return ok ? true : false;
    }
}

bool ane_bridge_eval_with_signal_event_obj(ANEClientHandle *h,
                                           void *signal_shared_event_obj,
                                           uint64_t signal_value,
                                           bool disable_fences_use_shared_events,
                                           bool enable_fw_to_fw_signal) {
    @autoreleasepool {
        if (!h || !h->client || !h->model || !h->inObj || !h->outObj || !signal_shared_event_obj) {
            return false;
        }
        if (!g_ANESignalEvent || !g_ANESharedEvents || !g_ANEReq) {
            return false;
        }
        id signalShared = (__bridge id)signal_shared_event_obj;
        id signalEvent = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
            g_ANESignalEvent, @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:), signal_value, 0u, 5ll, signalShared);
        if (!signalEvent) {
            return false;
        }
        id sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            g_ANESharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[signalEvent], @[]);
        if (!sharedEvents) {
            return false;
        }
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:),
            @[h->inObj], @[@0], @[h->outObj], @[@0], nil, nil, @0, sharedEvents, @1);
        if (!req) {
            return false;
        }
        ane_attach_request_completion_handler(req);

        NSError *e = nil;
        BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            h->client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), h->model, req, YES, &e);
        if (!mapped) {
            return false;
        }

        NSMutableDictionary *opts = [NSMutableDictionary dictionary];
        if (disable_fences_use_shared_events) {
            id k = ane_const_obj("kANEFDisableIOFencesUseSharedEventsKey");
            if (!k) k = @"kANEFDisableIOFencesUseSharedEventsKey";
            opts[k] = @YES;
        }
        if (enable_fw_to_fw_signal) {
            id k = ane_const_obj("kANEFEnableFWToFWSignal");
            if (!k) k = @"kANEFEnableFWToFWSignal";
            opts[k] = @YES;
        }

        SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
        if (![h->client respondsToSelector:evalSel]) {
            evalSel = @selector(evaluateWithModel:options:request:qos:error:);
        }
        e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            h->client, evalSel, h->model, opts, req, 21, &e);
        bool doUnmap = false;
        const char *ru = getenv("ANE_BRIDGE_UNMAP");
        if (ru && ru[0] == '1') {
            doUnmap = true;
        }
        if (doUnmap) {
            ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
        }
        return ok ? true : false;
    }
}

static id ane_iosurface_shared_event_from_port(mach_port_t port) {
    if (port == MACH_PORT_NULL) {
        return nil;
    }
    Class cls = NSClassFromString(@"IOSurfaceSharedEvent");
    if (!cls) {
        return nil;
    }
    SEL classSel = NSSelectorFromString(@"sharedEventWithMachPort:");
    if ([cls respondsToSelector:classSel]) {
        return ((id(*)(Class, SEL, mach_port_t))objc_msgSend)(cls, classSel, port);
    }
    id ev = ((id(*)(Class, SEL))objc_msgSend)(cls, @selector(alloc));
    if (!ev) {
        return nil;
    }
    SEL initSel = NSSelectorFromString(@"initWithMachPort:");
    if ([ev respondsToSelector:initSel]) {
        return ((id(*)(id, SEL, mach_port_t))objc_msgSend)(ev, initSel, port);
    }
    SEL initWithOptsSel = NSSelectorFromString(@"initWithMachPort:options:");
    if ([ev respondsToSelector:initWithOptsSel]) {
        return ((id(*)(id, SEL, mach_port_t, unsigned long long))objc_msgSend)(ev, initWithOptsSel, port, 0ULL);
    }
    return nil;
}

static id ane_build_shared_request(ANEClientHandle *h, id sharedEvents) {
    if (!h || !h->inObj || !h->outObj || !g_ANEReq) {
        return nil;
    }
    return ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:),
        @[h->inObj], @[@0], @[h->outObj], @[@0], nil, nil, @0, sharedEvents, @1);
}

static NSMutableDictionary *ane_shared_event_options(bool disable_fences_use_shared_events,
                                                     bool enable_fw_to_fw_signal) {
    NSMutableDictionary *opts = [NSMutableDictionary dictionary];
    if (disable_fences_use_shared_events) {
        id k = ane_const_obj("kANEFDisableIOFencesUseSharedEventsKey");
        if (!k) {
            k = @"kANEFDisableIOFencesUseSharedEventsKey";
        }
        opts[k] = @YES;
    }
    id fwKey = ane_const_obj("kANEFEnableFWToFWSignal");
    if (!fwKey) {
        fwKey = @"kANEFEnableFWToFWSignal";
    }
    opts[fwKey] = enable_fw_to_fw_signal ? @YES : @NO;
    return opts;
}

static BOOL ane_eval_direct_request(ANEClientHandle *h, id req, NSDictionary *opts, NSError **err) {
    SEL evalSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    if (![h->client respondsToSelector:evalSel]) {
        evalSel = @selector(evaluateWithModel:options:request:qos:error:);
    }
    return ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
        h->client, evalSel, h->model, opts ?: @{}, req, 21, err);
}

static int ane_eval_shared_events_request(ANEClientHandle *h,
                                          id req,
                                          NSDictionary *opts,
                                          float *output,
                                          uint32_t output_count) {
    if (!h || !req) {
        return -1;
    }
    SEL setCompletionSel = NSSelectorFromString(@"setCompletionHandler:");
    if (![req respondsToSelector:setCompletionSel]) {
        return -2;
    }
    dispatch_semaphore_t completionSem = dispatch_semaphore_create(0);
    __block BOOL completionFired = NO;
    __block BOOL completionSuccess = NO;
    __block NSError *completionError = nil;
    void (^handler)(BOOL, NSError *) = ^(BOOL success, NSError *error) {
        completionFired = YES;
        completionSuccess = success;
        completionError = error;
        dispatch_semaphore_signal(completionSem);
    };
    id copiedHandler = [handler copy];
    ((void(*)(id, SEL, id))objc_msgSend)(req, setCompletionSel, copiedHandler);
    objc_setAssociatedObject(req, g_completion_handler_assoc_key, copiedHandler, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

    NSError *e = nil;
    BOOL mapped = ((BOOL(*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        h->client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), h->model, req, YES, &e);
    if (!mapped) {
        return -3;
    }

    e = nil;
    BOOL ok = ane_eval_direct_request(h, req, opts, &e);
    long waited = -1;
    if (ok) {
        waited = dispatch_semaphore_wait(
            completionSem,
            dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * NSEC_PER_SEC)));
    }
    ((void(*)(id, SEL, id, id))objc_msgSend)(h->client, @selector(unmapIOSurfacesWithModel:request:), h->model, req);
    if (!ok) {
        return -4;
    }
    if (waited != 0 || !completionFired || !completionSuccess || completionError != nil) {
        return -5;
    }
    if (output && output_count > 0) {
        ane_bridge_client_read_output(h, output, (int)output_count);
    }
    return 0;
}

int ane_bridge_eval_with_signal_event(ane_bridge_client_t client,
                                      const float *input, uint32_t input_count,
                                      float *output, uint32_t output_count,
                                      mach_port_t signal_event_port,
                                      uint64_t signal_value) {
    @autoreleasepool {
        ANEClientHandle *h = client;
        if (!h || !h->client || !h->model || !h->inObj || !h->outObj || !g_ANESignalEvent || !g_ANESharedEvents || !g_ANEReq) {
            return -1;
        }
        if (input_count > 0 && input == NULL) {
            return -1;
        }
        if (output_count > 0 && output == NULL) {
            return -1;
        }
        if ((size_t)input_count * sizeof(float) > h->inBytes || (size_t)output_count * sizeof(float) > h->outBytes) {
            return -1;
        }
        if (input && input_count > 0) {
            ane_bridge_client_write_input(h, input, (int)input_count);
        }
        id signalShared = ane_iosurface_shared_event_from_port(signal_event_port);
        if (!signalShared) {
            return -2;
        }
        id signalEvent = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
            g_ANESignalEvent,
            @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
            signal_value, 0u, 5ll, signalShared);
        if (!signalEvent) {
            return -2;
        }
        id sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            g_ANESharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[signalEvent], @[]);
        if (!sharedEvents) {
            return -2;
        }
        id req = ane_build_shared_request(h, sharedEvents);
        if (!req) {
            return -2;
        }
        NSDictionary *opts = ane_shared_event_options(true, false);
        return ane_eval_shared_events_request(h, req, opts, output, output_count);
    }
}

int ane_bridge_eval_bidirectional(ane_bridge_client_t client,
                                  const float *input, uint32_t input_count,
                                  float *output, uint32_t output_count,
                                  mach_port_t wait_event_port,
                                  uint64_t wait_value,
                                  mach_port_t signal_event_port,
                                  uint64_t signal_value) {
    @autoreleasepool {
        ANEClientHandle *h = client;
        if (!h || !h->client || !h->model || !h->inObj || !h->outObj || !g_ANEWaitEvent || !g_ANESignalEvent || !g_ANESharedEvents || !g_ANEReq) {
            return -1;
        }
        if (input_count > 0 && input == NULL) {
            return -1;
        }
        if (output_count > 0 && output == NULL) {
            return -1;
        }
        if ((size_t)input_count * sizeof(float) > h->inBytes || (size_t)output_count * sizeof(float) > h->outBytes) {
            return -1;
        }
        if (input && input_count > 0) {
            ane_bridge_client_write_input(h, input, (int)input_count);
        }

        id waitShared = ane_iosurface_shared_event_from_port(wait_event_port);
        id signalShared = ane_iosurface_shared_event_from_port(signal_event_port);
        if (!waitShared || !signalShared) {
            return -2;
        }
        id waitEvent = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(
            g_ANEWaitEvent, @selector(waitEventWithValue:sharedEvent:), wait_value, waitShared);
        id signalEvent = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
            g_ANESignalEvent,
            @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
            signal_value, 0u, 5ll, signalShared);
        if (!waitEvent || !signalEvent) {
            return -2;
        }
        id sharedEvents = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            g_ANESharedEvents, @selector(sharedEventsWithSignalEvents:waitEvents:), @[signalEvent], @[waitEvent]);
        if (!sharedEvents) {
            return -2;
        }
        id req = ane_build_shared_request(h, sharedEvents);
        if (!req) {
            return -2;
        }
        NSDictionary *opts = ane_shared_event_options(true, false);
        return ane_eval_shared_events_request(h, req, opts, output, output_count);
    }
}

int ane_bridge_signal_event_cpu(mach_port_t event_port, uint64_t value) {
    @autoreleasepool {
        id ev = ane_iosurface_shared_event_from_port(event_port);
        if (!ev) {
            return -1;
        }
        SEL setSel = @selector(setSignaledValue:);
        if (![ev respondsToSelector:setSel]) {
            return -2;
        }
        ((void(*)(id, SEL, unsigned long long))objc_msgSend)(ev, setSel, value);
        return 0;
    }
}

int ane_bridge_wait_event_cpu(mach_port_t event_port, uint64_t value, uint32_t timeout_ms) {
    @autoreleasepool {
        id ev = ane_iosurface_shared_event_from_port(event_port);
        if (!ev) {
            return -1;
        }
        SEL getSel = @selector(signaledValue);
        if (![ev respondsToSelector:getSel]) {
            return -2;
        }
        unsigned long long cur = ((unsigned long long(*)(id, SEL))objc_msgSend)(ev, getSel);
        if (cur >= value) {
            return 0;
        }
        uint64_t waitedUs = 0;
        const uint32_t sleepUs = 50;
        const uint64_t timeoutUs = (uint64_t)timeout_ms * 1000ULL;
        while (waitedUs < timeoutUs) {
            usleep(sleepUs);
            waitedUs += sleepUs;
            cur = ((unsigned long long(*)(id, SEL))objc_msgSend)(ev, getSel);
            if (cur >= value) {
                return 0;
            }
        }
        return 1;
    }
}

void *ane_bridge_zero_copy_buffer(IOSurfaceRef surface, size_t bytes) {
    @autoreleasepool {
        if (!surface || bytes == 0) {
            return NULL;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            return NULL;
        }
        void *base = IOSurfaceGetBaseAddress(surface);
        id<MTLBuffer> buf = nil;
        if (base) {
            buf = [dev newBufferWithBytesNoCopy:base length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        }
        return buf ? (__bridge_retained void *)buf : NULL;
    }
}

bool ane_bridge_gcd_overlap(ANEClientHandle *h,
                            void *wait_shared_event_obj,
                            uint64_t wait_value,
                            bool disable_fences_use_shared_events,
                            bool enable_fw_to_fw_signal,
                            uint32_t cpu_rounds,
                            double *out_total_ms) {
    @autoreleasepool {
        if (!h || !wait_shared_event_obj) {
            return false;
        }
        dispatch_queue_t q = dispatch_queue_create("ane.bridge.overlap", DISPATCH_QUEUE_SERIAL);
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block bool evalOK = false;
        volatile double sink = 0.0;
        double t0 = (double)CFAbsoluteTimeGetCurrent() * 1000.0;
        dispatch_async(q, ^{
            evalOK = ane_bridge_eval_with_wait_event(
                h, wait_shared_event_obj, wait_value,
                disable_fences_use_shared_events,
                enable_fw_to_fw_signal);
            dispatch_semaphore_signal(sem);
        });
        for (uint32_t i = 0; i < cpu_rounds; i++) {
            double x = (double)(i + 1);
            sink += sqrt(x) * 0.000001;
        }
        (void)sink;
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        double t1 = (double)CFAbsoluteTimeGetCurrent() * 1000.0;
        if (out_total_ms) {
            *out_total_ms = t1 - t0;
        }
        return evalOK;
    }
}
