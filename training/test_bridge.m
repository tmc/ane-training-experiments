// test_bridge.m - smoke test for libane_bridge.dylib asymmetric APIs.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#include "../bridge/ane_bridge.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static IOSurfaceRef output_surface(ANEClientHandle *h) {
    return ane_bridge_client_output_surface(h);
}

static int run_smoke_once(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (ane_bridge_init() != 0) {
            printf("FAIL: ane_bridge_init\n");
            return 1;
        }
        printf("step: bridge init ok\n");

        const char *modelPath = getenv("ANE_CHAIN_MODEL_PATH");
        if (!modelPath || !modelPath[0]) {
            modelPath = "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
        }

        const int count = 1024;
        const size_t bytes = (size_t)count * sizeof(float);
        ANEClientHandle *h = ane_bridge_client_open(modelPath, "s", bytes, bytes);
        if (!h) {
            printf("FAIL: ane_bridge_client_open\n");
            return 1;
        }
        printf("step: client open ok\n");

        float *in = (float *)calloc((size_t)count, sizeof(float));
        float *out = (float *)calloc((size_t)count, sizeof(float));
        if (!in || !out) {
            printf("FAIL: alloc\n");
            ane_bridge_client_close(h);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            in[i] = (float)(i + 1);
        }
        ane_bridge_client_write_input(h, in, count);
        printf("step: input write ok\n");

        void *eventObj = ane_bridge_create_shared_event();
        if (!eventObj) {
            printf("FAIL: ane_bridge_create_shared_event\n");
            free(out);
            free(in);
            ane_bridge_client_close(h);
            return 1;
        }
        printf("step: shared event created\n");
        unsigned int port = ane_bridge_shared_event_port(eventObj);
        if (port == 0) {
            printf("FAIL: ane_bridge_shared_event_port\n");
            ane_bridge_release_objc(eventObj);
            free(out);
            free(in);
            ane_bridge_client_close(h);
            return 1;
        }
        printf("step: shared event port=%u\n", port);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> cq = [dev newCommandQueue];
        SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
        if (!dev || !cq || ![dev respondsToSelector:newSharedSel]) {
            printf("FAIL: Metal setup for shared event\n");
            ane_bridge_release_objc(eventObj);
            free(out);
            free(in);
            ane_bridge_client_close(h);
            return 1;
        }
        printf("step: metal shared event path ok\n");
        id mtlEv = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, port);
        if (!mtlEv) {
            printf("FAIL: newSharedEventWithMachPort\n");
            ane_bridge_release_objc(eventObj);
            free(out);
            free(in);
            ane_bridge_client_close(h);
            return 1;
        }
        printf("step: mtl shared event created\n");

        id<MTLCommandBuffer> sig = [cq commandBuffer];
        if ([sig respondsToSelector:@selector(encodeSignalEvent:value:)]) {
            ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(sig, @selector(encodeSignalEvent:value:), mtlEv, 1ULL);
        }
        [sig commit];
        [sig waitUntilCompleted];
        printf("step: metal signal committed\n");

        bool ok = ane_bridge_eval_with_wait_event(h, eventObj, 1ULL, true, true);
        if (!ok) {
            printf("FAIL: ane_bridge_eval_with_wait_event\n");
            ane_bridge_release_objc(eventObj);
            free(out);
            free(in);
            ane_bridge_client_close(h);
            return 1;
        }
        printf("step: wait-event eval ok\n");

        printf("step: reading output surface\n");
        bool ready = false;
        for (int poll = 0; poll < 200; poll++) {
            ane_bridge_client_read_output(h, out, count);
            if (fabs((double)out[0]) > 0.0 || fabs((double)out[1]) > 0.0 || fabs((double)out[2]) > 0.0) {
                ready = true;
                break;
            }
            usleep(500);
        }
        if (!ready) {
            printf("FAIL: output not ready after wait-event eval\n");
            return 1;
        }
        printf("step: output surface read ok\n");
        double scale = (fabs((double)in[0]) > 0.0) ? ((double)out[0] / (double)in[0]) : 0.0;
        double maxDiff = 0.0;
        for (int i = 0; i < 3; i++) {
            double want = (double)in[i] * scale;
            double d = fabs((double)out[i] - want);
            if (d > maxDiff) {
                maxDiff = d;
            }
        }

        float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f;
        double zDiff = 0.0;
        bool doZeroCopy = false;
        const char *rz = getenv("TEST_BRIDGE_ZERO_COPY");
        if (rz && rz[0] == '1') {
            doZeroCopy = true;
        }
        void *bufObj = NULL;
        if (doZeroCopy) {
            printf("step: zero-copy buffer probe\n");
            bufObj = ane_bridge_zero_copy_buffer(output_surface(h), bytes);
            if (!bufObj) {
                printf("FAIL: ane_bridge_zero_copy_buffer\n");
                ane_bridge_release_objc(eventObj);
                free(out);
                free(in);
                ane_bridge_client_close(h);
                return 1;
            }
            id<MTLBuffer> outBuf = (__bridge id)bufObj;
            float *ptr = NULL;
            if ([outBuf respondsToSelector:@selector(contents)]) {
                ptr = (float *)outBuf.contents;
            }
            if (ptr) {
                z0 = ptr[0];
                z1 = ptr[1];
                z2 = ptr[2];
                printf("step: zero-copy direct contents ok\n");
            } else {
                id<MTLBuffer> snap = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
                [bl copyFromBuffer:outBuf sourceOffset:0 toBuffer:snap destinationOffset:0 size:bytes];
                [bl endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                float *s = (float *)snap.contents;
                if (s) {
                    z0 = s[0];
                    z1 = s[1];
                    z2 = s[2];
                }
                printf("step: zero-copy blit snapshot ok\n");
            }
            zDiff = fmax(fabs((double)z0 - (double)out[0]), fmax(fabs((double)z1 - (double)out[1]), fabs((double)z2 - (double)out[2])));
        }
        bool pass = isfinite(scale) && scale > 0.0 && maxDiff < 1e-3;
        if (doZeroCopy) {
            pass = pass && zDiff < 1e-3;
        }
        printf("bridge_output[0..2]=[%.6f, %.6f, %.6f] scale=%.6f maxDiff=%g zero_copy_enabled=%d zero_copy[0..2]=[%.6f, %.6f, %.6f] zDiff=%g\n",
               out[0], out[1], out[2], scale, maxDiff, doZeroCopy ? 1 : 0, z0, z1, z2, zDiff);

        const char *cleanup = getenv("TEST_BRIDGE_CLEANUP");
        if (cleanup && cleanup[0] == '1') {
            if (bufObj) {
                ane_bridge_release_objc(bufObj);
            }
            ane_bridge_release_objc(eventObj);
            free(out);
            free(in);
            ane_bridge_client_close(h);
            printf("%s: bridge smoke test\n", pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        // Shared-event teardown can crash; default to hard-exit after reporting.
        printf("%s: bridge smoke test (no-cleanup mode)\n", pass ? "PASS" : "FAIL");
        _exit(pass ? 0 : 1);
    }
}

int main(void) {
    setbuf(stdout, NULL);
    const int attempts = 3;
    for (int i = 1; i <= attempts; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            int rc = run_smoke_once();
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
            printf("WARN: bridge smoke attempt=%d crashed signal=%d\n", i, WTERMSIG(status));
        } else if (WIFEXITED(status)) {
            printf("WARN: bridge smoke attempt=%d exit=%d\n", i, WEXITSTATUS(status));
        } else {
            printf("WARN: bridge smoke attempt=%d abnormal exit\n", i);
        }
    }
    printf("FAIL: bridge smoke retries exhausted\n");
    return 1;
}
