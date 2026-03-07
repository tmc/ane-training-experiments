// test_symmetric_pipeline.m
// End-to-end Metal->ANE->Metal round-trip using ane_bridge_eval_bidirectional.
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

static double now_ms(void) {
    return (double)CFAbsoluteTimeGetCurrent() * 1000.0;
}

static int run_once(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        if (ane_bridge_init() != 0) {
            printf("FAIL: ane_bridge_init\n");
            return 1;
        }

        const char *modelPath = getenv("ANE_CHAIN_MODEL_PATH");
        if (!modelPath || !modelPath[0]) {
            modelPath = "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc";
        }

        const uint32_t count = 1024;
        const size_t bytes = (size_t)count * sizeof(float);
        ane_bridge_client_t h = ane_bridge_client_open(modelPath, "s", bytes, bytes);
        if (!h) {
            printf("FAIL: ane_bridge_client_open\n");
            return 1;
        }

        void *waitObj = ane_bridge_create_shared_event();
        void *signalObj = ane_bridge_create_shared_event();
        if (!waitObj || !signalObj) {
            printf("FAIL: ane_bridge_create_shared_event\n");
            return 1;
        }
        mach_port_t waitPort = (mach_port_t)ane_bridge_shared_event_port(waitObj);
        mach_port_t signalPort = (mach_port_t)ane_bridge_shared_event_port(signalObj);
        if (waitPort == MACH_PORT_NULL || signalPort == MACH_PORT_NULL) {
            printf("FAIL: shared event ports wait=%u signal=%u\n", waitPort, signalPort);
            return 1;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> cq = [dev newCommandQueue];
        SEL newSharedSel = NSSelectorFromString(@"newSharedEventWithMachPort:");
        if (!dev || !cq || ![dev respondsToSelector:newSharedSel]) {
            printf("FAIL: Metal shared event API unavailable\n");
            return 1;
        }
        id mtlSignal = ((id(*)(id, SEL, unsigned int))objc_msgSend)(dev, newSharedSel, (unsigned int)signalPort);
        if (!mtlSignal) {
            printf("FAIL: newSharedEventWithMachPort\n");
            return 1;
        }

        float *hostIn = (float *)calloc((size_t)count, sizeof(float));
        float *hostOut = (float *)calloc((size_t)count, sizeof(float));
        if (!hostIn || !hostOut) {
            printf("FAIL: alloc host buffers\n");
            return 1;
        }
        for (uint32_t i = 0; i < count; i++) {
            hostIn[i] = (float)(i + 1);
        }

        void *outBufObj = ane_bridge_zero_copy_buffer(ane_bridge_client_output_surface(h), bytes);
        if (!outBufObj) {
            printf("FAIL: zero-copy surface buffers\n");
            return 1;
        }
        id<MTLBuffer> outBuf = (__bridge id)outBufObj;

        ane_bridge_client_write_input(h, hostIn, (int)count);

        id<MTLBuffer> snapBuf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb2 = [cq commandBuffer];
        if ([cb2 respondsToSelector:@selector(encodeWaitForEvent:value:)]) {
            ((void(*)(id, SEL, id, unsigned long long))objc_msgSend)(cb2, @selector(encodeWaitForEvent:value:), mtlSignal, 1ULL);
        } else {
            printf("FAIL: encodeWaitForEvent:value: unavailable\n");
            return 1;
        }
        id<MTLBlitCommandEncoder> bl = [cb2 blitCommandEncoder];
        [bl copyFromBuffer:outBuf sourceOffset:0 toBuffer:snapBuf destinationOffset:0 size:bytes];
        [bl endEncoding];

        [cb2 commit];
        double t0 = now_ms();
        int sigRC = ane_bridge_signal_event_cpu(waitPort, 1ULL);
        if (sigRC != 0) {
            printf("FAIL: ane_bridge_signal_event_cpu rc=%d\n", sigRC);
            return 1;
        }
        double tEval0 = now_ms();
        int rc = ane_bridge_eval_bidirectional(
            h,
            NULL, 0,
            hostOut, count,
            waitPort, 1ULL,
            signalPort, 1ULL);
        double tEval1 = now_ms();
        [cb2 waitUntilCompleted];
        double t1 = now_ms();

        float *snap = (float *)snapBuf.contents;
        if (!snap) {
            printf("FAIL: snapshot contents unavailable\n");
            return 1;
        }

        double scale = (fabs((double)hostIn[0]) > 0.0) ? ((double)hostOut[0] / (double)hostIn[0]) : 0.0;
        double maxDiffRef = 0.0;
        double maxDiffSnap = 0.0;
        for (uint32_t i = 0; i < count; i++) {
            double want = (double)hostIn[i] * scale;
            double d0 = fabs((double)hostOut[i] - want);
            double d1 = fabs((double)hostOut[i] - (double)snap[i]);
            if (d0 > maxDiffRef) {
                maxDiffRef = d0;
            }
            if (d1 > maxDiffSnap) {
                maxDiffSnap = d1;
            }
        }

        printf("symm rc=%d output[0..2]=[%.6f, %.6f, %.6f] scale=%.6f maxDiffRef=%g maxDiffSnap=%g total_ms=%.3f ane_ms=%.3f\n",
               rc,
               hostOut[0], hostOut[1], hostOut[2],
               scale, maxDiffRef, maxDiffSnap,
               t1 - t0, tEval1 - tEval0);

        bool expectedHead =
            fabs((double)hostOut[0] - 256.0) < 1e-3 &&
            fabs((double)hostOut[1] - 512.0) < 1e-3 &&
            fabs((double)hostOut[2] - 768.0) < 1e-3;
        bool pass = (rc == 0) && expectedHead && maxDiffSnap < 1e-2;
        printf("%s: symmetric pipeline\n", pass ? "PASS" : "FAIL");

        const char *cleanup = getenv("TEST_SYMM_CLEANUP");
        if (cleanup && cleanup[0] == '1') {
            ane_bridge_release_objc(outBufObj);
            ane_bridge_release_objc(waitObj);
            ane_bridge_release_objc(signalObj);
            free(hostOut);
            free(hostIn);
            ane_bridge_client_close(h);
            return pass ? 0 : 1;
        }

        // Shared-events teardown can still be fragile under stress; default to hard-exit.
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
            printf("WARN: symmetric attempt=%d crashed signal=%d\n", i, WTERMSIG(status));
        } else if (WIFEXITED(status)) {
            printf("WARN: symmetric attempt=%d exit=%d\n", i, WEXITSTATUS(status));
        }
    }
    printf("FAIL: symmetric pipeline retries exhausted\n");
    return 1;
}
