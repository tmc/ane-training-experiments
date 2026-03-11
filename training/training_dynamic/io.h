// io.h — IOSurface helpers, NEON conversion, kernel compile/eval
// Updated for GQA (Qwen3-0.6B): Q_DIM != DIM, separate KV heads
#pragma once
#include "config.h"

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Blob builders for const weights (mask, rms)
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// IOSurface I/O (channel-first [C,S] layout, fp16 on surface)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
static void io_copy_rect(IOSurfaceRef dst, int dst_ch, int dst_sp,
                         IOSurfaceRef src, int src_ch, int src_sp,
                         int channels, int cols) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    _Float16 *db = (_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*dst_sp;
    _Float16 *sb = (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*src_sp;
    for (int c = 0; c < channels; c++)
        memcpy(db + c*dst_sp, sb + c*src_sp, (size_t)cols * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}

// fp16 IOSurface I/O (for dynamic matmul kernels with fp16 input/output)
static void io_write_dyn(IOSurfaceRef s, const float *act, int ic, int seq,
                         const float *W, int oc) {
    int sp = seq + oc;
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < ic; d++) {
        cvt_f32_f16(buf + d*sp, act + d*seq, seq);
        cvt_f32_f16(buf + d*sp + seq, W + d*oc, oc);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// Read output from dynamic matmul kernel: [1, OC, 1, SEQ]
static void io_read_dyn(IOSurfaceRef s, float *out, int oc, int seq) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(out, (_Float16*)IOSurfaceGetBaseAddress(s), oc * seq);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// RMSNorm backward: input layout per channel is dy[SEQ], x[SEQ], w[1].
#define RMS_BWD_SP (2*SEQ + 1)
static void io_write_rmsnorm_bwd(IOSurfaceRef s, const float *dy, const float *x, const float *w) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        _Float16 *row = buf + d*RMS_BWD_SP;
        cvt_f32_f16(row, dy + d*SEQ, SEQ);
        cvt_f32_f16(row + SEQ, x + d*SEQ, SEQ);
        row[2*SEQ] = (_Float16)w[d];
    }
    IOSurfaceUnlock(s, 0, NULL);
}
// Final RMSNorm backward: channels are concatenated as dy, x, and w broadcast across SEQ.
static void io_write_rmsnorm_bwd_chan(IOSurfaceRef s, const float *dy, const float *x, const float *w) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        _Float16 wd = (_Float16)w[d];
        cvt_f32_f16(buf + d*SEQ, dy + d*SEQ, SEQ);
        cvt_f32_f16(buf + (DIM + d)*SEQ, x + d*SEQ, SEQ);
        _Float16 *wrow = buf + (2*DIM + d)*SEQ;
        for (int t = 0; t < SEQ; t++) wrow[t] = wd;
    }
    IOSurfaceUnlock(s, 0, NULL);
}

static bool compile_model_mil_w(NSString *mil, NSDictionary *weights, void **model_out, void **tmp_dir_out) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return false; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return false;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return false;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    *model_out = (void*)CFBridgingRetain(mdl);
    *tmp_dir_out = (void*)CFBridgingRetain(td);
    return true;
    }
}

static void *make_request_multi(IOSurfaceRef *inputs, int nin, IOSurfaceRef *outputs, int nout) {
    NSMutableArray *in_arr = [NSMutableArray arrayWithCapacity:(NSUInteger)nin];
    NSMutableArray *in_idx = [NSMutableArray arrayWithCapacity:(NSUInteger)nin];
    NSMutableArray *out_arr = [NSMutableArray arrayWithCapacity:(NSUInteger)nout];
    NSMutableArray *out_idx = [NSMutableArray arrayWithCapacity:(NSUInteger)nout];
    for (int i = 0; i < nin; i++) {
        [in_arr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), inputs[i])];
        [in_idx addObject:@(i)];
    }
    for (int i = 0; i < nout; i++) {
        [out_arr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outputs[i])];
        [out_idx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        in_arr, in_idx, out_arr, out_idx, nil, nil, @0);
    return (void*)CFBridgingRetain(req);
}

// Compile MIL to ANE kernel
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    if (!compile_model_mil_w(mil, weights, &k->model, &k->tmpDir)) {
        free(k);
        return NULL;
    }
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    IOSurfaceRef ins[] = {k->ioIn};
    IOSurfaceRef outs[] = {k->ioOut};
    k->request = make_request_multi(ins, 1, outs, 1);
    return k;
}
static void unload_compiled_model(void *model, void *tmp_dir) {
    if (!model || !tmp_dir) return;
    id mdl = (__bridge id)model;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)tmp_dir error:nil];
    CFRelease(model);
    CFRelease(tmp_dir);
}
static void free_kern(Kern *k) {
    if (!k) return;
    if (k->ioIn) CFRelease(k->ioIn);
    if (k->ioOut) CFRelease(k->ioOut);
    unload_compiled_model(k->model, k->tmpDir);
    if (k->request) CFRelease(k->request);
    free(k);
}
static void ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void ane_eval_model_req(void *model, void *request) {
    id mdl = (__bridge id)model; id req = (__bridge id)request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void ane_eval_req(Kern *k, void *request) {
    ane_eval_model_req(k->model, request);
}
static void *make_request(Kern *k, IOSurfaceRef ioIn) {
    IOSurfaceRef ins[] = {ioIn};
    IOSurfaceRef outs[] = {k->ioOut};
    return make_request_multi(ins, 1, outs, 1);
}

// ===== Per-layer weight staging for GQA =====
// sdpaFwd: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16 — xnorm then QKV weights.
#define SDPA_FWD_SP (SEQ + Q_DIM + KV_DIM + KV_DIM)
static void stage_sdpa_fwd_weights(IOSurfaceRef s, const float *Wq, const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        _Float16 *row = buf + d*SDPA_FWD_SP;
        cvt_f32_f16(row + SEQ,                  Wq + d*Q_DIM, Q_DIM);
        cvt_f32_f16(row + SEQ + Q_DIM,          Wk + d*KV_DIM, KV_DIM);
        cvt_f32_f16(row + SEQ + Q_DIM + KV_DIM, Wv + d*KV_DIM, KV_DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_sdpa_fwd_acts(IOSurfaceRef s, const float *xnorm) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*SDPA_FWD_SP, xnorm + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// woFwd: [1, Q_DIM, 1, SEQ + DIM] fp16 — Wo: [Q_DIM, DIM]
#define WO_FWD_SP (SEQ + DIM)
static void stage_wo_fwd_weights(IOSurfaceRef s, const float *Wo) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP + SEQ, Wo + d*DIM, DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_wo_fwd_acts(IOSurfaceRef s, const float *attn_out) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP, attn_out + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnFused: [1, DIM, 1, 2*SEQ + 3*HIDDEN] fp16 — x2norm, x2, then FFN weights.
#define FFN_FUSED_SP (2*SEQ + 3*HIDDEN)
static void stage_ffn_fused_weights(IOSurfaceRef s,
                                    const float *W1t, const float *W3t, const float *W2_orig) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        _Float16 *row = buf + d*FFN_FUSED_SP;
        cvt_f32_f16(row + 2*SEQ,             W1t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(row + 2*SEQ + HIDDEN,    W3t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(row + 2*SEQ + 2*HIDDEN,  W2_orig + d*HIDDEN, HIDDEN);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_fused_acts(IOSurfaceRef s, const float *x2norm, const float *x2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP,       x2norm + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ, x2 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnBwdW2t: [1, DIM, 1, SEQ+HIDDEN] fp16
#define FFN_BWD_W2T_SP (SEQ + HIDDEN)
static void stage_ffn_bwd_w2t_weights(IOSurfaceRef s, const float *W2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP + SEQ, W2 + d*HIDDEN, HIDDEN);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w2t_acts(IOSurfaceRef s, const float *dffn) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP, dffn + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnBwdW13t: [1, HIDDEN, 1, 3*SEQ+2*DIM] fp16
#define FFN_BWD_W13T_SP (3*SEQ + 2*DIM)
static void stage_ffn_bwd_w13t_weights(IOSurfaceRef s, const float *W1, const float *W3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 3*SEQ,       W1 + d*DIM, DIM);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 3*SEQ + DIM, W3 + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w13t_acts(IOSurfaceRef s, const float *h1, const float *h3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + SEQ,   h1 + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ, h3 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// wotBwd: [1, DIM, 1, SEQ+Q_DIM] fp16 — Wo is [DIM, Q_DIM], matmul gives Wo^T @ dy
#define WOT_BWD_SP (SEQ + Q_DIM)
static void stage_wot_bwd_weights(IOSurfaceRef s, const float *Wo) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP + SEQ, Wo + d*Q_DIM, Q_DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_wot_bwd_acts(IOSurfaceRef s, const float *dy) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP, dy + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// qBwd: [1, Q_DIM, 1, SEQ+DIM] fp16 — Wq is [Q_DIM, DIM], matmul gives Wq^T @ dq
#define Q_BWD_SP (SEQ + DIM)
static void stage_q_bwd_weights(IOSurfaceRef s, const float *Wq) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP + SEQ, Wq + d*DIM, DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_q_bwd_acts(IOSurfaceRef s, const float *dq) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP, dq + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// kvBwd: [1, KV_DIM, 1, 2*SEQ+2*DIM] fp16 — dk @ Wk + dv @ Wv → dx_kv
#define KV_BWD_SP (2*SEQ + 2*DIM)
static void stage_kv_bwd_weights(IOSurfaceRef s, const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ,       Wk + d*DIM, DIM);
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ + DIM, Wv + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_kv_bwd_acts(IOSurfaceRef s, const float *dk, const float *dv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP,       dk + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*KV_BWD_SP + SEQ, dv + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// Batched write for sdpaBwd1 input: Q,K,V,da each Q_DIM channels, single lock
static void io_write_sdpa_bwd1_acts(IOSurfaceRef s, const float *Q, const float *K,
                                     const float *V, const float *da) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f32_f16(buf,                    Q,  Q_DIM * SEQ);
    cvt_f32_f16(buf + Q_DIM*SEQ,        K,  Q_DIM * SEQ);
    cvt_f32_f16(buf + 2*Q_DIM*SEQ,      V,  Q_DIM * SEQ);
    cvt_f32_f16(buf + 3*Q_DIM*SEQ,      da, Q_DIM * SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// Batched write for sdpaBwd2 input: Q and K_tiled after the score channels, single lock
static void io_write_sdpa_bwd2_qk(IOSurfaceRef s, int score_off, const float *Q, const float *K) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f32_f16(buf + score_off*SEQ,         Q, Q_DIM * SEQ);
    cvt_f32_f16(buf + (score_off+Q_DIM)*SEQ, K, Q_DIM * SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// Batched read for sdpaBwd2 output: dQ[Q_DIM] and dK_full[Q_DIM], single lock
static void io_read_sdpa_bwd2_outputs(IOSurfaceRef s, float *dq, float *dk_full) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f16_f32(dq,      buf,                Q_DIM * SEQ);
    cvt_f16_f32(dk_full, buf + Q_DIM*SEQ,    Q_DIM * SEQ);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// Free per-layer surfaces and requests
static void free_per_layer(PerLayerSurfaces *pls, PerLayerRequests *plr) {
    for (int L = 0; L < NLAYERS; L++) {
        CFRelease(pls[L].sdpaFwd_in); CFRelease(pls[L].woFwd_in); CFRelease(pls[L].ffnFused_in);
        CFRelease(pls[L].ffnBwdW2t_in); CFRelease(pls[L].ffnBwdW13t_in);
        CFRelease(pls[L].wotBwd_in); CFRelease(pls[L].qBwd_in); CFRelease(pls[L].kvBwd_in);
        CFRelease(plr[L].sdpaFwd); CFRelease(plr[L].woFwd); CFRelease(plr[L].ffnFused);
        CFRelease(plr[L].ffnBwdW2t); CFRelease(plr[L].ffnBwdW13t);
        CFRelease(plr[L].wotBwd); CFRelease(plr[L].qBwd); CFRelease(plr[L].kvBwd);
    }
}

// GQA helpers: tile KV from KV_HEADS to HEADS, and reduce HEADS to KV_HEADS
// tile_kv: input [KV_DIM, SEQ], output [Q_DIM, SEQ]
// Each KV head is duplicated GQA_RATIO times
static void gqa_tile_kv(float *out, const float *in, int seq) {
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            memcpy(out + q_head * HD * seq, in + kv * HD * seq, HD * seq * sizeof(float));
        }
    }
}
// reduce_kv: input [Q_DIM, SEQ], output [KV_DIM, SEQ]
// Sum contributions from Q heads sharing each KV head
static void gqa_reduce_kv(float *out, const float *in, int seq) {
    memset(out, 0, KV_DIM * seq * sizeof(float));
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            const float *src = in + q_head * HD * seq;
            float *dst = out + kv * HD * seq;
            for (int i = 0; i < HD * seq; i++)
                dst[i] += src[i];
        }
    }
}
