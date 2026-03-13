// train.m — Dynamic weight ANE training (model-agnostic GQA support)
// Model selected at compile time via: make MODEL=qwen3_06b (or stories110m)
// Compile kernels ONCE at startup, update weights via IOSurface every step.
#ifdef SEQ_OVERRIDE
#undef SEQ
#define SEQ SEQ_OVERRIDE
#endif

#ifdef ACCUM_STEPS_OVERRIDE
#undef ACCUM_STEPS
#define ACCUM_STEPS ACCUM_STEPS_OVERRIDE
#endif

#include "mil_dynamic.h"
#include "cpu_ops.h"

#define CKPT_PATH_DEFAULT "ane_stories110M_dyn_ckpt.bin"
#define MODEL_PATH_DEFAULT "../../../assets/models/stories110M.bin"
#define DATA_PATH_DEFAULT "../tinystories_data00.bin"
// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    Kern *ffnBwdW2t;   // dffn @ W2^T → dsilu_raw (DIM → HIDDEN)
    Kern *ffnBwdW13t;  // dsilu + SiLU bwd + dh1@W1^T + dh3@W3^T
    Kern *wotBwd;      // dx2 @ Wo → da (DIM → Q_DIM)
    Kern *sdpaBwd1;    // Q,K,V,da → dV_full,probs,dp (weight-free, has mask)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK_full (weight-free)
    Kern *qBwd;        // dq @ Wq → dx_q (Q_DIM → DIM)
    Kern *kvBwd;       // dk@Wk + dv@Wv → dx_kv (KV_DIM → DIM)
} DynLayerKernels;

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM;

    // SDPA forward (no Wo): [1, DIM, 1, SDPA_FWD_SP] → [1, sdpa_out_ch, 1, SEQ]
    printf("  Compiling sdpaFwd (GQA)...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!dk->sdpaFwd) return false;

    // Wo forward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling woFwd...\n");
    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!dk->woFwd) return false;

    // Fused FFN: [1, DIM, 1, FFN_FUSED_SP] → [1, DIM+3*HIDDEN, 1, SEQ]
    printf("  Compiling ffnFused...\n");
    int ffn_fused_och = DIM + 3*HIDDEN;
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!dk->ffnFused) return false;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] → [1, HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
    if (!dk->ffnBwdW2t) return false;

    // FFN backward fused tail: [1, HIDDEN, 1, 3*SEQ+2*DIM] → [1, DIM+2*HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, (DIM + 2*HIDDEN)*SEQ*2);
    if (!dk->ffnBwdW13t) return false;

    // Wo^T backward: [1, DIM, 1, SEQ+Q_DIM] → [1, Q_DIM, 1, SEQ]
    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
    if (!dk->wotBwd) return false;

    // SDPA bwd1 (weight-free, has mask): [1, 4*Q_DIM, 1, SEQ] → [1, Q_DIM+2*SCORE_CH, 1, SEQ]
    printf("  Compiling sdpaBwd1 (GQA)...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*Q_DIM*SEQ*2, (Q_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    // SDPA bwd2 (weight-free): [1, 2*SCORE_CH+2*Q_DIM, 1, SEQ] → [1, 2*Q_DIM, 1, SEQ]
    printf("  Compiling sdpaBwd2 (GQA)...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*Q_DIM)*SEQ*2, 2*Q_DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    // Q backward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qBwd...\n");
    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        Q_DIM*Q_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qBwd) return false;

    // KV backward: [1, KV_DIM, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling kvBwd...\n");
    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) return false;

    return true;
}

static bool compile_dynamic_aux_kernels(int compact_vocab,
                                        Kern **rmsBwd,
                                        Kern **rmsFinalBwd,
                                        Kern **clsFwd,
                                        Kern **clsBwd,
                                        Kern **softmax) {
    *rmsBwd = compile_kern_mil_w(gen_rmsnorm_bwd_dynamic(), @{},
        DIM*RMS_BWD_SP*2, DIM*SEQ*2);
    if (!*rmsBwd) {
        return false;
    }

    *rmsFinalBwd = compile_kern_mil_w(gen_rmsnorm_bwd_chan_dynamic(), @{},
        3*DIM*SEQ*2, DIM*SEQ*2);
    if (!*rmsFinalBwd) {
        free_kern(*rmsBwd);
        return false;
    }

    *clsFwd = compile_kern_mil_w(gen_classifier_fwd_dynamic(compact_vocab), @{},
        DIM*(SEQ + compact_vocab)*2, compact_vocab*SEQ*2);
    if (!*clsFwd) {
        free_kern(*rmsBwd);
        free_kern(*rmsFinalBwd);
        return false;
    }

    *clsBwd = compile_kern_mil_w(gen_classifier_bwd_dynamic(compact_vocab), @{},
        compact_vocab*(SEQ + DIM)*2, DIM*SEQ*2);
    if (!*clsBwd) {
        free_kern(*rmsBwd);
        free_kern(*rmsFinalBwd);
        free_kern(*clsFwd);
        return false;
    }

    *softmax = compile_kern_mil_w(gen_softmax_dynamic(compact_vocab), @{},
        compact_vocab*SEQ*2, compact_vocab*SEQ*2);
    if (!*softmax) {
        free_kern(*rmsBwd);
        free_kern(*rmsFinalBwd);
        free_kern(*clsFwd);
        free_kern(*clsBwd);
        return false;
    }

    return true;
}

// ===== Checkpoint =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 4;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WK_SZ,f); fwrite(la[L].Wk.v,4,WK_SZ,f);
        fwrite(la[L].Wv.m,4,WV_SZ,f); fwrite(la[L].Wv.v,4,WV_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *ct, double *cw, int *cs, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 4) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WK_SZ,f); fread(la[L].Wk.v,4,WK_SZ,f);
        fread(la[L].Wv.m,4,WV_SZ,f); fread(la[L].Wv.v,4,WV_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float max_lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.95f, adam_eps=1e-8f, wd=0.1f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 100;
        float grad_clip = 1.0f;
        float loss_scale = 256.0f;
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);
        float min_lr_frac = 0.1f;
        bool no_compact = false;

        bool do_resume = false, from_scratch = false;
        const char *model_path = MODEL_PATH_DEFAULT;
        const char *data_path = DATA_PATH_DEFAULT;
        const char *ckpt_path = CKPT_PATH_DEFAULT;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
            else if (strcmp(argv[i], "--model") == 0 && i+1<argc) model_path = argv[++i];
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--ckpt") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--no-compact") == 0) no_compact = true;
        }
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_train=0, cum_wall=0; int cum_steps=0;
        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(ckpt_path, &start_step, &total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Dynamic Training: %s (%d layers, GQA %d/%d heads) ===\n",
                   MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
            printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
                   DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
            double xformer_m = (double)NLAYERS*(WQ_SZ + WK_SZ + WV_SZ + (double)WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2.0*DIM) / 1e6;
            double embed_m = (double)VOCAB*DIM / 1e6;
            printf("Params: %.1fM (transformer %.1fM + embed %.1fM)\n", xformer_m+embed_m, xformer_m, embed_m);
            printf("Kernels: 15 compiled (base dynamic stack with fused FFN bwd tail + RMSNorm bwd/final-bwd + classifier fwd/bwd + softmax)\n");
            printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);
            // FLOPs estimate: 6*N*B*T for transformer (forward+backward ≈ 3x forward)
            double fwd_flops = 2.0*NLAYERS*(4.0*WQ_SZ + 2.0*W1_SZ + W2_SZ + W3_SZ) * SEQ;
            double total_flops = 3.0 * fwd_flops;  // fwd + bwd ≈ 3x fwd
            printf("FLOPs/step: fwd=%.1fM bwd_dx=%.1fM bwd_dW=%.1fM sdpa_bwd=0.0M total=%.1fM\n",
                   fwd_flops/1e6, fwd_flops/1e6, fwd_flops/1e6, total_flops/1e6);
            printf("ANE FLOPs/step: %.1fM\n", total_flops/1e6);
            if (from_scratch || !load_pretrained(lw, rms_final, embed, model_path)) {
                if (from_scratch) printf("  Training from scratch (random init)\n");
                else printf("  Pretrained load failed, using random init\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_qd=1.0f/sqrtf(Q_DIM), scale_h=1.0f/sqrtf(HIDDEN);
                float res_scale = 1.0f/sqrtf(2.0f*NLAYERS);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++) lw[L].Wq[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WK_SZ;i++) lw[L].Wk[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WV_SZ;i++) lw[L].Wv[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_qd*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            }
        }

        // Precompute transposed weights for forward/backward kernels
        // Forward: sdpaFwd needs Wq^T[Q_DIM,DIM], Wk^T[KV_DIM,DIM], Wv^T[KV_DIM,DIM]
        //          woFwd needs Wo^T[DIM,Q_DIM]
        // Backward uses original (non-transposed) weights
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WK_SZ*4);
            Wvt_buf[L]=(float*)malloc(WV_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W2t_buf[L]=(float*)malloc(W2_SZ*4);
            W3t_buf[L]=(float*)malloc(W3_SZ*4);
            // Wq is [Q_DIM, DIM] → Wq^T is [DIM, Q_DIM] (staged as [DIM channels, Q_DIM spatial])
            transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
            // Wk is [KV_DIM, DIM] → Wk^T is [DIM, KV_DIM]
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            // Wv is [KV_DIM, DIM] → Wv^T is [DIM, KV_DIM]
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            // Wo is [DIM, Q_DIM] → Wo^T is [Q_DIM, DIM]
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // mmap token data
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Vocab compaction (or identity map when disabled for parity runs)
        VocabMap vm;
        if (no_compact) {
            vm.compact_vocab = VOCAB;
            vm.full_to_compact = (int*)malloc(VOCAB * sizeof(int));
            vm.compact_to_full = (int*)malloc(VOCAB * sizeof(int));
            for (int v = 0; v < VOCAB; v++) {
                vm.full_to_compact[v] = v;
                vm.compact_to_full[v] = v;
            }
            printf("Vocab compaction: disabled (using full vocab %d)\n", VOCAB);
        } else {
            vm = vocab_map_build(token_data, n_tokens, VOCAB);
            printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, vm.compact_vocab, (float)VOCAB/vm.compact_vocab);
        }
        int CV = vm.compact_vocab;

        float *cembed = vocab_compact_embed(embed, &vm, DIM);
        float *cembed_t = (float*)malloc((size_t)DIM*CV*4);
        transpose_weight(cembed_t, cembed, CV, DIM);
        float *gcembed = (float*)calloc((size_t)CV*DIM, 4);
        AdamState acembed = adam_alloc((size_t)CV*DIM);

        // ===== Compile all kernels ONCE =====
        printf("Compiling 15 dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        Kern *rmsBwdKern = NULL, *rmsFinalBwdKern = NULL;
        Kern *clsFwdKern = NULL, *clsBwdKern = NULL, *softmaxKern = NULL;
        if (!compile_dynamic_aux_kernels(CV, &rmsBwdKern, &rmsFinalBwdKern,
                                         &clsFwdKern, &clsBwdKern, &softmaxKern)) {
            printf("Aux kernel compilation failed!\n");
            return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 15 kernels in %.0fms (shared across all %d layers)\n", compile_ms, NLAYERS);

        // Allocate per-layer IOSurfaces + requests
        printf("Allocating per-layer IOSurfaces...\n");
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
            pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
            pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
            pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
            pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
            pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
            pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
            pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);

            plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
            plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
            plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
            plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
            plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
            plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
            plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
            plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
        }

        // Stage weights into per-layer surfaces
        for (int L = 0; L < NLAYERS; L++) {
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
            stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
            stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
            stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
            stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
            stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
        }
        printf("Per-layer weight staging complete\n\n");

        // Pre-allocated capture buffers for async dW cblas (one set per layer)
        typedef struct {
            // FFN dW capture
            float *dffn, *silu_out, *dh1, *dh3, *x2norm;
            // Wo dW capture
            float *dx2_scaled, *attn_out;
            // QKV dW capture
            float *dq, *dk, *dv, *xnorm;
        } DwCapture;
        DwCapture dw_capt[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            dw_capt[L].dffn      = (float*)malloc(SEQ*DIM*4);
            dw_capt[L].silu_out  = (float*)malloc(SEQ*HIDDEN*4);
            dw_capt[L].dh1       = (float*)malloc(SEQ*HIDDEN*4);
            dw_capt[L].dh3       = (float*)malloc(SEQ*HIDDEN*4);
            dw_capt[L].x2norm    = (float*)malloc(SEQ*DIM*4);
            dw_capt[L].dx2_scaled= (float*)malloc(SEQ*DIM*4);
            dw_capt[L].attn_out  = (float*)malloc(SEQ*Q_DIM*4);
            dw_capt[L].dq        = (float*)malloc(SEQ*Q_DIM*4);
            dw_capt[L].dk        = (float*)malloc(SEQ*KV_DIM*4);
            dw_capt[L].dv        = (float*)malloc(SEQ*KV_DIM*4);
            dw_capt[L].xnorm     = (float*)malloc(SEQ*DIM*4);
        }

        // Gradient + work buffers (GQA: Q has Q_DIM, K/V have KV_DIM)
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*Q_DIM*4);     // Q_DIM for Q grads
        float *dk_buf = (float*)malloc(SEQ*KV_DIM*4); // KV_DIM for K grads
        float *dv = (float*)malloc(SEQ*KV_DIM*4);     // KV_DIM for V grads
        float *da_buf = (float*)malloc(SEQ*Q_DIM*4);  // Q_DIM for attn grads
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *logits_cf = (float*)malloc(SEQ*CV*4);   // channel-first [CV, SEQ]
        float *logits_sv = (float*)malloc(SEQ*CV*4);   // row-major [SEQ, CV] for CE
        float *dlogits_sv = (float*)malloc(SEQ*CV*4);   // row-major [SEQ, CV] for CE grad
        float *dlogits_cf = (float*)malloc(SEQ*CV*4);   // channel-first [CV, SEQ] for ANE cls bwd
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        // GQA tile/reduce buffers
        float *k_tiled = (float*)malloc(SEQ*Q_DIM*4);  // KV_DIM → Q_DIM
        float *v_tiled = (float*)malloc(SEQ*Q_DIM*4);
        float *dq_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2
        float *dk_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2 (needs reduce)
        float *dv_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd1 (needs reduce)
        float *dx_kv = (float*)malloc(SEQ*DIM*4);    // reused each layer for kvBwd + rms1_bwd scratch

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t1, t_step = mach_absolute_time();

            // Sample data
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens_raw = token_data + pos + 1;

            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];

            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0, t_cls=0, t_dw_copy=0;

            // ===== FORWARD (28 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // RMSNorm1 on CPU.
                t0 = mach_absolute_time();
                rmsnorm(ac->xnorm, x_cur, lw[L].rms_att, DIM, SEQ);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Wait for any pending dW cblas
                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);

                // SDPA forward (ANE): xnorm + Wq,Wk,Wv → attn_out[Q_DIM], Q_rope[Q_DIM], K_rope[KV_DIM], V[KV_DIM]
                t0 = mach_absolute_time();
                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, ac->xnorm);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.sdpaFwd, plr[L].sdpaFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read SDPA output: [1, Q_DIM+Q_DIM+KV_DIM+KV_DIM, 1, SEQ] fp16
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                int off = 0;
                cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                cvt_f16_f32(ac->Q,        fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
                cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ);
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Wo forward (ANE): attn_out[Q_DIM] → o_out[DIM]
                t0 = mach_absolute_time();
                write_wo_fwd_acts(pls[L].woFwd_in, ac->attn_out);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.woFwd, plr[L].woFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.woFwd->ioOut, ac->o_out, DIM, SEQ);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // CPU residual, then RMSNorm2 on CPU.
                t0 = mach_absolute_time();
                vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                t_rms += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Fused FFN (ANE)
                t0 = mach_absolute_time();
                write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnFused, plr[L].ffnFused);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read fused output: [1, DIM+3*HIDDEN, 1, SEQ]
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnFused->ioOut);
                off = 0;
                cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
                IOSurfaceUnlock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
            }

            // Final RMSNorm on CPU, then classifier + softmax on ANE, CE/NLL on CPU.
            t0 = mach_absolute_time();
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            t_rms += tb_ms(mach_absolute_time() - t0);

            t0 = mach_absolute_time();
            // Classifier forward on CPU (compact vocab): logits_cf[CV,SEQ] = cembed[CV,DIM] @ x_final[DIM,SEQ].
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits_cf, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            t0 = mach_absolute_time();
            transpose_vs(logits_sv, logits_cf, CV, SEQ);
            float loss = cross_entropy_loss_rowmajor(dlogits_sv, logits_sv, ctargets, CV, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);
            last_loss = loss;

            // ===== BACKWARD =====
            vDSP_vsmul(dlogits_sv, 1, &loss_scale, dlogits_sv, 1, (vDSP_Length)(SEQ*CV));
            transpose_vs(dlogits_cf, dlogits_sv, SEQ, CV);

            // Classifier backward on CPU: dy[DIM,SEQ] = cembed^T[DIM,CV] @ dlogits_cf[CV,SEQ].
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits_cf, SEQ, 0.0f, dy, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            // dEmbed async: gcembed[CV,DIM] += dlogits_sv^T @ x_final^T
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, dlogits_sv, CV, x_final, SEQ, 1.0f, gcembed, DIM);
            });

            // Final RMSNorm backward on ANE via the channel-concatenated kernel.
            t0 = mach_absolute_time();
            io_write_rmsnorm_bwd_chan(rmsFinalBwdKern->ioIn, dy, x_cur, rms_final);
            t_io_bwd += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            rmsnorm_dw_only(grms_final, dy, x_cur, rms_final, DIM, SEQ);
            t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            ane_eval(rmsFinalBwdKern);
            t_ane_bwd += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            io_read_dyn(rmsFinalBwdKern->ioOut, dy, DIM, SEQ);
            t_io_bwd += tb_ms(mach_absolute_time() - t0);

            // ===== BACKWARD (28 layers, reverse) =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));

                // FFN backward: dffn @ W2^T → dsilu_raw
                t0 = mach_absolute_time();
                write_ffn_bwd_w2t_acts(pls[L].ffnBwdW2t_in, dffn);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Fused FFN tail: dsilu + SiLU derivative + dh1@W1^T + dh3@W3^T.
                t0 = mach_absolute_time();
                io_copy_rect(pls[L].ffnBwdW13t_in, 0, FFN_BWD_W13T_SP,
                             dk.ffnBwdW2t->ioOut, 0, SEQ, HIDDEN, SEQ);
                write_ffn_bwd_w13t_acts(pls[L].ffnBwdW13t_in, ac->h1, ac->h3);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnBwdW13t, plr[L].ffnBwdW13t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_fp16(dk.ffnBwdW13t->ioOut, dx_ffn, 0, DIM, SEQ);
                io_read_fp16(dk.ffnBwdW13t->ioOut, dh1, DIM, HIDDEN, SEQ);
                io_read_fp16(dk.ffnBwdW13t->ioOut, dh3, DIM + HIDDEN, HIDDEN, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dW FFN async (pre-allocated capture buffers)
                t0 = mach_absolute_time();
                {
                    DwCapture *dc = &dw_capt[L];
                    memcpy(dc->dffn, dffn, SEQ*DIM*4);
                    memcpy(dc->silu_out, ac->silu_out, SEQ*HIDDEN*4);
                    memcpy(dc->dh1, dh1, SEQ*HIDDEN*4);
                    memcpy(dc->dh3, dh3, SEQ*HIDDEN*4);
                    memcpy(dc->x2norm, ac->x2norm, SEQ*DIM*4);
                }
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                {
                    DwCapture *dc = &dw_capt[L];
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                    1.0f, dc->dffn, SEQ, dc->silu_out, SEQ, 1.0f, gr->W2, HIDDEN);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, dc->dh1, SEQ, dc->x2norm, SEQ, 1.0f, gr->W1, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, dc->dh3, SEQ, dc->x2norm, SEQ, 1.0f, gr->W3, DIM);
                    });
                }

                // RMSNorm2 backward: dx on ANE, dw on CPU.
                t0 = mach_absolute_time();
                io_write_rmsnorm_bwd(rmsBwdKern->ioIn, dx_ffn, ac->x2, lw[L].rms_ffn);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(rmsBwdKern);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(rmsBwdKern->ioOut, dx2, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                rmsnorm_dw_only(gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // Wo^T backward (ANE): alpha*dx2 @ Wo → da[Q_DIM]
                float *dx2_scaled = dw_capt[L].dx2_scaled;
                vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
                t0 = mach_absolute_time();
                write_wot_bwd_acts(pls[L].wotBwd_in, dx2_scaled);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.wotBwd, plr[L].wotBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.wotBwd->ioOut, da_buf, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dWo async: gr->Wo[DIM,Q_DIM] += dx2_scaled[DIM,SEQ] @ attn_out^T[SEQ,Q_DIM]
                t0 = mach_absolute_time();
                {
                    DwCapture *dc = &dw_capt[L];
                    // dx2_scaled already in dc->dx2_scaled
                    memcpy(dc->attn_out, ac->attn_out, SEQ*Q_DIM*4);
                }
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                {
                    DwCapture *dc = &dw_capt[L];
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                                    1.0f, dc->dx2_scaled, SEQ, dc->attn_out, SEQ, 1.0f, gr->Wo, Q_DIM);
                    });
                }

                // GQA: tile K,V from KV_DIM → Q_DIM for SDPA backward
                t0 = mach_absolute_time();
                gqa_tile_kv(k_tiled, ac->K, SEQ);
                gqa_tile_kv(v_tiled, ac->V, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 1: Q[Q_DIM],K_tiled[Q_DIM],V_tiled[Q_DIM],da[Q_DIM] → dV_full[Q_DIM],probs,dp
                t0 = mach_absolute_time();
                io_write_sdpa_bwd1_acts(dk.sdpaBwd1->ioIn, ac->Q, k_tiled, v_tiled, da_buf);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd1);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 2: probs,dp,Q[Q_DIM],K_tiled[Q_DIM] → dQ[Q_DIM],dK_full[Q_DIM]
                t0 = mach_absolute_time();
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, Q_DIM, 2*SCORE_CH, SEQ);
                io_write_sdpa_bwd2_qk(dk.sdpaBwd2->ioIn, 2*SCORE_CH, ac->Q, k_tiled);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd2);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read SDPA backward outputs (batched: 2 reads from bwd2, 1 from bwd1)
                t0 = mach_absolute_time();
                io_read_sdpa_bwd2_outputs(dk.sdpaBwd2->ioOut, dq_full, dk_full);
                io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // GQA: reduce dK, dV from Q_DIM (HEADS) → KV_DIM (KV_HEADS)
                gqa_reduce_kv(dk_buf, dk_full, SEQ);
                gqa_reduce_kv(dv, dv_full, SEQ);
                // dQ stays at Q_DIM — no reduction needed
                memcpy(dq, dq_full, SEQ*Q_DIM*4);

                // RoPE backward on dQ[Q_DIM] and dK[KV_DIM]
                rope_backward_inplace(dq, SEQ, Q_DIM, HD);
                rope_backward_inplace(dk_buf, SEQ, KV_DIM, HD);

                if (L == 0 && step % 10 == 0) {
                    float dqmx, dkmx, dvmx;
                    vDSP_maxmgv(dq, 1, &dqmx, (vDSP_Length)(SEQ*Q_DIM));
                    vDSP_maxmgv(dk_buf, 1, &dkmx, (vDSP_Length)(SEQ*KV_DIM));
                    vDSP_maxmgv(dv, 1, &dvmx, (vDSP_Length)(SEQ*KV_DIM));
                    printf("    L0 sdpa_bwd: |dq|=%.6f |dk|=%.6f |dv|=%.6f\n", dqmx, dkmx, dvmx);
                }

                // dWq/dWk/dWv async (pre-allocated capture buffers)
                // dWq[Q_DIM,DIM] += dq[Q_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWk[KV_DIM,DIM] += dk[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWv[KV_DIM,DIM] += dv[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                t0 = mach_absolute_time();
                {
                    DwCapture *dc = &dw_capt[L];
                    memcpy(dc->dq, dq, SEQ*Q_DIM*4);
                    memcpy(dc->dk, dk_buf, SEQ*KV_DIM*4);
                    memcpy(dc->dv, dv, SEQ*KV_DIM*4);
                    memcpy(dc->xnorm, ac->xnorm, SEQ*DIM*4);
                }
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                {
                    DwCapture *dc = &dw_capt[L];
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q_DIM, DIM, SEQ,
                                    1.0f, dc->dq, SEQ, dc->xnorm, SEQ, 1.0f, gr->Wq, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                    1.0f, dc->dk, SEQ, dc->xnorm, SEQ, 1.0f, gr->Wk, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                    1.0f, dc->dv, SEQ, dc->xnorm, SEQ, 1.0f, gr->Wv, DIM);
                    });
                }

                // Q backward (ANE): dq[Q_DIM] @ Wq → dx_q[DIM]
                t0 = mach_absolute_time();
                write_q_bwd_acts(pls[L].qBwd_in, dq);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.qBwd, plr[L].qBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // KV backward (ANE): dk[KV_DIM]@Wk + dv[KV_DIM]@Wv → dx_kv[DIM]
                t0 = mach_absolute_time();
                write_kv_bwd_acts(pls[L].kvBwd_in, dk_buf, dv);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.kvBwd, plr[L].kvBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dx_attn = dx_q + dx_kv
                for(int i=0; i<SEQ*DIM; i++) dx_attn[i] += dx_kv[i];

                // RMSNorm1 backward: dx on ANE, dw on CPU.
                t0 = mach_absolute_time();
                io_write_rmsnorm_bwd(rmsBwdKern->ioIn, dx_attn, ac->layer_in, lw[L].rms_att);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(rmsBwdKern);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(rmsBwdKern->ioOut, dx_kv, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                rmsnorm_dw_only(gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_kv[i] + dx2[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

            // Embedding backward
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            if (step % 10 == 0 || step == start_step) {
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f silu=%.1f rms_bwd=%.1f cls=%.1f cblas_wait=%.1f dw_copy=%.1f\n",
                       t_ane_fwd, t_io_fwd, t_rms, t_ane_bwd, t_io_bwd, t_silu, t_rms_bwd, t_cls, t_cblas_wait, t_dw_copy);
                float xmx, xmn;
                vDSP_maxv(x_cur,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(x_cur,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d pos=%zu loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, pos, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            // Adam update every accum_steps
            if ((step+1) % accum_steps == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / (accum_steps * loss_scale);
                adam_t++;

                // Scale gradients
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for(size_t i=0;i<WQ_SZ;i++) g->Wq[i]*=gsc;
                    for(size_t i=0;i<WK_SZ;i++) g->Wk[i]*=gsc;
                    for(size_t i=0;i<WV_SZ;i++) g->Wv[i]*=gsc;
                    for(size_t i=0;i<WO_SZ;i++) g->Wo[i]*=gsc;
                    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                    for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                }
                for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
                vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;

                // Global gradient norm
                float grad_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    float s;
                    vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                }
                { float s;
                  vDSP_dotpr(grms_final,1,grms_final,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                  vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); grad_norm_sq+=s;
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step+1) % 10 == 0) {
                    float attn_sq=0, ffn_sq=0, embed_sq=0;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L]; float s;
                        vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); attn_sq+=s;
                        vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); ffn_sq+=s;
                    }
                    { float s;
                      vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); embed_sq=s;
                    }
                    printf("  grad_norm=%.4f  attn=%.4f ffn=%.4f embed=%.4f\n",
                           grad_norm, sqrtf(attn_sq), sqrtf(ffn_sq), sqrtf(embed_sq));
                }

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L];
                        vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WK_SZ);
                        vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WV_SZ);
                        vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
                        vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
                        vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
                        vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
                        vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
                    }
                    vDSP_vsmul(grms_final,1,&clip_scale,grms_final,1,(vDSP_Length)DIM);
                    vDSP_vsmul(gembed,1,&clip_scale,gembed,1,(vDSP_Length)(VOCAB*DIM));
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Adam update
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);

                    // Update transposed weight buffers
                    transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
                    transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                    transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                    transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
                    transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                    transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                    transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);

                    // Re-stage weights
                    stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
                    stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
                    stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
                    stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
                    stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
                    stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
                    stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
                    stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                free(cembed);
                cembed = vocab_compact_embed(embed, &vm, DIM);
                transpose_weight(cembed_t, cembed, CV, DIM);

                // Zero grads
                for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
                memset(grms_final, 0, DIM*4);
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
                memset(gcembed, 0, (size_t)CV*DIM*4);

                // Checkpoint — only save on best loss
                if ((step+1) % 100 == 0 && last_loss < best_loss) {
                    best_loss = last_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    save_checkpoint(ckpt_path, step+1, total_steps, lr, last_loss,
                        total_train_ms+cum_train, wall+cum_wall, total_steps_done+cum_steps, adam_t,
                        lw, la, rms_final, &arms_final, embed, &aembed);
                    printf("  [ckpt saved, best_loss=%.4f]\n", best_loss);
                }
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms/total_steps_done);
        printf("Wall time:    %.1fs\n", (wall+cum_wall)/1000);

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free_per_layer(pls, plr);
        free_kern(dk.sdpaFwd); free_kern(dk.woFwd); free_kern(dk.ffnFused);
        free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.wotBwd);
        free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2);
        free_kern(dk.qBwd); free_kern(dk.kvBwd);
        free_kern(rmsBwdKern); free_kern(rmsFinalBwdKern);
        free_kern(clsFwdKern); free_kern(clsBwdKern); free_kern(softmaxKern);
        free(dy); free(dffn); free(dx_ffn); free(dx2); free(dx_attn);
        free(da_buf); free(k_tiled); free(v_tiled);
        free(dq_full); free(dk_full); free(dv_full);
        free(dq); free(dk_buf); free(dv);
        free(x_cur); free(x_final);
        free(logits_cf); free(logits_sv); free(dlogits_sv); free(dlogits_cf);
        free(dh1); free(dh3); free(dx_kv);
        free(cembed); free(gcembed); adam_free(&acembed);
        free(cembed_t);
        free(vm.full_to_compact); free(vm.compact_to_full);
        munmap(token_data, data_len); close(data_fd);
    }
    return 0;
}
