package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/maderix/ANE/ane/stories"
)

func main() {
	var (
		modelPath = flag.String("model", "/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin", "path to stories110M.bin")
		dataPath  = flag.String("data", "/Volumes/tmc/go/src/github.com/maderix/ANE/training/tinystories_data00.bin", "path to token data")
		ckptPath  = flag.String("ckpt", "/Volumes/tmc/go/src/github.com/maderix/ANE/training/ane_stories110M_ckpt.go.bin", "checkpoint path")
		resume    = flag.Bool("resume", false, "resume from checkpoint")
		steps     = flag.Int("steps", 100, "training steps")
		seq       = flag.Int("seq", stories.SeqDefault, "sequence length")
		seed      = flag.Int64("seed", time.Now().UnixNano(), "rng seed")
		lr        = flag.Float64("lr", 3e-4, "learning rate")
		jsonOut   = flag.Bool("json", true, "emit JSON telemetry to stderr")
		saveEvery = flag.Int("save-every", 10, "checkpoint every N steps")
		accum     = flag.Int("accum-steps", 10, "batch telemetry window")
		saveFinal = flag.Bool("save-final", false, "write final checkpoint at end")
	)
	flag.Parse()

	toks, err := loadTokens(*dataPath)
	if err != nil {
		fatalf("load tokens: %v", err)
	}
	if len(toks) < *seq+1 {
		fatalf("not enough tokens: %d", len(toks))
	}

	mw := stories.NewModelWeights(stories.Vocab)
	opt := stories.NewOptimState(stories.Vocab)
	meta := stories.TrainMeta{TotalSteps: *steps, LR: float32(*lr)}

	if *resume {
		m, err := stories.LoadCheckpointV2(*ckptPath, mw, opt)
		if err != nil {
			fatalf("resume load: %v", err)
		}
		meta = m
		if *steps > 0 {
			meta.TotalSteps = meta.Step + *steps
		}
		if *lr > 0 {
			meta.LR = float32(*lr)
		}
		fmt.Printf("[RESUMED step %d, loss=%.4f]\n", meta.Step, meta.Loss)
	} else {
		cfg, err := preloadOrRandom(mw, *modelPath, *seed)
		if err != nil {
			fatalf("init model: %v", err)
		}
		fmt.Printf("=== ANE Training: Stories110M Go (CPU reference path) ===\n")
		fmt.Printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", stories.Dim, stories.Hidden, stories.Heads, *seq, stories.Vocab, stories.NLayers)
		fmt.Printf("Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n", cfg.Dim, cfg.HiddenDim, cfg.NLayers, cfg.NHeads, abs32(cfg.VocabSize), cfg.SeqLen)
		fmt.Printf("Accum %d steps per telemetry | Adam LR=%.1e b1=%.1f b2=%.3f\n", *accum, meta.LR, 0.9, 0.999)
	}

	rng := rand.New(rand.NewSource(*seed))
	vocab := len(mw.Embed) / stories.Dim
	x := make([]float32, stories.Dim**seq)
	xNorm := make([]float32, stories.Dim**seq)
	logits := make([]float32, vocab**seq)
	dLogits := make([]float32, vocab**seq)
	dy := make([]float32, stories.Dim**seq)
	dx := make([]float32, stories.Dim**seq)
	gRMS := make([]float32, stories.Dim)
	gEmbed := make([]float32, len(mw.Embed))

	start := time.Now()
	batchStart := time.Now()
	var (
		batchMs   float64
		batchCls  float64
		batchElem float64
		batchRMS  float64
	)
	for step := meta.Step; step < meta.TotalSteps; step++ {
		pos := rng.Intn(len(toks) - *seq - 1)
		input := toks[pos : pos+*seq]
		target := toks[pos+1 : pos+1+*seq]
		for i := range gRMS {
			gRMS[i] = 0
		}
		for i := range gEmbed {
			gEmbed[i] = 0
		}

		t0 := time.Now()
		tRMS0 := time.Now()
		stories.EmbedLookup(x, mw.Embed, input, stories.Dim, *seq)
		stories.RMSNorm(xNorm, x, mw.RMSFinal, stories.Dim, *seq)
		tRMS := ms(time.Since(tRMS0))

		tCls0 := time.Now()
		stories.MatMulVocabSeq(logits, mw.Embed, xNorm, vocab, stories.Dim, *seq)
		tCls := ms(time.Since(tCls0))

		tElem0 := time.Now()
		loss := stories.CrossEntropyLoss(dLogits, logits, target, vocab, *seq)
		stories.MatMulEmbedT(dy, mw.Embed, dLogits, vocab, stories.Dim, *seq)
		stories.MatMulGradEmbed(gEmbed, dLogits, xNorm, vocab, stories.Dim, *seq)
		stories.RMSNormBackward(dx, gRMS, dy, x, mw.RMSFinal, stories.Dim, *seq)
		stories.EmbedBackward(gEmbed, dx, input, stories.Dim, *seq)
		tElem := ms(time.Since(tElem0))

		meta.AdamT++
		stories.AdamUpdate(mw.RMSFinal, gRMS, &opt.RMSFinal, meta.AdamT, meta.LR, 0.9, 0.999, 1e-8)
		stories.AdamUpdate(mw.Embed, gEmbed, &opt.Embed, meta.AdamT, meta.LR, 0.9, 0.999, 1e-8)
		dt := time.Since(t0)
		stepMS := ms(dt)
		batchMs += stepMS
		batchCls += tCls
		batchElem += tElem
		batchRMS += tRMS

		meta.Step = step + 1
		meta.Loss = loss
		meta.CumTrain += ms(dt)
		meta.CumWall = ms(time.Since(start))
		meta.CumSteps++

		fmt.Printf("step %d    loss=%.6f step_ms=%.3f cls_ms=%.3f elem_ms=%.3f rms_ms=%.3f\n", step, loss, stepMS, tCls, tElem, tRMS)
		if *jsonOut {
			fmt.Fprintf(os.Stderr, "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,\"t_ane\":0.000,\"t_io\":0.000,\"t_cls\":%.3f,\"t_elem\":%.3f,\"t_rms\":%.3f,\"t_cblas_wait\":0.000,\"compiles\":0}\n", step, loss, tCls, tElem, tRMS)
		}

		if *saveEvery > 0 && meta.Step%*saveEvery == 0 {
			if err := stories.SaveCheckpointV2(*ckptPath, meta, mw, opt); err != nil {
				fatalf("save checkpoint: %v", err)
			}
		}
		if *accum > 0 && meta.Step%*accum == 0 {
			meta.CumBatches++
			fmt.Printf("  [batch %d: compile=0ms train=%.1fms (%.1fms/step) compiles=0]\n", *accum, batchMs, batchMs/float64(*accum))
			fmt.Printf("    ane=0.0 io=0.0 cls=%.1f elem=%.1f rms=%.1f cblas_wait=0.0 ms/step\n", batchCls/float64(*accum), batchElem/float64(*accum), batchRMS/float64(*accum))
			if *jsonOut {
				fmt.Fprintf(os.Stderr, "{\"type\":\"batch\",\"batch\":%d,\"compile_ms\":0.0,\"train_ms\":%.1f,\"ms_per_step\":%.1f}\n", *accum, batchMs, batchMs/float64(*accum))
			}
			batchStart = time.Now()
			_ = batchStart
			batchMs = 0
			batchCls = 0
			batchElem = 0
			batchRMS = 0
		}
	}

	if *saveFinal {
		if err := stories.SaveCheckpointV2(*ckptPath, meta, mw, opt); err != nil {
			fatalf("save final checkpoint: %v", err)
		}
		fmt.Printf("saved checkpoint: %s\n", *ckptPath)
	}
}

func preloadOrRandom(mw *stories.ModelWeights, modelPath string, seed int64) (stories.Llama2Config, error) {
	loaded, cfg, err := stories.LoadPretrained(modelPath)
	if err == nil {
		*mw = *loaded
		return cfg, nil
	}
	stories.RandomInit(mw, seed)
	return stories.Llama2Config{Dim: stories.Dim, HiddenDim: stories.Hidden, NLayers: stories.NLayers, NHeads: stories.Heads, VocabSize: stories.Vocab, SeqLen: stories.SeqDefault}, nil
}

func loadTokens(path string) ([]uint16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("odd file size %d", len(b))
	}
	t := make([]uint16, len(b)/2)
	for i := range t {
		t[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return t, nil
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
