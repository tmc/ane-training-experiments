//go:build darwin

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/maderix/ANE/ane/stories"
	"github.com/maderix/ANE/ane/storiesane"
)

const (
	defaultModel = "stories110M.bin"
	defaultData  = "tinystories_data00.bin"
)

func main() {
	var (
		modelPath   = flag.String("model", defaultModel, "path to stories .bin model")
		dataPath    = flag.String("data", defaultData, "path to TinyStories uint16 token data")
		steps       = flag.Int("steps", 100, "training steps")
		seq         = flag.Int("seq", stories.SeqDefault, "sequence length")
		accumSteps  = flag.Int("accum", 10, "optimizer accumulation steps")
		lr          = flag.Float64("lr", 3e-4, "peak learning rate")
		warmup      = flag.Int("warmup", 100, "warmup steps for cosine schedule")
		minLRFrac   = flag.Float64("min-lr-frac", 0.1, "minimum learning-rate fraction")
		adamB1      = flag.Float64("adam-b1", 0.9, "Adam beta1")
		adamB2      = flag.Float64("adam-b2", 0.95, "Adam beta2")
		adamEps     = flag.Float64("adam-eps", 1e-8, "Adam epsilon")
		weightDecay = flag.Float64("weight-decay", 0.1, "Adam weight decay")
		gradClip    = flag.Float64("grad-clip", 1.0, "global grad clip (0 disables)")
		gradTasks   = flag.Int("dw-concurrency", 3, "max concurrent CPU dW tasks")
		hybridBwd   = flag.Bool("hybrid-bwd", true, "enable ANE hybrid backward path")
		printEvery  = flag.Int("print-every", 10, "print every N steps")
	)
	flag.Parse()

	if *steps < 1 {
		fatalf("steps must be >= 1")
	}
	if *seq < 1 {
		fatalf("seq must be >= 1")
	}
	if *accumSteps < 1 {
		fatalf("accum must be >= 1")
	}
	if *gradTasks < 0 {
		fatalf("dw-concurrency must be >= 0")
	}
	if *lr <= 0 {
		fatalf("lr must be > 0")
	}
	if *warmup < 0 {
		fatalf("warmup must be >= 0")
	}
	if *minLRFrac < 0 || *minLRFrac > 1 {
		fatalf("min-lr-frac must be in [0,1]")
	}
	if *adamB1 <= 0 || *adamB1 >= 1 {
		fatalf("adam-b1 must be in (0,1)")
	}
	if *adamB2 <= 0 || *adamB2 >= 1 {
		fatalf("adam-b2 must be in (0,1)")
	}
	if *adamEps <= 0 {
		fatalf("adam-eps must be > 0")
	}
	if *weightDecay < 0 {
		fatalf("weight-decay must be >= 0")
	}
	if *gradClip < 0 {
		fatalf("grad-clip must be >= 0")
	}
	if *printEvery < 1 {
		fatalf("print-every must be >= 1")
	}
	if err := storiesane.ProbeDirectSequence(*seq); err != nil {
		fatalf("seq=%d unsupported for direct dynamic path: %v", *seq, err)
	}

	tokens, err := loadTokens(*dataPath)
	if err != nil {
		fatalf("load token data: %v", err)
	}
	if len(tokens) < *seq+1 {
		fatalf("token data too short: got %d want at least %d", len(tokens), *seq+1)
	}

	fmt.Printf("=== Go Dynamic Training (minimal) ===\n")
	fmt.Printf("seq=%d accum=%d lr=%g steps=%d\n", *seq, *accumSteps, *lr, *steps)

	wallStart := time.Now()
	compileStart := time.Now()
	engine, err := storiesane.Open(storiesane.Options{
		ModelPath:      *modelPath,
		Tokens:         tokens,
		Seq:            *seq,
		AccumSteps:     *accumSteps,
		LR:             float32(*lr),
		Seed:           42,
		AdamBeta1:      float32(*adamB1),
		AdamBeta2:      float32(*adamB2),
		AdamEps:        float32(*adamEps),
		WeightDecay:    float32(*weightDecay),
		GradClip:       float32(*gradClip),
		GradTaskLimit:  *gradTasks,
		UseANE:         true,
		HybridBackward: *hybridBwd,
	})
	if err != nil {
		fatalf("open engine: %v", err)
	}
	defer engine.Close()
	engine.Prepare()
	compileDur := time.Since(compileStart)
	status := engine.DynamicStatus()
	if !status.FullyDynamic() {
		fatalf("dynamic runtime validation failed: %s", status.String())
	}
	fmt.Printf("Compiled dynamic kernels in %.0fms\n", ms(compileDur))

	var totalTrainMS float64
	currLR := *lr
	for step := 0; step < *steps; step++ {
		if shouldApplyUpdate(step, *steps, *accumSteps) {
			currLR = scheduledLR(step, *steps, *warmup, *lr, *minLRFrac)
			if err := engine.SetLR(float32(currLR)); err != nil {
				fatalf("set lr at step %d: %v", step, err)
			}
		}
		res, err := engine.Step()
		if err != nil {
			fatalf("step %d: %v", step, err)
		}
		stepMS := ms(res.StepDuration)
		totalTrainMS += stepMS
		if step == 0 || step%*printEvery == 0 {
			fmt.Printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step\n", step, res.Loss, currLR, stepMS)
		}
	}

	totalMS := ms(time.Since(wallStart))
	compileMS := ms(compileDur)
	compilePct := 0.0
	if totalMS > 0 {
		compilePct = 100.0 * compileMS / totalMS
	}
	fmt.Printf("\n=== Efficiency Report ===\n")
	fmt.Printf("Total steps:  %d\n", *steps)
	fmt.Printf("Compile:      %.0fms (one-time, %.1f%%)\n", compileMS, compilePct)
	fmt.Printf("Train time:   %.0fms (%.1fms/step)\n", totalTrainMS, totalTrainMS/float64(*steps))
	fmt.Printf("Wall time:    %.1fs\n", totalMS/1000.0)
}

func loadTokens(path string) ([]uint16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("token file has odd byte length %d", len(b))
	}
	toks := make([]uint16, len(b)/2)
	for i := 0; i < len(toks); i++ {
		toks[i] = binary.LittleEndian.Uint16(b[2*i : 2*i+2])
	}
	return toks, nil
}

func ms(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func shouldApplyUpdate(step, totalSteps, accumSteps int) bool {
	return (step+1)%accumSteps == 0 || step == totalSteps-1
}

func scheduledLR(step, totalSteps, warmupSteps int, maxLR, minLRFrac float64) float64 {
	if step < warmupSteps && warmupSteps > 0 {
		return maxLR * float64(step+1) / float64(warmupSteps)
	}
	minLR := maxLR * minLRFrac
	if totalSteps <= warmupSteps {
		return minLR
	}
	decayRatio := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
	return minLR + 0.5*(1+math.Cos(math.Pi*decayRatio))*(maxLR-minLR)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
