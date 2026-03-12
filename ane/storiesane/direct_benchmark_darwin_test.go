//go:build darwin

package storiesane

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

func BenchmarkDirectGoANEDirect(b *testing.B) {
	for _, seq := range []int{stories.SeqDefault, 384} {
		seq := seq
		b.Run(fmt.Sprintf("final_head_seq_%d", seq), func(b *testing.B) {
			engine := openBenchmarkEngine(b, seq, 1<<30)
			finalHidden := make([]float32, stories.Dim*seq)
			targets := make([]uint16, seq)
			fillBenchmarkFloats(finalHidden, 0.015)
			for i := range targets {
				targets[i] = uint16((i * 997) % stories.Vocab)
			}
			engine.ensureOffload()
			if engine.off == nil {
				b.Skip("ANE offload unavailable")
			}
			if _, err := engine.runFinalHead(finalHidden, targets); err != nil {
				b.Fatalf("warmup final head: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := engine.runFinalHead(finalHidden, targets); err != nil {
					b.Fatalf("runFinalHead: %v", err)
				}
			}
		})
		b.Run(fmt.Sprintf("step_steady_seq_%d", seq), func(b *testing.B) {
			engine := openBenchmarkEngine(b, seq, 1<<30)
			if _, err := engine.Step(); err != nil {
				b.Fatalf("warmup step: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := engine.Step(); err != nil {
					b.Fatalf("Step: %v", err)
				}
			}
		})
		b.Run(fmt.Sprintf("refresh_seq_%d", seq), func(b *testing.B) {
			engine := openBenchmarkEngine(b, seq, 1<<30)
			if d := engine.refreshANERuntimeForWeights(); d == 0 {
				b.Skip("ANE refresh unavailable")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				engine.mw.RMSFinal[i%len(engine.mw.RMSFinal)] += 1e-6
				_ = engine.refreshANERuntimeForWeights()
			}
		})
	}

	b.Run("step_update_seq_256", func(b *testing.B) {
		engine := openBenchmarkEngine(b, stories.SeqDefault, 1)
		if _, err := engine.Step(); err != nil {
			b.Fatalf("warmup update step: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := engine.Step(); err != nil {
				b.Fatalf("Step(update): %v", err)
			}
		}
	})
}

func openBenchmarkEngine(b *testing.B, seq, accumSteps int) *Engine {
	b.Helper()
	if os.Getenv("ANE_BENCH") != "1" {
		b.Skip("set ANE_BENCH=1 to run direct-Go ANE benchmarks")
	}
	modelPath, dataPath := benchmarkPaths(b)
	if err := ProbeDirectSequence(seq); err != nil {
		b.Skipf("direct seq=%d unavailable: %v", seq, err)
	}
	tokens, err := readTokenPrefix(dataPath, seq+1+64)
	if err != nil {
		b.Fatalf("read benchmark tokens: %v", err)
	}
	engine, err := Open(Options{
		ModelPath:      modelPath,
		Tokens:         tokens,
		Seq:            seq,
		AccumSteps:     accumSteps,
		LR:             3e-4,
		Seed:           42,
		GradTaskLimit:  3,
		UseANE:         true,
		HybridBackward: true,
	})
	if err != nil {
		b.Fatalf("Open benchmark engine: %v", err)
	}
	b.Cleanup(engine.Close)
	engine.Prepare()
	d := engine.Diagnostics()
	if !d.LayerForwardEnabled || !d.FinalHeadOffloadEnabled || !d.HybridBackwardEnabled {
		b.Fatalf("unexpected diagnostics: %+v", d)
	}
	return engine
}

func benchmarkPaths(b *testing.B) (modelPath, dataPath string) {
	b.Helper()
	root := benchmarkRepoRoot(b)
	modelPath = firstBenchmarkPath(
		os.Getenv("ANE_BENCH_MODEL"),
		filepath.Join(root, "..", "..", "assets", "models", "stories110M.bin"),
		"/Volumes/tmc/go/src/github.com/assets/models/stories110M.bin",
	)
	if modelPath == "" {
		b.Skip("stories110M.bin not found; set ANE_BENCH_MODEL")
	}
	dataPath = firstBenchmarkPath(
		os.Getenv("ANE_BENCH_DATA"),
		filepath.Join(root, "training", "tinystories_data00.bin"),
		filepath.Join(root, "tinystories_data00.bin"),
	)
	if dataPath == "" {
		b.Skip("tinystories_data00.bin not found; set ANE_BENCH_DATA")
	}
	return modelPath, dataPath
}

func benchmarkRepoRoot(b *testing.B) string {
	b.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		b.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func firstBenchmarkPath(paths ...string) string {
	for _, path := range paths {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}
