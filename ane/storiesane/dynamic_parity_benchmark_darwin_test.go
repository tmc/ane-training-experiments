//go:build darwin

package storiesane

import (
	"testing"
)

// BenchmarkDynamicTrainStep measures steady-state train-step latency for the
// fully dynamic Go path (compile once, refresh weights in place).
func BenchmarkDynamicTrainStep(b *testing.B) {
	engine := openDynamicParityEngine(b, 384, 10)
	if _, err := engine.Step(); err != nil {
		b.Fatalf("warmup Step: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Step(); err != nil {
			b.Fatalf("Step: %v", err)
		}
	}
}

// BenchmarkDynamicStartupCompile measures one-time dynamic startup compile.
func BenchmarkDynamicStartupCompile(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping startup compile benchmark in short mode")
	}
	modelPath, dataPath := benchmarkPaths(b)
	tokens, err := readTokenPrefix(dataPath, 512)
	if err != nil {
		b.Fatalf("read benchmark tokens: %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		engine, err := Open(Options{
			ModelPath:      modelPath,
			Tokens:         tokens,
			Seq:            384,
			AccumSteps:     10,
			LR:             3e-4,
			Seed:           42,
			GradTaskLimit:  3,
			UseANE:         true,
			HybridBackward: true,
		})
		if err != nil {
			b.Fatalf("Open: %v", err)
		}
		engine.Prepare()
		st := engine.DynamicStatus()
		engine.Close()
		if !st.FullyDynamic() {
			b.Skipf("fully dynamic startup unavailable: %s", st.String())
		}
	}
}

func openDynamicParityEngine(b *testing.B, seq, accumSteps int) *Engine {
	b.Helper()
	if b.N > 0 && testing.Short() {
		b.Skip("skipping dynamic parity benchmark in short mode")
	}
	if err := ProbeDirectSequence(seq); err != nil {
		b.Skipf("direct seq=%d unavailable: %v", seq, err)
	}
	modelPath, dataPath := benchmarkPaths(b)
	tokens, err := readTokenPrefix(dataPath, seq+65)
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
	st := engine.DynamicStatus()
	if !st.FullyDynamic() {
		b.Skipf("fully dynamic path unavailable: %s", st.String())
	}
	return engine
}
