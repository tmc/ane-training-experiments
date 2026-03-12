//go:build darwin

package model

import (
	"os"
	"testing"

	"github.com/maderix/ANE/ane/mil"
	xane "github.com/tmc/apple/x/ane"
)

func BenchmarkXANETelemetryReportMetrics(b *testing.B) {
	if os.Getenv("ANE_BENCH") != "1" {
		b.Skip("set ANE_BENCH=1 to run x/ane telemetry benchmarks")
	}

	rt, err := xane.Open()
	if err != nil {
		b.Fatalf("x/ane open: %v", err)
	}
	defer rt.Close()
	snapshot := rt.Snapshot()

	const channels = 32
	milText := mil.GenIdentity(channels, 1)
	blob, err := mil.BuildIdentityWeightBlob(channels)
	if err != nil {
		b.Fatalf("build identity weights: %v", err)
	}

	opts := xane.CompileOptions{
		ModelType:     xane.ModelTypeMIL,
		MILText:       []byte(milText),
		WeightBlob:    blob,
		PerfStatsMask: ^uint32(0),
	}

	input := make([]float32, channels)
	for i := range input {
		input[i] = float32(i + 1)
	}

	for _, tc := range []struct {
		name   string
		report func(*testing.B, *xane.Kernel)
	}{
		{
			name: "eval_stats",
			report: func(b *testing.B, k *xane.Kernel) {
				var last xane.EvalStats
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					stats, err := k.EvalWithStats()
					if err != nil {
						b.Fatalf("EvalWithStats: %v", err)
					}
					last = stats
				}
				last.ReportMetrics(b)
			},
		},
		{
			name: "eval_telemetry",
			report: func(b *testing.B, k *xane.Kernel) {
				var last xane.EvalTelemetry
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					stats, err := k.EvalWithStats()
					if err != nil {
						b.Fatalf("EvalWithStats: %v", err)
					}
					last = k.Telemetry(stats)
				}
				last.ReportMetrics(b)
			},
		},
	} {
		b.Run(tc.name, func(b *testing.B) {
			k, cs, err := rt.CompileWithStats(opts)
			if err != nil {
				b.Fatalf("CompileWithStats: %v", err)
			}
			defer k.Close()
			if err := k.WriteInputF32(0, input); err != nil {
				b.Fatalf("WriteInputF32: %v", err)
			}
			if _, err := k.EvalWithStats(); err != nil {
				b.Fatalf("warmup EvalWithStats: %v", err)
			}
			tc.report(b, k)
			snapshot.ReportMetrics(b)
			cs.ReportMetrics(b)
		})
	}
}
