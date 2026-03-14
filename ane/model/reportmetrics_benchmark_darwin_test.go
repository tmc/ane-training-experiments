//go:build darwin

package model

import (
	"os"
	"testing"

	"github.com/maderix/ANE/ane/mil"
)

func BenchmarkModelReportMetrics(b *testing.B) {
	if os.Getenv("ANE_BENCH") != "1" {
		b.Skip("set ANE_BENCH=1 to run model telemetry benchmarks")
	}

	const channels = 32
	milText := mil.GenIdentity(channels, 1)
	blob, err := mil.BuildIdentityWeightBlob(channels)
	if err != nil {
		b.Fatalf("build identity weights: %v", err)
	}

	opts := CompileOptions{
		MILText:       milText,
		WeightBlob:    blob,
		WeightPath:    "@model_path/weights/weight.bin",
		PerfStatsMask: ^uint32(0),
	}

	input := make([]float32, channels)
	for i := range input {
		input[i] = float32(i + 1)
	}

	for _, tc := range []struct {
		name   string
		report func(*testing.B, *Kernel)
	}{
		{
			name: "eval_stats",
			report: func(b *testing.B, k *Kernel) {
				var last EvalStats
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					stats, err := k.EvalWithStats()
					if err != nil {
						b.Fatalf("EvalWithStats: %v", err)
					}
					last = stats
				}
				reportEvalStatsMetrics(b, last)
			},
		},
		{
			name: "eval_metrics_map",
			report: func(b *testing.B, k *Kernel) {
				var last EvalStats
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					stats, err := k.EvalWithStats()
					if err != nil {
						b.Fatalf("EvalWithStats: %v", err)
					}
					last = stats
				}
				// Emit only metrics map values to benchmark reporting overhead separately.
				for name, value := range last.Metrics {
					b.ReportMetric(value, "ane_"+name)
				}
			},
		},
	} {
		b.Run(tc.name, func(b *testing.B) {
			k, cs, err := CompileWithStats(opts)
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
			reportCompileStatsMetrics(b, cs)
		})
	}
}

func reportEvalStatsMetrics(b *testing.B, st EvalStats) {
	b.ReportMetric(float64(st.HWExecutionNS), "ane_hw_execution_ns/op")
	for name, value := range st.Metrics {
		b.ReportMetric(value, "ane_"+name)
	}
}

func reportCompileStatsMetrics(b *testing.B, cs CompileStats) {
	b.ReportMetric(float64(cs.CompileNS), "ane_compile_ns")
	b.ReportMetric(float64(cs.LoadNS), "ane_load_ns")
	b.ReportMetric(float64(cs.TotalNS), "ane_total_ns")
}
