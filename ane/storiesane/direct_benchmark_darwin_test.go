//go:build darwin

package storiesane

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode"

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
			var totals benchmarkMetricTotals
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				engine.stepMetrics.reset()
				start := time.Now()
				if _, err := engine.runFinalHead(finalHidden, targets); err != nil {
					b.Fatalf("runFinalHead: %v", err)
				}
				totals.addWall(time.Since(start))
				totals.addStepMetrics(&engine.stepMetrics)
			}
			reportBenchmarkMetrics(b, totals)
		})
		b.Run(fmt.Sprintf("step_steady_seq_%d", seq), func(b *testing.B) {
			engine := openBenchmarkEngine(b, seq, 1<<30)
			if _, err := engine.Step(); err != nil {
				b.Fatalf("warmup step: %v", err)
			}
			b.ReportAllocs()
			var totals benchmarkMetricTotals
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := engine.Step()
				if err != nil {
					b.Fatalf("Step: %v", err)
				}
				totals.addStepResult(res)
				totals.addCustom(engine.stepMetrics.customMetrics())
			}
			reportBenchmarkMetrics(b, totals)
		})
		b.Run(fmt.Sprintf("refresh_seq_%d", seq), func(b *testing.B) {
			engine := openBenchmarkEngine(b, seq, 1<<30)
			if d := engine.refreshANERuntimeForWeights(); d == 0 {
				b.Skip("ANE refresh unavailable")
			}
			b.ReportAllocs()
			var totals benchmarkMetricTotals
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				engine.mw.RMSFinal[i%len(engine.mw.RMSFinal)] += 1e-6
				d := engine.refreshANERuntimeForWeights()
				totals.addRefresh(d)
				totals.addWall(d)
			}
			reportBenchmarkMetrics(b, totals)
		})
	}

	b.Run("step_update_seq_256", func(b *testing.B) {
		engine := openBenchmarkEngine(b, stories.SeqDefault, 1)
		if _, err := engine.Step(); err != nil {
			b.Fatalf("warmup update step: %v", err)
		}
		b.ReportAllocs()
		var totals benchmarkMetricTotals
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := engine.Step()
			if err != nil {
				b.Fatalf("Step(update): %v", err)
			}
			totals.addStepResult(res)
			totals.addCustom(engine.stepMetrics.customMetrics())
		}
		reportBenchmarkMetrics(b, totals)
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
	engine.stepMetrics.enableCustomMetrics()
	engine.attachStepMetrics()
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

type benchmarkMetricTotals struct {
	wall      time.Duration
	aneEval   time.Duration
	cpuWork   time.Duration
	finalHead time.Duration
	embedGrad time.Duration
	rmsDW     time.Duration
	dwGEMM    time.Duration
	dwWait    time.Duration
	adam      time.Duration
	refresh   time.Duration
	custom    map[string]float64
}

func (m *benchmarkMetricTotals) addWall(d time.Duration) {
	m.wall += d
}

func (m *benchmarkMetricTotals) addRefresh(d time.Duration) {
	m.refresh += d
}

func (m *benchmarkMetricTotals) addStepMetrics(sm *aneStepMetrics) {
	if sm == nil {
		return
	}
	m.aneEval += sm.aneEval()
	m.finalHead += sm.finalHead()
	m.embedGrad += sm.embedGrad()
	m.rmsDW += sm.rmsDW()
	m.dwGEMM += sm.dwGEMM()
	m.dwWait += sm.dwWait()
	m.adam += sm.adam()
	m.refresh += sm.refresh()
	m.addCustom(sm.customMetrics())
}

func (m *benchmarkMetricTotals) addStepResult(res StepResult) {
	m.wall += res.StepDuration
	m.aneEval += res.ANEEvalDuration
	m.cpuWork += res.CPUWorkDuration
	m.finalHead += res.FinalHeadDuration
	m.embedGrad += res.EmbedGradDuration
	m.rmsDW += res.RMSDWDuration
	m.dwGEMM += res.DWGEMMDuration
	m.dwWait += res.DWWaitDuration
	m.adam += res.AdamDuration
	m.refresh += res.WeightRefreshDuration
}

func (m *benchmarkMetricTotals) addCustom(metrics map[string]float64) {
	if len(metrics) == 0 {
		return
	}
	if m.custom == nil {
		m.custom = make(map[string]float64, len(metrics))
	}
	for k, v := range metrics {
		m.custom[k] += v
	}
}

func reportBenchmarkMetrics(b *testing.B, totals benchmarkMetricTotals) {
	if b.N == 0 {
		return
	}
	reportDurationMetric(b, "wall_ns/op", totals.wall)
	reportDurationMetric(b, "ane_hw_ns/op", totals.aneEval)
	reportDurationMetric(b, "cpu_work_ns/op", totals.cpuWork)
	reportDurationMetric(b, "final_head_ns/op", totals.finalHead)
	reportDurationMetric(b, "embed_grad_ns/op", totals.embedGrad)
	reportDurationMetric(b, "rms_dw_ns/op", totals.rmsDW)
	reportDurationMetric(b, "dw_gemm_ns/op", totals.dwGEMM)
	reportDurationMetric(b, "dw_wait_ns/op", totals.dwWait)
	reportDurationMetric(b, "adam_ns/op", totals.adam)
	reportDurationMetric(b, "refresh_ns/op", totals.refresh)
	if totals.wall > 0 && totals.aneEval > 0 {
		b.ReportMetric(100*float64(totals.aneEval)/float64(totals.wall), "ane_hw_pct")
	}
	if len(totals.custom) == 0 {
		return
	}
	keys := slices.Sorted(maps.Keys(totals.custom))
	for _, key := range keys {
		b.ReportMetric(totals.custom[key]/float64(b.N), benchmarkMetricName(key))
	}
}

func reportDurationMetric(b *testing.B, unit string, total time.Duration) {
	if total <= 0 || b.N == 0 {
		return
	}
	b.ReportMetric(float64(total)/float64(b.N), unit)
}

func benchmarkMetricName(name string) string {
	var sb strings.Builder
	sb.WriteString("ane_")
	var prev rune
	for i, r := range name {
		if unicode.IsUpper(r) {
			if i > 0 && (unicode.IsLower(prev) || unicode.IsDigit(prev)) {
				sb.WriteByte('_')
			}
			sb.WriteRune(unicode.ToLower(r))
			prev = r
			continue
		}
		if unicode.IsLower(r) || unicode.IsDigit(r) {
			sb.WriteRune(r)
			prev = r
			continue
		}
		if sb.Len() > len("ane_") && prev != '_' {
			sb.WriteByte('_')
		}
		prev = '_'
	}
	s := strings.TrimSuffix(sb.String(), "_")
	if s == "ane" {
		s = "ane_metric"
	}
	return s + "/op"
}
