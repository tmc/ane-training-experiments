//go:build darwin

package model

import "testing"

func TestCompileOptionsWithDefaultsBenchmarkPerfMask(t *testing.T) {
	t.Setenv("ANE_BENCH", "1")
	t.Setenv("ANE_PERF_STATS_MASK", "")

	got := compileOptionsWithDefaults(CompileOptions{})
	if got.QoS != defaultQoS {
		t.Fatalf("QoS=%d, want %d", got.QoS, defaultQoS)
	}
	if got.PerfStatsMask != ^uint32(0) {
		t.Fatalf("PerfStatsMask=%#x, want %#x", got.PerfStatsMask, ^uint32(0))
	}
}

func TestCompileOptionsWithDefaultsExplicitPerfMask(t *testing.T) {
	t.Setenv("ANE_BENCH", "1")
	t.Setenv("ANE_PERF_STATS_MASK", "0x12")

	got := compileOptionsWithDefaults(CompileOptions{})
	if got.PerfStatsMask != 0x12 {
		t.Fatalf("PerfStatsMask=%#x, want %#x", got.PerfStatsMask, 0x12)
	}
}
