//go:build darwin

package model

import (
	"testing"

	xane "github.com/tmc/apple/x/ane"
	xanetelemetry "github.com/tmc/apple/x/ane/telemetry"
)

func TestEvalStatsMetrics(t *testing.T) {
	st := xanetelemetry.EvalStats{
		HWExecutionNS:         123,
		PerfCounterData:       make([]byte, 7),
		RawStatsData:          make([]byte, 11),
		PerfCounters:          []xanetelemetry.PerfCounter{{Index: 3, Name: "cycles", Value: 17}, {Index: 9, Value: 29}},
		PerfCountersTruncated: true,
	}

	got := evalStatsMetrics(st)
	want := map[string]float64{
		"PerfCounterBytes":      7,
		"RawStatsBytes":         11,
		"PerfCounter.cycles":    17,
		"PerfCounter.9":         29,
		"PerfCountersTruncated": 1,
	}
	for key, wantValue := range want {
		if got[key] != wantValue {
			t.Fatalf("%s=%v, want %v", key, got[key], wantValue)
		}
	}
	if _, ok := got["HWExecutionNS"]; ok {
		t.Fatal("HWExecutionNS should not be duplicated into Metrics")
	}
}

func TestAdaptCompileStats(t *testing.T) {
	got := adaptCompileStats(xane.CompileStats{
		CompileNS: 13,
		LoadNS:    21,
		TotalNS:   34,
	})
	if got.CompileNS != 13 || got.LoadNS != 21 || got.TotalNS != 34 {
		t.Fatalf("compile stats=%+v", got)
	}
}
