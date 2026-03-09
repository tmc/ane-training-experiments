//go:build darwin

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	xane "github.com/tmc/apple/x/ane"
)

type report struct {
	Probe             *xane.DeviceInfo    `json:"probe,omitempty"`
	ProbeError        string              `json:"probe_error,omitempty"`
	ProbeOnly         bool                `json:"probe_only,omitempty"`
	ModelTarget       string              `json:"model_target,omitempty"`
	RuntimeInfo       *xane.DeviceInfo    `json:"runtime_info,omitempty"`
	CompileCount      int64               `json:"compile_count,omitempty"`
	CompileError      string              `json:"compile_error,omitempty"`
	KernelDiagnostics xane.Diagnostics    `json:"kernel_diagnostics,omitempty"`
	InputLayouts      []xane.TensorLayout `json:"input_layouts,omitempty"`
	OutputLayouts     []xane.TensorLayout `json:"output_layouts,omitempty"`
	InputAllocBytes   []int               `json:"input_alloc_bytes,omitempty"`
	OutputAllocBytes  []int               `json:"output_alloc_bytes,omitempty"`
	EvalAttempted     bool                `json:"eval_attempted,omitempty"`
	EvalOK            bool                `json:"eval_ok,omitempty"`
	EvalError         string              `json:"eval_error,omitempty"`
	EvalDurationMS    float64             `json:"eval_duration_ms,omitempty"`
	HWExecutionNS     uint64              `json:"hw_execution_ns,omitempty"`
	OutputSample      []float32           `json:"output_sample,omitempty"`
}

func main() {
	mlpackage := flag.String("mlpackage", "", "path to source .mlpackage")
	compiled := flag.String("compiled", "", "path to precompiled .mlmodelc")
	modelKey := flag.String("model-key", "s", "_ANEModel key")
	probeOnly := flag.Bool("probe-only", false, "collect ANE probe information without compiling a model")
	qos := flag.Uint("qos", 21, "ANE QoS")
	eval := flag.Bool("eval", true, "run one eval")
	flag.Parse()

	if !*probeOnly && *mlpackage == "" && *compiled == "" {
		fatalf("set -mlpackage or -compiled")
	}

	r := report{ProbeOnly: *probeOnly}

	info, err := xane.Probe()
	if err != nil {
		r.ProbeError = err.Error()
		if info.HasANE {
			r.Probe = &info
		}
	} else {
		r.Probe = &info
	}

	if *probeOnly {
		emitReport(r)
		return
	}

	rt, err := xane.Open()
	if err != nil {
		r.CompileError = fmt.Sprintf("open runtime: %v", err)
		emitAndExit(r, 2)
	}
	defer rt.Close()

	info = rt.Info()
	r.RuntimeInfo = &info
	r.CompileCount = rt.CompileCount()

	target := *compiled
	if target == "" {
		target = *mlpackage
	}
	r.ModelTarget = target

	k, err := rt.Compile(xane.CompileOptions{
		ModelType:     xane.ModelTypePackage,
		PackagePath:   target,
		ModelKey:      *modelKey,
		QoS:           uint32(*qos),
		PerfStatsMask: 1,
	})
	if err != nil {
		r.CompileError = err.Error()
		emitAndExit(r, 2)
	}
	defer k.Close()

	r.CompileCount = rt.CompileCount()
	r.KernelDiagnostics = k.Diagnostics()
	r.InputLayouts = make([]xane.TensorLayout, k.NumInputs())
	r.OutputLayouts = make([]xane.TensorLayout, k.NumOutputs())
	r.InputAllocBytes = make([]int, k.NumInputs())
	r.OutputAllocBytes = make([]int, k.NumOutputs())
	for i := 0; i < k.NumInputs(); i++ {
		r.InputLayouts[i] = k.InputLayout(i)
		r.InputAllocBytes[i] = k.InputAllocSize(i)
	}
	for i := 0; i < k.NumOutputs(); i++ {
		r.OutputLayouts[i] = k.OutputLayout(i)
		r.OutputAllocBytes[i] = k.OutputAllocSize(i)
	}

	r.EvalAttempted = *eval
	if *eval {
		if err := writePatternInput(k); err != nil {
			r.EvalError = fmt.Sprintf("write input: %v", err)
			emitAndExit(r, 2)
		}
		t0 := time.Now()
		stats, err := k.EvalWithStats()
		r.EvalDurationMS = float64(time.Since(t0)) / float64(time.Millisecond)
		if err != nil {
			r.EvalError = err.Error()
			emitAndExit(r, 2)
		}
		r.EvalOK = true
		r.HWExecutionNS = stats.HWExecutionNS
		if sample, err := sampleOutput(k, 0, 16); err == nil {
			r.OutputSample = sample
		}
	}

	emitReport(r)
}

func writePatternInput(k *xane.Kernel) error {
	layout := k.InputLayout(0)
	switch layout.ElemSize {
	case 4:
		data := make([]float32, layout.LogicalElements())
		for i := range data {
			data[i] = float32(i%251) * 0.25
		}
		return k.WriteInputF32(0, data)
	case 2:
		data := make([]float32, layout.LogicalElements())
		for i := range data {
			data[i] = float32(i%251) * 0.25
		}
		return k.WriteInputFP16(0, data)
	default:
		data := make([]byte, k.InputAllocSize(0))
		for i := range data {
			data[i] = byte(i % 251)
		}
		return k.WriteInput(0, data)
	}
}

func sampleOutput(k *xane.Kernel, output, n int) ([]float32, error) {
	layout := k.OutputLayout(output)
	limit := n
	if logical := layout.LogicalElements(); logical > 0 && logical < limit {
		limit = logical
	}
	switch layout.ElemSize {
	case 4:
		data := make([]float32, layout.LogicalElements())
		if err := k.ReadOutputF32(output, data); err != nil {
			return nil, err
		}
		return append([]float32(nil), data[:limit]...), nil
	case 2:
		data := make([]float32, layout.LogicalElements())
		if err := k.ReadOutputFP16(output, data); err != nil {
			return nil, err
		}
		return append([]float32(nil), data[:limit]...), nil
	default:
		return nil, fmt.Errorf("unsupported output elem size %d", layout.ElemSize)
	}
}

func emitAndExit(r report, code int) {
	emitReport(r)
	os.Exit(code)
}

func emitReport(r report) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		fatalf("encode report: %v", err)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
