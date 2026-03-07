//go:build darwin

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/maderix/ANE/ane/pipeline"
)

type iterJSON struct {
	Type      string  `json:"type"`
	Iter      int     `json:"iter"`
	StepMS    float64 `json:"step_ms"`
	WaitMS    float64 `json:"wait_ms"`
	TotalMS   float64 `json:"total_ms"`
	SignalVal uint64  `json:"signal_value"`
}

func main() {
	var (
		modelPath   = flag.String("model", "/Users/tmc/ml-explore/mlx-go/experiment/mlx-go-ane/testdata/chaining/simple_add_nn.mlmodelc", "path to .mlmodelc")
		modelKey    = flag.String("model-key", "s", "_ANEModel key")
		inputBytes  = flag.Uint("input-bytes", 4096, "input tensor bytes")
		outputBytes = flag.Uint("output-bytes", 4096, "output tensor bytes")
		iters       = flag.Int("iters", 10, "iterations")
		timeoutMS   = flag.Uint("timeout-ms", 5000, "signal wait timeout")
		jsonOut     = flag.Bool("json", true, "emit JSON iteration telemetry to stderr")
	)
	flag.Parse()

	if *iters <= 0 {
		fatalf("iters must be > 0")
	}
	if *inputBytes == 0 || *outputBytes == 0 {
		fatalf("input-bytes and output-bytes must be > 0")
	}

	inCount := int(*inputBytes) / 4
	outCount := int(*outputBytes) / 4
	input := make([]float32, inCount)
	output := make([]float32, outCount)

	runner, err := pipeline.Open(pipeline.Options{
		ModelPath:     *modelPath,
		ModelKey:      *modelKey,
		InputBytes:    uint32(*inputBytes),
		OutputBytes:   uint32(*outputBytes),
		WaitTimeoutMS: uint32(*timeoutMS),
	})
	if err != nil {
		fatalf("open pipeline runner: %v", err)
	}
	defer func() { _ = runner.Close() }()

	fmt.Println("=== Asymmetric Pipeline (Go direct) ===")
	fmt.Printf("model=%s wait_port=%d signal_port=%d iters=%d input_f32=%d output_f32=%d\n",
		*modelPath, runner.WaitPort(), runner.SignalPort(), *iters, inCount, outCount)

	var totalMS float64
	for i := 1; i <= *iters; i++ {
		waitValue := uint64(i)
		signalValue := uint64(i)
		for j := range input {
			input[j] = float32(i + (j % 7))
		}

		stepStart := time.Now()
		if err := runner.SignalWaitFromCPU(waitValue); err != nil {
			fatalf("iter %d signal wait event: %v", i, err)
		}
		evalStart := time.Now()
		if err := runner.Eval(waitValue, signalValue, input, output); err != nil {
			fatalf("iter %d eval: %v", i, err)
		}
		evalMS := msSince(evalStart)
		waitStart := time.Now()
		ok, err := runner.WaitForSignal(signalValue, time.Duration(*timeoutMS)*time.Millisecond)
		if err != nil {
			fatalf("iter %d wait signal: %v", i, err)
		}
		if !ok {
			fatalf("iter %d wait signal timed out", i)
		}
		waitMS := msSince(waitStart)
		tot := msSince(stepStart)
		totalMS += tot

		fmt.Printf("iter=%d eval_ms=%.3f wait_ms=%.3f total_ms=%.3f out[0..3]=[%.4f %.4f %.4f %.4f]\n",
			i, evalMS, waitMS, tot, sample(output, 0), sample(output, 1), sample(output, 2), sample(output, 3))

		if *jsonOut {
			emitJSON(iterJSON{
				Type:      "iter",
				Iter:      i,
				StepMS:    evalMS,
				WaitMS:    waitMS,
				TotalMS:   tot,
				SignalVal: signalValue,
			})
		}
	}

	fmt.Printf("avg_total_ms=%.3f\n", totalMS/float64(*iters))
}

func sample(v []float32, idx int) float32 {
	if idx < 0 || idx >= len(v) {
		return 0
	}
	return v[idx]
}

func msSince(t time.Time) float64 {
	return float64(time.Since(t)) / float64(time.Millisecond)
}

func emitJSON(v any) {
	enc := json.NewEncoder(os.Stderr)
	if err := enc.Encode(v); err != nil {
		fatalf("emit json: %v", err)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
