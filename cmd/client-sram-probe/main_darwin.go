//go:build darwin

package main

import (
	"context"
	"flag"
	"fmt"
	"time"

	"github.com/maderix/ANE/ane"
	"github.com/maderix/ANE/ane/clientmodel"
)

type probeCase struct {
	channels int
	spatial  int
}

func main() {
	pattern := flag.String("pattern", "/tmp/ane_sram_%dch_%dsp.mlpackage", "model package path pattern (used when -compiled-pattern is empty)")
	compiledPattern := flag.String("compiled-pattern", "", "compiled model path pattern (.mlmodelc), preferred over -pattern when set")
	warmup := flag.Int("warmup", 5, "warmup eval count")
	iters := flag.Int("iters", 50, "measured eval count")
	qos := flag.Uint("qos", 21, "ANE QoS")
	flag.Parse()

	if *iters <= 0 {
		fmt.Println("iters must be > 0")
		return
	}
	if *warmup < 0 {
		fmt.Println("warmup must be >= 0")
		return
	}

	probe, probeErr := ane.New().Probe(context.Background())
	if probeErr == nil {
		fmt.Printf("ANE: has=%v ne_cores=%d devices=%d arch=%q build=%q\n",
			probe.HasANE, probe.NumANECores, probe.NumANEs, probe.Architecture, probe.BuildVersion)
	}

	cases := []probeCase{
		{channels: 256, spatial: 64},
		{channels: 512, spatial: 64},
		{channels: 1024, spatial: 64},
		{channels: 1536, spatial: 64},
		{channels: 2048, spatial: 64},
		{channels: 2560, spatial: 64},
		{channels: 3072, spatial: 64},
		{channels: 3584, spatial: 64},
		{channels: 4096, spatial: 64},
		{channels: 4608, spatial: 64},
		{channels: 5120, spatial: 64},
		{channels: 6144, spatial: 64},
		{channels: 8192, spatial: 32},
	}

	fmt.Println("=== ANE SRAM Fine Probe (_ANEClient path) ===")
	fmt.Println()
	fmt.Printf("%-12s %8s %10s %8s %12s\n", "Channels", "W (MB)", "ms/eval", "TFLOPS", "GFLOPS/MB")
	fmt.Println("--------------------------------------------------------------")

	for i, tc := range cases {
		res, err := benchCase(*pattern, *compiledPattern, tc.channels, tc.spatial, *warmup, *iters, uint32(*qos))
		wMB := weightMiB(tc.channels)
		if err != nil {
			fmt.Printf("%6d ch   %7.1f  %8s %7s  %10s   (%v)\n", tc.channels, wMB, "ERR", "ERR", "ERR", err)
			continue
		}

		eff := 0.0
		if wMB > 0 {
			eff = res.tflops * 1000 / wMB
		}
		spillHint := ""
		if i > 0 && eff < 100 {
			spillHint = " <-- spilling?"
		}
		fmt.Printf("%6d ch   %7.1f  %8.3f ms %7.2f  %10.1f%s\n",
			tc.channels, wMB, res.msPerEval, res.tflops, eff, spillHint)
	}
	fmt.Println()
	fmt.Println("note: TFLOPS is measured fp16 throughput; Apple's published TOPS may use mixed precision and is not directly comparable.")
}

type benchResult struct {
	msPerEval float64
	tflops    float64
}

func benchCase(pattern, compiledPattern string, channels, spatial, warmup, iters int, qos uint32) (benchResult, error) {
	path := fmt.Sprintf(pattern, channels, spatial)
	compiledPath := ""
	if compiledPattern != "" {
		compiledPath = fmt.Sprintf(compiledPattern, channels, spatial)
	}
	bytes := channels * spatial * 4
	opts := clientmodel.CompileOptions{
		CompiledModelPath: compiledPath,
		ModelPackagePath:  path,
		QoS:               qos,
		InputBytes:        []int{bytes},
		OutputBytes:       []int{bytes},
	}
	if compiledPath != "" {
		opts.ModelPackagePath = ""
	}
	k, err := clientmodel.Compile(opts)
	if err != nil {
		return benchResult{}, err
	}
	defer k.Close()

	zeros := make([]byte, bytes)
	if err := k.WriteInput(0, zeros); err != nil {
		return benchResult{}, err
	}
	for i := 0; i < warmup; i++ {
		if err := k.Eval(); err != nil {
			return benchResult{}, err
		}
	}

	start := time.Now()
	for i := 0; i < iters; i++ {
		if err := k.Eval(); err != nil {
			return benchResult{}, err
		}
	}
	elapsed := time.Since(start)

	msPerEval := float64(elapsed) / float64(time.Millisecond) / float64(iters)
	gf := 2.0 * float64(channels) * float64(channels) * float64(spatial) / 1e9
	tflops := 0.0
	if msPerEval > 0 {
		tflops = gf / msPerEval
	}
	return benchResult{msPerEval: msPerEval, tflops: tflops}, nil
}

func weightMiB(ch int) float64 {
	return float64(ch*ch*2) / (1024 * 1024)
}
