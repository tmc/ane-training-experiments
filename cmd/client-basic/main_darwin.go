//go:build darwin

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/maderix/ANE/ane"
)

func main() {
	mlpackage := flag.String("mlpackage", "", "path to source .mlpackage")
	compiled := flag.String("compiled", "", "path to precompiled .mlmodelc (optional)")
	channels := flag.Int("channels", 1024, "channels for throughput estimate")
	spatial := flag.Int("spatial", 64, "spatial for throughput estimate")
	iters := flag.Int("iters", 50, "benchmark iterations")
	warmup := flag.Int("warmup", 5, "warmup iterations")
	qos := flag.Uint("qos", 21, "ANE QoS")
	modelKey := flag.String("model-key", "s", "_ANEModel key")
	modelType := flag.String("model-type", "", "optional compile option kANEFModelType value")
	netPlist := flag.String("net-plist", "", "optional compile option kANEFNetPlistFilenameKey value")
	useEspressoIO := flag.Bool("espresso-io", false, "use Espresso-backed IO pool")
	espressoFrames := flag.Uint64("espresso-frames", 1, "Espresso IO frame count")
	flag.Parse()

	if *mlpackage == "" && *compiled == "" {
		log.Fatal("set -mlpackage or -compiled")
	}
	if *channels <= 0 || *spatial <= 0 {
		log.Fatal("channels and spatial must be > 0")
	}
	if *iters <= 0 {
		log.Fatal("iters must be > 0")
	}
	if *warmup < 0 {
		log.Fatal("warmup must be >= 0")
	}

	report, probeErr := ane.New().Probe(context.Background())

	bytesPerTensor := (*channels) * (*spatial) * 4
	ev, err := ane.OpenEvaluator(ane.EvalOptions{
		ModelPath:        *compiled,
		ModelPackagePath: *mlpackage,
		ModelKey:         *modelKey,
		ModelType:        *modelType,
		NetPlistFilename: *netPlist,
		QoS:              uint32(*qos),
		InputBytes:       uint32(bytesPerTensor),
		OutputBytes:      uint32(bytesPerTensor),
		UseEspressoIO:    *useEspressoIO,
		EspressoFrames:   *espressoFrames,
	})
	if err != nil {
		log.Fatalf("open evaluator: %v", err)
	}
	defer ev.Close()

	in := make([]float32, (*channels)*(*spatial))
	outFull := make([]float32, len(in))
	for i := range in {
		in[i] = float32(i%23) * 0.25
	}
	for i := 0; i < *warmup; i++ {
		if err := ev.EvalF32(in, outFull); err != nil {
			log.Fatalf("warmup eval %d/%d: %v", i+1, *warmup, err)
		}
	}

	start := time.Now()
	for i := 0; i < *iters; i++ {
		if err := ev.EvalF32(in, outFull); err != nil {
			log.Fatalf("benchmark eval %d/%d: %v", i+1, *iters, err)
		}
	}
	elapsed := time.Since(start)
	out := outFull
	if len(out) > 16 {
		out = out[:16]
	}

	ops := 2.0 * float64(*channels) * float64(*channels) * float64(*spatial) * float64(*iters)
	tflops := ops / elapsed.Seconds() / 1e12
	msPerEval := float64(elapsed) / float64(time.Millisecond) / float64(*iters)

	if probeErr != nil {
		fmt.Printf("ANE probe warning: %v\n", probeErr)
	} else {
		fmt.Printf("ANE: has=%v cores=%d devices=%d arch=%q build=%q\n",
			report.HasANE, report.NumANECores, report.NumANEs, report.Architecture, report.BuildVersion)
	}
	fmt.Printf("evaluator: espresso_io=%v espresso_frames=%d\n", ev.EspressoEnabled(), *espressoFrames)
	fmt.Printf("benchmark: warmup=%d iters=%d ms/eval=%.3f approx_tflops=%.3f\n", *warmup, *iters, msPerEval, tflops)
	fmt.Printf("output sample f32: %v\n", out)
}
