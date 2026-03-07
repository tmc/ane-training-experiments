//go:build darwin

package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/maderix/ANE/ane/linear"
)

func main() {
	var (
		batch  = flag.Int("batch", 32, "batch size")
		inDim  = flag.Int("in", 1024, "input dimension")
		outDim = flag.Int("out", 1024, "output dimension")
		warmup = flag.Int("warmup", 2, "warmup iterations")
		iters  = flag.Int("iters", 20, "timed iterations")
		qos    = flag.Uint("qos", 21, "ANE QoS")
		seed   = flag.Int64("seed", 1, "random seed")
	)
	flag.Parse()

	if *batch <= 0 || *inDim <= 0 || *outDim <= 0 {
		fmt.Println("batch/in/out must be > 0")
		return
	}
	if *iters <= 0 || *warmup < 0 {
		fmt.Println("iters must be > 0 and warmup must be >= 0")
		return
	}

	rng := rand.New(rand.NewSource(*seed))
	x := make([]float32, (*batch)*(*inDim))
	w := make([]float32, (*outDim)*(*inDim))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range w {
		w[i] = rng.Float32()*2 - 1
	}

	ex := linear.New(linear.Options{QoS: uint32(*qos)})
	defer ex.Close()

	ctx := context.Background()
	for i := 0; i < *warmup; i++ {
		if _, err := ex.Linear(ctx, x, w, *batch, *inDim, *outDim); err != nil {
			fmt.Printf("warmup failed: %v\n", err)
			return
		}
	}

	start := time.Now()
	var out []float32
	for i := 0; i < *iters; i++ {
		y, err := ex.Linear(ctx, x, w, *batch, *inDim, *outDim)
		if err != nil {
			fmt.Printf("eval failed at iter %d: %v\n", i+1, err)
			return
		}
		out = y
	}
	elapsed := time.Since(start)
	msPerEval := float64(elapsed) / float64(time.Millisecond) / float64(*iters)
	ops := 2.0 * float64(*batch) * float64(*inDim) * float64(*outDim) * float64(*iters)
	tflops := ops / elapsed.Seconds() / 1e12
	s := ex.Stats()

	sample := 8
	if len(out) < sample {
		sample = len(out)
	}
	fmt.Printf("linear-go batch=%d in=%d out=%d warmup=%d iters=%d\n", *batch, *inDim, *outDim, *warmup, *iters)
	fmt.Printf("ms/eval=%.3f approx_tflops=%.3f compiles=%d cache_hits=%d kernels=%d\n", msPerEval, tflops, s.Compiles, s.CacheHits, s.Kernels)
	fmt.Printf("output_sample=%v\n", out[:sample])
}
