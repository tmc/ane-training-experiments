//go:build darwin

package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/tmc/mlx-go/mlx"
)

func main() {
	n := flag.Int("n", 2048, "matrix dimension")
	iters := flag.Int("iters", 10, "matmul iterations")
	flag.Parse()

	aData := make([]float32, (*n)*(*n))
	bData := make([]float32, (*n)*(*n))
	for i := range aData {
		aData[i] = 1
		bData[i] = 1
	}

	a, err := mlx.FromSlice(aData, []int{*n, *n})
	if err != nil {
		log.Fatal(err)
	}
	defer a.Free()
	b, err := mlx.FromSlice(bData, []int{*n, *n})
	if err != nil {
		log.Fatal(err)
	}
	defer b.Free()

	start := time.Now()
	var c *mlx.Array
	for i := 0; i < *iters; i++ {
		if c != nil {
			c.Free()
		}
		c = mlx.MustMatmul(a, b, nil)
		if err := c.Eval(); err != nil {
			log.Fatal(err)
		}
	}
	elapsed := time.Since(start)
	if c != nil {
		defer c.Free()
	}

	ops := 2.0 * float64(*n) * float64(*n) * float64(*n) * float64(*iters)
	tflops := ops / elapsed.Seconds() / 1e12
	fmt.Printf("n=%d iters=%d elapsed=%v approx_tflops=%.3f device=%s\n", *n, *iters, elapsed, tflops, mlx.DefaultDevice())
}
