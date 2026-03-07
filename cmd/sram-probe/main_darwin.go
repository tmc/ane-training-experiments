//go:build darwin

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/maderix/ANE/ane"
	"github.com/tmc/mlx-go/mlx"
)

func main() {
	sizeMiB := flag.Int("size-mib", 64, "working set in MiB")
	iters := flag.Int("iters", 200, "number of add/eval iterations")
	flag.Parse()

	report, err := ane.New().Probe(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	elements := (*sizeMiB * 1024 * 1024) / 4
	data := make([]float32, elements)
	for i := range data {
		data[i] = float32(i % 13)
	}

	a, err := mlx.FromSlice(data, []int{elements})
	if err != nil {
		log.Fatal(err)
	}
	defer a.Free()

	b, err := mlx.FromSlice(data, []int{elements})
	if err != nil {
		log.Fatal(err)
	}
	defer b.Free()

	start := time.Now()
	var out *mlx.Array
	for i := 0; i < *iters; i++ {
		if out != nil {
			out.Free()
		}
		out = mlx.MustAdd(a, b, nil)
		if err := out.Eval(); err != nil {
			log.Fatal(err)
		}
	}
	elapsed := time.Since(start)
	if out != nil {
		defer out.Free()
	}

	bytesMoved := float64(*iters) * float64(elements*4*3)
	gbps := bytesMoved / elapsed.Seconds() / (1024 * 1024 * 1024)
	fmt.Printf("ANE has=%v cores=%d devices=%d | sizeMiB=%d iters=%d elapsed=%v approx_mem_gib_per_s=%.2f\n",
		report.HasANE, report.NumANECores, report.NumANEs, *sizeMiB, *iters, elapsed, gbps,
	)
}
