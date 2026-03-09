//go:build darwin

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maderix/ANE/ane"
	"github.com/maderix/ANE/ane/mil"
	"github.com/maderix/ANE/ane/model"
	"github.com/tmc/mlx-go/mlx"
)

func main() {
	r := ane.New()
	report, err := r.Probe(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	a, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2})
	if err != nil {
		log.Fatalf("create a: %v", err)
	}
	defer a.Free()

	b, err := mlx.FromSlice([]float32{0.5, 1.5, 2.0, 3.0}, []int{2, 2})
	if err != nil {
		log.Fatalf("create b: %v", err)
	}
	defer b.Free()

	y := mlx.MustAdd(mlx.MustMultiply(a, b, nil), a, nil)
	defer y.Free()
	if err := y.Eval(); err != nil {
		log.Fatalf("eval result: %v", err)
	}

	aneOut, aneErr := runANESmokeKernel()

	fmt.Printf("ANE: has=%v cores=%d devices=%d arch=%q build=%q connectAttempt=%v connectStatus=%d\n",
		report.HasANE, report.NumANECores, report.NumANEs, report.Architecture, report.BuildVersion,
		report.ConnectAttempt, report.ConnectStatus,
	)
	if aneErr != nil {
		fmt.Printf("ANE in-memory kernel: warning: %v\n", aneErr)
	} else {
		fmt.Printf("ANE in-memory kernel output sample (fp16->f32): %v\n", aneOut)
	}
	fmt.Printf("MLX: device=%s result=%s\n", mlx.DefaultDevice(), y.String())
}

func mustWeightBlob(weights []float32, outCh, inCh int) []byte {
	blob, err := mil.BuildWeightBlob(weights, outCh, inCh)
	if err != nil {
		log.Fatalf("build ANE weight blob: %v", err)
	}
	return blob
}

func runANESmokeKernel() ([]float32, error) {
	const channels = 32
	const spatial = 32

	k, err := model.Compile(model.CompileOptions{
		MILText:    mil.GenConvFP16(channels, channels, spatial),
		WeightBlob: mustWeightBlob(identityWeights(channels), channels, channels),
	})
	if err != nil {
		return nil, err
	}
	defer k.Close()

	in := make([]float32, channels*spatial)
	for i := range in {
		in[i] = 1
	}
	xCF := f32ToF16Bits(in)
	if err := k.WriteInput(0, u16AsBytes(xCF)); err != nil {
		return nil, err
	}
	if err := k.Eval(); err != nil {
		return nil, err
	}
	outBits := make([]uint16, channels*spatial)
	if err := k.ReadOutput(0, u16AsBytes(outBits)); err != nil {
		return nil, err
	}
	out := f16BitsToF32(outBits)
	if len(out) > 16 {
		return out[:16], nil
	}
	return out, nil
}

func u16AsBytes(v []uint16) []byte {
	if len(v) == 0 {
		return nil
	}
	out := make([]byte, 2*len(v))
	for i, x := range v {
		out[2*i] = byte(x)
		out[2*i+1] = byte(x >> 8)
	}
	return out
}

func f32ToF16Bits(v []float32) []uint16 {
	out := make([]uint16, len(v))
	for i, x := range v {
		out[i] = mil.Float32ToHalfBits(x)
	}
	return out
}

func f16BitsToF32(v []uint16) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = mil.HalfBitsToFloat32(x)
	}
	return out
}

func identityWeights(ch int) []float32 {
	w := make([]float32, ch*ch)
	for i := 0; i < ch; i++ {
		w[i*ch+i] = 1
	}
	return w
}
