//go:build darwin

package model

import (
	"math"
	"os"
	"testing"

	"github.com/maderix/ANE/ane/mil"
)

func TestSmokeCompileAndEvalMatmul(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const channels = 32
	const spatial = 32

	k, err := Compile(CompileOptions{
		MILText:     mil.GenConvFP16(channels, channels, spatial),
		WeightBlob:  mustWeightBlob(t, identityWeights(channels), channels, channels),
		InputBytes:  []int{channels * spatial * 2},
		OutputBytes: []int{channels * spatial * 2},
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	inF32 := make([]float32, channels*spatial)
	for i := range inF32 {
		inF32[i] = 1
	}
	x := f32ToF16Bits(inF32)
	if err := k.WriteInput(0, u16AsBytes(x)); err != nil {
		t.Fatalf("WriteInput(x): %v", err)
	}
	if err := k.Eval(); err != nil {
		t.Fatalf("Eval(): %v", err)
	}

	outBits := make([]uint16, channels*spatial)
	if err := k.ReadOutput(0, u16AsBytes(outBits)); err != nil {
		t.Fatalf("ReadOutput(): %v", err)
	}
	out := f16BitsToF32(outBits)
	for i := range out {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("out[%d] is not finite: %v", i, out[i])
		}
	}
}

func mustWeightBlob(t *testing.T, weights []float32, outCh, inCh int) []byte {
	t.Helper()
	blob, err := mil.BuildWeightBlob(weights, outCh, inCh)
	if err != nil {
		t.Fatalf("BuildWeightBlob() failed: %v", err)
	}
	return blob
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
