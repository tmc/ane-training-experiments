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
		MILText:    mil.GenConvFP16(channels, channels, spatial),
		WeightBlob: mustWeightBlob(t, identityWeights(channels), channels, channels),
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

func TestSmokeCompileAndEvalFFNForwardTaps(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		dim    = 8
		hidden = 16
		seq    = 4
	)
	k, err := Compile(CompileOptions{
		MILText: mil.GenFFNForwardTaps(dim, hidden, seq),
		WeightFiles: []WeightFile{
			{Path: "@model_path/weights/rms2.bin", Blob: mustVectorWeightBlob(t, filledWeights(dim, 1))},
			{Path: "@model_path/weights/w1.bin", Blob: mustWeightBlob(t, filledWeights(hidden*dim, 0.25), hidden, dim)},
			{Path: "@model_path/weights/w2.bin", Blob: mustWeightBlob(t, filledWeights(dim*hidden, 0.125), dim, hidden)},
			{Path: "@model_path/weights/w3.bin", Blob: mustWeightBlob(t, filledWeights(hidden*dim, 0.5), hidden, dim)},
		},
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	inF32 := make([]float32, dim*seq)
	for i := range inF32 {
		inF32[i] = 1
	}
	if err := k.WriteInputFP16(0, inF32); err != nil {
		t.Fatalf("WriteInputFP16(x): %v", err)
	}
	if err := k.Eval(); err != nil {
		t.Fatalf("Eval(): %v", err)
	}

	out := make([]float32, (2*dim+3*hidden)*seq)
	if err := k.ReadOutputFP16(0, out); err != nil {
		t.Fatalf("ReadOutputFP16(): %v", err)
	}
	nonZero := false
	for i := range out {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("out[%d] is not finite: %v", i, out[i])
		}
		if math.Abs(float64(out[i])) > 1e-4 {
			nonZero = true
		}
	}
	if !nonZero {
		t.Fatal("ffn taps output is all zero")
	}
}

func TestSmokeCompileAndEvalFFNBackward(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		dim    = 8
		hidden = 16
		seq    = 4
	)
	k, err := Compile(CompileOptions{
		MILText: mil.GenFFNBackward(dim, hidden, seq),
		WeightFiles: []WeightFile{
			{Path: "@model_path/weights/w2t.bin", Blob: mustWeightBlob(t, filledWeights(hidden*dim, 0.125), hidden, dim)},
			{Path: "@model_path/weights/w1t.bin", Blob: mustWeightBlob(t, filledWeights(dim*hidden, 0.25), dim, hidden)},
			{Path: "@model_path/weights/w3t.bin", Blob: mustWeightBlob(t, filledWeights(dim*hidden, 0.5), dim, hidden)},
		},
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	inF32 := make([]float32, (dim+2*hidden)*seq)
	for i := range inF32 {
		inF32[i] = 1
	}
	if err := k.WriteInputFP16(0, inF32); err != nil {
		t.Fatalf("WriteInputFP16(x): %v", err)
	}
	if err := k.Eval(); err != nil {
		t.Fatalf("Eval(): %v", err)
	}

	out := make([]float32, (dim+2*hidden)*seq)
	if err := k.ReadOutputFP16(0, out); err != nil {
		t.Fatalf("ReadOutputFP16(): %v", err)
	}
	nonZero := false
	for i := range out {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("out[%d] is not finite: %v", i, out[i])
		}
		if math.Abs(float64(out[i])) > 1e-4 {
			nonZero = true
		}
	}
	if !nonZero {
		t.Fatal("ffn backward output is all zero")
	}
}

func TestSmokeCompileAndEvalSDPAForwardTaps(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		dim   = 8
		heads = 2
		seq   = 4
	)
	k, err := Compile(CompileOptions{
		MILText: mil.GenSDPAForwardTaps(dim, heads, seq),
		WeightFiles: []WeightFile{
			{Path: "@model_path/weights/rms1.bin", Blob: mustVectorWeightBlob(t, filledWeights(dim, 1))},
			{Path: "@model_path/weights/wq.bin", Blob: mustWeightBlob(t, filledWeights(dim*dim, 0.25), dim, dim)},
			{Path: "@model_path/weights/wk.bin", Blob: mustWeightBlob(t, filledWeights(dim*dim, 0.125), dim, dim)},
			{Path: "@model_path/weights/wv.bin", Blob: mustWeightBlob(t, filledWeights(dim*dim, 0.5), dim, dim)},
			{Path: "@model_path/weights/wo.bin", Blob: mustWeightBlob(t, filledWeights(dim*dim, 0.75), dim, dim)},
			{Path: "@model_path/weights/mask.bin", Blob: mustFP16Blob(t, buildCausalMask(t, seq))},
		},
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	checkFP16KernelOutput(t, k, dim*seq, 6*dim*seq, "sdpa forward taps")
}

func TestSmokeCompileAndEvalQKVBackward(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		dim   = 8
		heads = 2
		seq   = 4
	)
	base := rampWeights(dim * dim)
	k, err := Compile(CompileOptions{
		MILText: mil.GenQKVBackward(dim, heads, seq),
		WeightFiles: []WeightFile{
			{Path: "@model_path/weights/wqt.bin", Blob: mustTransposedWeightBlob(t, base, dim, dim)},
			{Path: "@model_path/weights/wkt.bin", Blob: mustTransposedWeightBlob(t, base, dim, dim)},
			{Path: "@model_path/weights/wvt.bin", Blob: mustTransposedWeightBlob(t, base, dim, dim)},
		},
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	checkFP16KernelOutput(t, k, 3*dim*seq, dim*seq, "qkv backward")
}

func TestSmokeCompileAndEvalSDPABackward1(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		dim   = 8
		heads = 2
		seq   = 4
	)
	scoreCh := heads * seq
	k, err := Compile(CompileOptions{
		MILText: mil.GenSDPABackward1(dim, heads, seq),
		WeightFiles: []WeightFile{
			{Path: "@model_path/weights/wot.bin", Blob: mustTransposedWeightBlob(t, rampWeights(dim*dim), dim, dim)},
			{Path: "@model_path/weights/mask.bin", Blob: mustFP16Blob(t, buildCausalMask(t, seq))},
		},
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	checkFP16KernelOutput(t, k, 4*dim*seq, (dim+2*scoreCh)*seq, "sdpa backward1")
}

func TestSmokeCompileAndEvalSDPABackward2(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		dim   = 8
		heads = 2
		seq   = 4
	)
	scoreCh := heads * seq
	k, err := Compile(CompileOptions{
		MILText: mil.GenSDPABackward2(dim, heads, seq),
	})
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer k.Close()

	checkFP16KernelOutput(t, k, (2*scoreCh+2*dim)*seq, 2*dim*seq, "sdpa backward2")
}

func mustWeightBlob(t *testing.T, weights []float32, outCh, inCh int) []byte {
	t.Helper()
	blob, err := mil.BuildWeightBlob(weights, outCh, inCh)
	if err != nil {
		t.Fatalf("BuildWeightBlob() failed: %v", err)
	}
	return blob
}

func mustVectorWeightBlob(t *testing.T, weights []float32) []byte {
	t.Helper()
	blob, err := mil.BuildVectorWeightBlob(weights)
	if err != nil {
		t.Fatalf("BuildVectorWeightBlob() failed: %v", err)
	}
	return blob
}

func mustTransposedWeightBlob(t *testing.T, weights []float32, rows, cols int) []byte {
	t.Helper()
	blob, err := mil.BuildTransposedWeightBlob(weights, rows, cols)
	if err != nil {
		t.Fatalf("BuildTransposedWeightBlob() failed: %v", err)
	}
	return blob
}

func mustFP16Blob(t *testing.T, data []float32) []byte {
	t.Helper()
	blob, err := mil.BuildFP16Blob(data)
	if err != nil {
		t.Fatalf("BuildFP16Blob() failed: %v", err)
	}
	return blob
}

func checkFP16KernelOutput(t *testing.T, k *Kernel, inputElems, outputElems int, label string) {
	t.Helper()
	in := make([]float32, inputElems)
	for i := range in {
		in[i] = 1
	}
	if err := k.WriteInputFP16(0, in); err != nil {
		t.Fatalf("WriteInputFP16(%s): %v", label, err)
	}
	if err := k.Eval(); err != nil {
		t.Fatalf("Eval(%s): %v", label, err)
	}
	out := make([]float32, outputElems)
	if err := k.ReadOutputFP16(0, out); err != nil {
		t.Fatalf("ReadOutputFP16(%s): %v", label, err)
	}
	nonZero := false
	for i := range out {
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("%s out[%d] is not finite: %v", label, i, out[i])
		}
		if math.Abs(float64(out[i])) > 1e-4 {
			nonZero = true
		}
	}
	if !nonZero {
		t.Fatalf("%s output is all zero", label)
	}
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

func filledWeights(n int, v float32) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = v
	}
	return w
}

func rampWeights(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = 0.01 * float32((i%13)+1)
	}
	return w
}

func buildCausalMask(t *testing.T, seq int) []float32 {
	t.Helper()
	mask := make([]float32, seq*seq)
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			if j > i {
				mask[i*seq+j] = -65504
			}
		}
	}
	return mask
}
