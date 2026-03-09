//go:build darwin

package dynamicmatmul

import (
	"math"
	"os"
	"testing"
)

func TestSmokeDynamicMatmulIdentityAndScale(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		batch  = 16
		inDim  = 16
		outDim = 16
	)

	ex, err := New(batch, inDim, outDim, Options{})
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer ex.Close()

	x := make([]float32, batch*inDim)
	for i := range x {
		x[i] = float32(i+1) * 0.01
	}

	identity := make([]float32, inDim*outDim)
	scaled := make([]float32, inDim*outDim)
	for d := 0; d < inDim; d++ {
		identity[d*outDim+d] = 1
		scaled[d*outDim+d] = 2
	}

	got, st, err := ex.EvalWithStats(x, identity)
	if err != nil {
		t.Fatalf("EvalWithStats(identity): %v", err)
	}
	if diff := maxDiff(got, x); diff > 0.02 {
		t.Fatalf("identity max diff=%v", diff)
	}
	t.Logf("identity HWExecutionNS=%d", st.HWExecutionNS)

	got, _, err = ex.EvalWithStats(x, scaled)
	if err != nil {
		t.Fatalf("EvalWithStats(scale): %v", err)
	}
	want := make([]float32, len(x))
	for i, v := range x {
		want[i] = 2 * v
	}
	if diff := maxDiff(got, want); diff > 0.04 {
		t.Fatalf("scaled max diff=%v", diff)
	}
}

func TestSmokeDynamicMatmulForcedTiling(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		batch  = 8
		inDim  = 8
		outDim = 16
	)

	ex, err := New(batch, inDim, outDim, Options{TileOut: 4})
	if err != nil {
		t.Fatalf("New(TileOut=4) failed: %v", err)
	}
	defer ex.Close()

	x := make([]float32, batch*inDim)
	for i := range x {
		x[i] = float32(i+1) * 0.01
	}

	w := make([]float32, inDim*outDim)
	for d := 0; d < inDim; d++ {
		w[d*outDim+d] = 1
	}
	got, _, err := ex.EvalWithStats(x, w)
	if err != nil {
		t.Fatalf("EvalWithStats(tiled): %v", err)
	}

	want := make([]float32, batch*outDim)
	for b := 0; b < batch; b++ {
		copy(want[b*outDim:b*outDim+inDim], x[b*inDim:(b+1)*inDim])
	}
	if diff := maxDiff(got, want); diff > 0.04 {
		t.Fatalf("forced-tiled max diff=%v", diff)
	}
}

func TestSmokeDynamicMatmulChannelFirstIdentity(t *testing.T) {
	if os.Getenv("ANE_SMOKE") == "" {
		t.Skip("set ANE_SMOKE=1 to run private-framework smoke test")
	}

	const (
		batch  = 8
		inDim  = 8
		outDim = 8
	)

	ex, err := New(batch, inDim, outDim, Options{TileOut: outDim})
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer ex.Close()

	wIO := make([]float32, inDim*outDim)
	for d := 0; d < inDim; d++ {
		wIO[d*outDim+d] = 1
	}
	if err := ex.PrimeWeightsIO(wIO); err != nil {
		t.Fatalf("PrimeWeightsIO(identity): %v", err)
	}

	xCF := make([]float32, inDim*batch)
	for i := range xCF {
		xCF[i] = float32(i+1) * 0.01
	}
	got := make([]float32, outDim*batch)
	if _, err := ex.EvalCFIOInto(got, xCF); err != nil {
		t.Fatalf("EvalCFIOInto(identity): %v", err)
	}
	if diff := maxDiff(got, xCF); diff > 0.04 {
		t.Fatalf("channel-first identity max diff=%v", diff)
	}
}

func maxDiff(got, want []float32) float64 {
	var max float64
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > max {
			max = d
		}
	}
	return max
}
