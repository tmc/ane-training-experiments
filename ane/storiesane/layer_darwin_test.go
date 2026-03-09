//go:build darwin

package storiesane

import (
	"math"
	"os"
	"testing"
)

func TestSmokeLayerForwardMatchesCPU(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run storiesane layer smoke test")
	}

	const (
		dim    = 8
		hidden = 16
		heads  = 2
		seq    = 4
	)
	w := layerForwardWeights{
		RMSAtt: filledSeq(dim, 1.0, 0.01),
		Wq:     filledSeq(dim*dim, 0.02, 0.001),
		Wk:     filledSeq(dim*dim, -0.015, 0.0015),
		Wv:     filledSeq(dim*dim, 0.01, -0.0008),
		Wo:     filledSeq(dim*dim, 0.03, 0.0009),
		RMSFFN: filledSeq(dim, 0.9, 0.02),
		W1:     filledSeq(hidden*dim, 0.01, 0.0007),
		W2:     filledSeq(dim*hidden, -0.005, 0.0006),
		W3:     filledSeq(hidden*dim, 0.015, -0.0005),
	}
	x := filledSeq(dim*seq, -0.2, 0.03)

	lf, err := compileLayerForward(dim, hidden, heads, seq, w)
	if err != nil {
		t.Fatalf("compileLayerForward: %v", err)
	}
	defer lf.close()

	got := make([]float32, dim*seq)
	if err := lf.run(got, x); err != nil {
		t.Fatalf("run: %v", err)
	}

	want := cpuLayerForward(dim, hidden, heads, seq, w, x)
	maxDiff := 0.0
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 0.1 {
		t.Fatalf("max diff=%v want <= 0.1", maxDiff)
	}
}

func cpuLayerForward(dim, hidden, heads, seq int, w layerForwardWeights, x []float32) []float32 {
	headDim := dim / heads
	xNorm := make([]float32, dim*seq)
	qf := make([]float32, dim*seq)
	kf := make([]float32, dim*seq)
	vf := make([]float32, dim*seq)
	attf := make([]float32, dim*seq)
	oo := make([]float32, dim*seq)
	x2 := make([]float32, dim*seq)
	x2Norm := make([]float32, dim*seq)
	h1 := make([]float32, hidden*seq)
	h3 := make([]float32, hidden*seq)
	gate := make([]float32, hidden*seq)
	y := make([]float32, dim*seq)
	out := make([]float32, dim*seq)

	rmsNormCF(xNorm, x, w.RMSAtt, dim, seq)
	linearCF(qf, w.Wq, xNorm, dim, dim, seq)
	linearCF(kf, w.Wk, xNorm, dim, dim, seq)
	linearCF(vf, w.Wv, xNorm, dim, dim, seq)
	causalAttentionCF(attf, qf, kf, vf, heads, headDim, seq)
	linearCF(oo, w.Wo, attf, dim, dim, seq)
	for i := range x2 {
		x2[i] = x[i] + oo[i]
	}

	rmsNormCF(x2Norm, x2, w.RMSFFN, dim, seq)
	linearCF(h1, w.W1, x2Norm, hidden, dim, seq)
	linearCF(h3, w.W3, x2Norm, hidden, dim, seq)
	for i := range gate {
		gate[i] = silu32(h1[i]) * h3[i]
	}
	linearCF(y, w.W2, gate, dim, hidden, seq)
	for i := range out {
		out[i] = x2[i] + y[i]
	}
	return out
}

func filledSeq(n int, base, delta float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = base + delta*float32(i%17)
	}
	return out
}
