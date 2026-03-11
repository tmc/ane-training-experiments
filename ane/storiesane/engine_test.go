package storiesane

import (
	"math"
	"os"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

func TestClassifierTileRanges(t *testing.T) {
	tests := []struct {
		name  string
		vocab int
		tile  int
		want  []tileRange
	}{
		{
			name:  "empty",
			vocab: 0,
			tile:  2048,
		},
		{
			name:  "single partial",
			vocab: 500,
			tile:  2048,
			want:  []tileRange{{start: 0, size: 500}},
		},
		{
			name:  "multiple",
			vocab: 5000,
			tile:  2048,
			want: []tileRange{
				{start: 0, size: 2048},
				{start: 2048, size: 2048},
				{start: 4096, size: 904},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifierTileRanges(tt.vocab, tt.tile)
			if len(got) != len(tt.want) {
				t.Fatalf("len=%d want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("range[%d]=%+v want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestTrailerRoundTrip(t *testing.T) {
	e := &Engine{
		state: State{
			TokenPos:     123,
			PendingSteps: 7,
		},
		accumGRMS:   []float32{1.25, -2.5},
		accumGEmbed: []float32{3.5, 4.5, -5.5},
	}

	f, err := os.CreateTemp(t.TempDir(), "storiesane-ckpt-*.bin")
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	path := f.Name()
	if err := f.Truncate(int64(storiesCheckpointSize())); err != nil {
		t.Fatalf("Truncate: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if err := e.appendTrailer(path); err != nil {
		t.Fatalf("appendTrailer: %v", err)
	}

	other := &Engine{
		accumGRMS:   make([]float32, len(e.accumGRMS)),
		accumGEmbed: make([]float32, len(e.accumGEmbed)),
	}
	if err := other.loadTrailer(path); err != nil {
		t.Fatalf("loadTrailer: %v", err)
	}

	if other.state.TokenPos != e.state.TokenPos {
		t.Fatalf("TokenPos=%d want %d", other.state.TokenPos, e.state.TokenPos)
	}
	if other.state.PendingSteps != e.state.PendingSteps {
		t.Fatalf("PendingSteps=%d want %d", other.state.PendingSteps, e.state.PendingSteps)
	}
	for i := range e.accumGRMS {
		if other.accumGRMS[i] != e.accumGRMS[i] {
			t.Fatalf("accumGRMS[%d]=%v want %v", i, other.accumGRMS[i], e.accumGRMS[i])
		}
	}
	for i := range e.accumGEmbed {
		if other.accumGEmbed[i] != e.accumGEmbed[i] {
			t.Fatalf("accumGEmbed[%d]=%v want %v", i, other.accumGEmbed[i], e.accumGEmbed[i])
		}
	}
}

func TestFlushPendingAppliesUpdate(t *testing.T) {
	accum := &modelGrad{
		RMSFinal: []float32{2, 4},
		Embed:    []float32{3, 6, 9},
	}
	e := &Engine{
		mw: &stories.ModelWeights{
			RMSFinal: []float32{1, 1},
			Embed:    []float32{1, 2, 3},
		},
		opt: &stories.OptimState{
			RMSFinal: stories.NewAdamState(2),
			Embed:    stories.NewAdamState(3),
		},
		accum:       accum,
		lr:          1e-3,
		accumGRMS:   accum.RMSFinal,
		accumGEmbed: accum.Embed,
		state: State{
			PendingSteps: 2,
		},
	}

	if _, err := e.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if e.state.PendingSteps != 0 {
		t.Fatalf("PendingSteps=%d want 0", e.state.PendingSteps)
	}
	if e.state.AdamT != 1 {
		t.Fatalf("AdamT=%d want 1", e.state.AdamT)
	}
	if e.state.CumBatches != 1 {
		t.Fatalf("CumBatches=%d want 1", e.state.CumBatches)
	}
	for i, v := range e.accumGRMS {
		if v != 0 {
			t.Fatalf("accumGRMS[%d]=%v want 0", i, v)
		}
	}
	for i, v := range e.accumGEmbed {
		if v != 0 {
			t.Fatalf("accumGEmbed[%d]=%v want 0", i, v)
		}
	}
}

func TestEnsureLayerCachesPopulateMissingTaps(t *testing.T) {
	const seq = 2

	layer := newLayerGrad()
	for i := range layer.RMSAtt {
		layer.RMSAtt[i] = 0.9 + 0.0001*float32(i%11)
		layer.RMSFFN[i] = 1.1 - 0.0001*float32(i%7)
	}
	for i := range layer.Wq {
		layer.Wq[i] = 0.0002 * float32((i%13)-6)
		layer.Wk[i] = 0.00015 * float32((i%17)-8)
		layer.Wv[i] = 0.0001 * float32((i%19)-9)
		layer.Wo[i] = 0.00018 * float32((i%23)-11)
	}
	for i := range layer.W1 {
		layer.W1[i] = 0.00012 * float32((i%29)-14)
		layer.W3[i] = 0.00009 * float32((i%31)-15)
	}
	for i := range layer.W2 {
		layer.W2[i] = 0.00011 * float32((i%27)-13)
	}

	cache := newLayerCache(seq)
	for i := range cache.x {
		cache.x[i] = 0.01 * float32((i%9)-4)
		cache.x2[i] = 0.015 * float32((i%7)-3)
	}
	e := &Engine{seq: seq}

	e.ensureAttentionCache(&layer, &cache)
	if !cache.attTapsReady {
		t.Fatal("attTapsReady=false want true")
	}
	wantXNorm := make([]float32, len(cache.xNorm))
	wantQ := make([]float32, len(cache.q))
	wantK := make([]float32, len(cache.k))
	wantV := make([]float32, len(cache.v))
	wantAtt := make([]float32, len(cache.attOut))
	rmsNormCF(wantXNorm, cache.x, layer.RMSAtt, stories.Dim, seq)
	linearCF(wantQ, layer.Wq, wantXNorm, stories.Dim, stories.Dim, seq)
	linearCF(wantK, layer.Wk, wantXNorm, stories.Dim, stories.Dim, seq)
	linearCF(wantV, layer.Wv, wantXNorm, stories.Dim, stories.Dim, seq)
	causalAttentionCF(wantAtt, wantQ, wantK, wantV, stories.Heads, stories.Dim/stories.Heads, seq)
	if !slicesClose(cache.xNorm, wantXNorm, 0) {
		t.Fatal("xNorm mismatch")
	}
	if !slicesClose(cache.q, wantQ, 0) {
		t.Fatal("q mismatch")
	}
	if !slicesClose(cache.k, wantK, 0) {
		t.Fatal("k mismatch")
	}
	if !slicesClose(cache.v, wantV, 0) {
		t.Fatal("v mismatch")
	}
	if !slicesClose(cache.attOut, wantAtt, 0) {
		t.Fatal("attOut mismatch")
	}

	e.ensureFFNCache(&layer, &cache)
	if !cache.ffnTapsReady {
		t.Fatal("ffnTapsReady=false want true")
	}
	wantX2Norm := make([]float32, len(cache.x2Norm))
	wantH1 := make([]float32, len(cache.h1))
	wantH3 := make([]float32, len(cache.h3))
	wantGate := make([]float32, len(cache.gate))
	rmsNormCF(wantX2Norm, cache.x2, layer.RMSFFN, stories.Dim, seq)
	linearCF(wantH1, layer.W1, wantX2Norm, stories.Hidden, stories.Dim, seq)
	linearCF(wantH3, layer.W3, wantX2Norm, stories.Hidden, stories.Dim, seq)
	for i := range wantGate {
		wantGate[i] = silu32(wantH1[i]) * wantH3[i]
	}
	if !slicesClose(cache.x2Norm, wantX2Norm, 0) {
		t.Fatal("x2Norm mismatch")
	}
	if !slicesClose(cache.h1, wantH1, 0) {
		t.Fatal("h1 mismatch")
	}
	if !slicesClose(cache.h3, wantH3, 0) {
		t.Fatal("h3 mismatch")
	}
	if !slicesClose(cache.gate, wantGate, 0) {
		t.Fatal("gate mismatch")
	}
}

func TestTrailerRoundTripAccumV2(t *testing.T) {
	accum := &modelGrad{
		RMSFinal: []float32{1.25, -2.5},
		Embed:    []float32{3.5, 4.5, -5.5},
	}
	e := &Engine{
		state: State{
			TokenPos:     321,
			PendingSteps: 2,
		},
		accum:       accum,
		accumGRMS:   accum.RMSFinal,
		accumGEmbed: accum.Embed,
	}

	f, err := os.CreateTemp(t.TempDir(), "storiesane-ckpt-*.bin")
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	path := f.Name()
	if err := f.Truncate(int64(storiesCheckpointSize())); err != nil {
		t.Fatalf("Truncate: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := e.appendTrailer(path); err != nil {
		t.Fatalf("appendTrailer: %v", err)
	}

	otherAccum := &modelGrad{
		RMSFinal: make([]float32, len(accum.RMSFinal)),
		Embed:    make([]float32, len(accum.Embed)),
	}
	other := &Engine{
		accum:       otherAccum,
		accumGRMS:   otherAccum.RMSFinal,
		accumGEmbed: otherAccum.Embed,
	}
	if err := other.loadTrailer(path); err != nil {
		t.Fatalf("loadTrailer: %v", err)
	}
	if other.state.TokenPos != e.state.TokenPos {
		t.Fatalf("TokenPos=%d want %d", other.state.TokenPos, e.state.TokenPos)
	}
	if other.state.PendingSteps != e.state.PendingSteps {
		t.Fatalf("PendingSteps=%d want %d", other.state.PendingSteps, e.state.PendingSteps)
	}
	for i, want := range accum.RMSFinal {
		if other.accum.RMSFinal[i] != want {
			t.Fatalf("RMSFinal[%d]=%v want %v", i, other.accum.RMSFinal[i], want)
		}
	}
	for i, want := range accum.Embed {
		if other.accum.Embed[i] != want {
			t.Fatalf("Embed[%d]=%v want %v", i, other.accum.Embed[i], want)
		}
	}
}

func slicesClose(got, want []float32, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			return false
		}
	}
	return true
}

func TestFillDecodeWindow(t *testing.T) {
	got := make([]uint16, 5)
	fillDecodeWindow(got, []int{7, 8})
	want := []uint16{storiesBOS, storiesBOS, storiesBOS, 7, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("window[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func TestDiagnosticsReportsLayerFailure(t *testing.T) {
	e := &Engine{
		useANE:                  true,
		layerInitErr:            os.ErrNotExist,
		hybridBackwardRequested: true,
		backwardInitErr:         os.ErrPermission,
		off:                     &offload{},
	}
	d := e.Diagnostics()
	if !d.UseANE || !d.LayerForwardRequested {
		t.Fatalf("Diagnostics() did not report ANE request: %+v", d)
	}
	if d.LayerInitError == "" {
		t.Fatalf("Diagnostics() missing LayerInitError: %+v", d)
	}
	if !d.HybridBackwardRequested {
		t.Fatalf("Diagnostics() missing HybridBackwardRequested: %+v", d)
	}
	if d.BackwardInitError == "" {
		t.Fatalf("Diagnostics() missing BackwardInitError: %+v", d)
	}
}

func TestEnsureLayersCompileFailureSetsDiagnostics(t *testing.T) {
	old := compileStoriesLayerForwardFunc
	compileStoriesLayerForwardFunc = func(stories.LayerWeights, int) (*layerForward, error) {
		return nil, os.ErrPermission
	}
	defer func() {
		compileStoriesLayerForwardFunc = old
	}()

	e := &Engine{
		mw: &stories.ModelWeights{
			Layers: make([]stories.LayerWeights, stories.NLayers),
		},
		seq:    8,
		useANE: true,
	}
	err := e.ensureLayers()
	if err == nil {
		t.Fatal("ensureLayers succeeded; want error")
	}
	d := e.Diagnostics()
	if !d.LayerForwardRequested || d.LayerForwardEnabled {
		t.Fatalf("unexpected diagnostics: %+v", d)
	}
	if d.LayerInitError == "" {
		t.Fatalf("LayerInitError is empty: %+v", d)
	}
}

func TestRMSNormGradWeightsMatchesCPUBackward(t *testing.T) {
	const (
		dim = 3
		seq = 2
	)
	dy := []float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
	}
	x := []float32{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
	}
	w := []float32{1.1, 0.9, 1.2}
	got := make([]float32, dim)
	rmsNormGradWeights(got, dy, x, w, dim, seq)

	wantDX := make([]float32, dim*seq)
	wantDW := make([]float32, dim)
	stories.RMSNormBackward(wantDX, wantDW, dy, x, w, dim, seq)
	for i, want := range wantDW {
		if diff := math.Abs(float64(got[i] - want)); diff > 1e-5 {
			t.Fatalf("dw[%d]=%v want %v diff=%v", i, got[i], want, diff)
		}
	}
}

func TestCrossEntropyLossFromProbsMatchesCPU(t *testing.T) {
	const (
		vocab = 4
		seq   = 3
	)
	logits := []float32{
		0.4, -0.2, 1.3,
		0.1, 0.7, -0.1,
		-0.3, 0.5, 0.2,
		1.0, -0.4, 0.0,
	}
	targets := []uint16{3, 2, 9}
	probs := make([]float32, len(logits))
	for t := 0; t < seq; t++ {
		maxv := logits[t]
		for v := 1; v < vocab; v++ {
			if x := logits[v*seq+t]; x > maxv {
				maxv = x
			}
		}
		sum := 0.0
		for v := 0; v < vocab; v++ {
			e := math.Exp(float64(logits[v*seq+t] - maxv))
			probs[v*seq+t] = float32(e)
			sum += e
		}
		for v := 0; v < vocab; v++ {
			probs[v*seq+t] /= float32(sum)
		}
	}
	gotGrad := make([]float32, len(logits))
	gotLoss := crossEntropyLossFromProbs(gotGrad, probs, targets, vocab, seq)

	wantGrad := make([]float32, len(logits))
	wantLoss := stories.CrossEntropyLoss(wantGrad, logits, targets, vocab, seq)
	if diff := math.Abs(float64(gotLoss - wantLoss)); diff > 1e-5 {
		t.Fatalf("loss=%v want %v diff=%v", gotLoss, wantLoss, diff)
	}
	for i, want := range wantGrad {
		if diff := math.Abs(float64(gotGrad[i] - want)); diff > 1e-5 {
			t.Fatalf("grad[%d]=%v want %v diff=%v", i, gotGrad[i], want, diff)
		}
	}
}
