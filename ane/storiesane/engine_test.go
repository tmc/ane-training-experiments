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

	if err := e.Flush(); err != nil {
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
