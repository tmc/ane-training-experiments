package storiesane

import (
	"os"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

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
	e := &Engine{
		mw: &stories.ModelWeights{
			RMSFinal: []float32{1, 1},
			Embed:    []float32{1, 2, 3},
		},
		opt: &stories.OptimState{
			RMSFinal: stories.NewAdamState(2),
			Embed:    stories.NewAdamState(3),
		},
		lr:          1e-3,
		accumGRMS:   []float32{2, 4},
		accumGEmbed: []float32{3, 6, 9},
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
