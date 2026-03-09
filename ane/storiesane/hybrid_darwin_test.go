//go:build darwin

package storiesane

import (
	"errors"
	"strings"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

func TestEnsureBackwardCompileFailureSetsDiagnostics(t *testing.T) {
	old := compileStoriesLayerBackwardFunc
	compileStoriesLayerBackwardFunc = func(stories.LayerWeights, int) (*layerBackward, error) {
		return nil, errors.New("boom")
	}
	defer func() {
		compileStoriesLayerBackwardFunc = old
	}()

	e := &Engine{
		mw: &stories.ModelWeights{
			Layers: make([]stories.LayerWeights, stories.NLayers),
		},
		seq:                     8,
		useANE:                  true,
		hybridBackwardRequested: true,
	}
	err := e.ensureBackward()
	if err == nil {
		t.Fatal("ensureBackward succeeded; want error")
	}
	d := e.Diagnostics()
	if !d.HybridBackwardRequested || d.HybridBackwardEnabled {
		t.Fatalf("unexpected diagnostics: %+v", d)
	}
	if !strings.Contains(d.BackwardInitError, "boom") {
		t.Fatalf("BackwardInitError=%q want substring boom", d.BackwardInitError)
	}
}
