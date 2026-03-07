package linear

import (
	"context"
	"testing"
)

func TestKernelKeyStable(t *testing.T) {
	w := []float32{1, 2, 3, 4}
	k1 := kernelKey(w, 2, 2, 2)
	k2 := kernelKey(w, 2, 2, 2)
	if k1 != k2 {
		t.Fatalf("kernelKey not stable: %q != %q", k1, k2)
	}
}

func TestKernelKeyChangesWithInputs(t *testing.T) {
	base := kernelKey([]float32{1, 2, 3, 4}, 2, 2, 2)
	tests := []struct {
		name string
		key  string
	}{
		{name: "weight", key: kernelKey([]float32{1, 2, 3, 5}, 2, 2, 2)},
		{name: "batch", key: kernelKey([]float32{1, 2, 3, 4}, 3, 2, 2)},
		{name: "inDim", key: kernelKey([]float32{1, 2, 3, 4}, 2, 3, 2)},
		{name: "outDim", key: kernelKey([]float32{1, 2, 3, 4}, 2, 2, 3)},
	}
	for _, tt := range tests {
		if tt.key == base {
			t.Fatalf("kernelKey unchanged for %s", tt.name)
		}
	}
}

func TestLinearRejectsBadShapes(t *testing.T) {
	ex := New(Options{})
	_, err := ex.Linear(context.Background(), []float32{1}, []float32{1}, 1, 2, 1)
	if err == nil {
		t.Fatal("Linear accepted invalid input length")
	}
	_, err = ex.Linear(context.Background(), []float32{1, 2}, []float32{1}, 1, 2, 1)
	if err == nil {
		t.Fatal("Linear accepted invalid weight length")
	}
}

func TestLinearRespectsContext(t *testing.T) {
	ex := New(Options{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ex.Linear(ctx, []float32{1, 2}, []float32{1, 2}, 1, 2, 1)
	if err == nil {
		t.Fatal("Linear ignored canceled context")
	}
}

func TestPrepareRejectsBadShapes(t *testing.T) {
	ex := New(Options{})
	if err := ex.Prepare([]float32{1}, 0, 1, 1); err == nil {
		t.Fatal("Prepare accepted invalid shape")
	}
	if err := ex.Prepare([]float32{1}, 1, 2, 1); err == nil {
		t.Fatal("Prepare accepted invalid weight length")
	}
}
