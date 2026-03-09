package linear

import (
	"context"
	"reflect"
	"testing"
)

func TestDynamicKernelKeyStable(t *testing.T) {
	k1 := dynamicKernelKey(2, 3, 4)
	k2 := dynamicKernelKey(2, 3, 4)
	if k1 != k2 {
		t.Fatalf("dynamicKernelKey not stable: %q != %q", k1, k2)
	}
}

func TestDynamicKernelKeyChangesWithShape(t *testing.T) {
	base := dynamicKernelKey(2, 3, 4)
	tests := []string{
		dynamicKernelKey(3, 3, 4),
		dynamicKernelKey(2, 4, 4),
		dynamicKernelKey(2, 3, 5),
	}
	for _, key := range tests {
		if key == base {
			t.Fatal("dynamicKernelKey unchanged for different shape")
		}
	}
}

func TestTransposeWeightsRowMajorOIToIO(t *testing.T) {
	src := []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	dst := make([]float32, len(src))
	transposeWeightsRowMajorOIToIO(dst, src, 2, 3)
	want := []float32{
		1, 3, 5,
		2, 4, 6,
	}
	if !reflect.DeepEqual(dst, want) {
		t.Fatalf("transposed weights=%v want %v", dst, want)
	}
}

func TestDynamicLinearRejectsBadShapes(t *testing.T) {
	ex := NewDynamic(Options{})
	_, err := ex.Linear(context.Background(), []float32{1}, []float32{1}, 1, 2, 1)
	if err == nil {
		t.Fatal("Linear accepted invalid input length")
	}
	_, err = ex.Linear(context.Background(), []float32{1, 2}, []float32{1}, 1, 2, 1)
	if err == nil {
		t.Fatal("Linear accepted invalid weight length")
	}
}

func TestDynamicLinearIntoRejectsBadShapes(t *testing.T) {
	ex := NewDynamic(Options{})
	_, err := ex.LinearIntoWithStats(context.Background(), []float32{0}, []float32{1}, []float32{1}, 1, 2, 1)
	if err == nil {
		t.Fatal("LinearIntoWithStats accepted invalid input length")
	}
	_, err = ex.LinearIntoWithStats(context.Background(), []float32{0}, []float32{1, 2}, []float32{1}, 1, 2, 1)
	if err == nil {
		t.Fatal("LinearIntoWithStats accepted invalid weight length")
	}
	_, err = ex.LinearIntoWithStats(context.Background(), []float32{}, []float32{1, 2}, []float32{1, 2}, 1, 2, 1)
	if err == nil {
		t.Fatal("LinearIntoWithStats accepted invalid output length")
	}
}

func TestDynamicLinearRespectsContext(t *testing.T) {
	ex := NewDynamic(Options{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ex.Linear(ctx, []float32{1, 2}, []float32{1, 2}, 1, 2, 1)
	if err == nil {
		t.Fatal("Linear ignored canceled context")
	}
}

func TestDynamicPrepareRejectsBadShapes(t *testing.T) {
	ex := NewDynamic(Options{})
	if err := ex.Prepare(0, 1, 1); err == nil {
		t.Fatal("Prepare accepted invalid shape")
	}
}

func TestDynamicLinearOneHotRejectsBadShapes(t *testing.T) {
	ex := NewDynamic(Options{})
	if _, err := ex.LinearOneHotIOIntoWithStats(context.Background(), make([]float32, 4), []int{0, 1, 2}, 2, 2, 2); err == nil {
		t.Fatal("LinearOneHotIOIntoWithStats accepted oversized batch")
	}
	if _, err := ex.LinearOneHotIOIntoWithStats(context.Background(), make([]float32, 3), []int{0, 1}, 2, 2, 2); err == nil {
		t.Fatal("LinearOneHotIOIntoWithStats accepted short output")
	}
}
