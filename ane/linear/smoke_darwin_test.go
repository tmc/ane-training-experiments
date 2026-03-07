//go:build darwin

package linear

import (
	"context"
	"math"
	"os"
	"testing"
)

func TestSmokeLinearIdentity(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run ANE linear smoke test")
	}

	ex := New(Options{})
	defer ex.Close()

	x := []float32{
		1, 2,
		3, 4,
	}
	w := []float32{
		1, 0,
		0, 1,
	}
	y, err := ex.Linear(context.Background(), x, w, 2, 2, 2)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	if len(y) != len(x) {
		t.Fatalf("output len=%d want=%d", len(y), len(x))
	}
	for i := range x {
		if math.Abs(float64(y[i]-x[i])) > 0.05 {
			t.Fatalf("y[%d]=%g want=%g", i, y[i], x[i])
		}
	}
}
