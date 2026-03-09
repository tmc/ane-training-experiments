//go:build darwin

package linear

import (
	"context"
	"math"
	"os"
	"testing"
)

func TestSmokeDynamicLinearIdentityAndScale(t *testing.T) {
	if os.Getenv("ANE_SMOKE") != "1" {
		t.Skip("set ANE_SMOKE=1 to run ANE linear smoke test")
	}

	ex := NewDynamic(Options{})
	defer ex.Close()

	x := []float32{
		1, 2,
		3, 4,
	}
	identity := []float32{
		1, 0,
		0, 1,
	}
	y, st, err := ex.LinearWithStats(context.Background(), x, identity, 2, 2, 2)
	if err != nil {
		t.Fatalf("LinearWithStats(identity): %v", err)
	}
	if !st.Compiled {
		t.Fatalf("first dynamic call did not report compile")
	}
	for i := range x {
		if math.Abs(float64(y[i]-x[i])) > 0.05 {
			t.Fatalf("identity y[%d]=%g want=%g", i, y[i], x[i])
		}
	}

	scaled := []float32{
		2, 0,
		0, 2,
	}
	y, st, err = ex.LinearWithStats(context.Background(), x, scaled, 2, 2, 2)
	if err != nil {
		t.Fatalf("LinearWithStats(scale): %v", err)
	}
	if st.Compiled {
		t.Fatalf("second dynamic call unexpectedly recompiled")
	}
	want := []float32{2, 4, 6, 8}
	for i := range want {
		if math.Abs(float64(y[i]-want[i])) > 0.1 {
			t.Fatalf("scaled y[%d]=%g want=%g", i, y[i], want[i])
		}
	}
	if got := ex.Stats().Compiles; got != 1 {
		t.Fatalf("dynamic compiles=%d want 1", got)
	}
}
