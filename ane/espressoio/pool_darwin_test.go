//go:build darwin

package espressoio

import "testing"

func TestPoolF32RoundTrip(t *testing.T) {
	p, err := Open(16, 1)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer p.Close()

	in := []float32{1, 2, 3, 4}
	if err := p.WriteFrameF32(0, in); err != nil {
		t.Fatalf("WriteFrameF32: %v", err)
	}

	out := make([]float32, 4)
	if err := p.ReadFrameF32(0, out); err != nil {
		t.Fatalf("ReadFrameF32: %v", err)
	}
	for i, want := range in {
		if got := out[i]; got != want {
			t.Fatalf("out[%d]=%v want %v", i, got, want)
		}
	}
}

func TestPoolF32LengthValidation(t *testing.T) {
	p, err := Open(16, 1)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer p.Close()

	if err := p.WriteFrameF32(0, []float32{1, 2, 3}); err == nil {
		t.Fatalf("WriteFrameF32 short input succeeded; want error")
	}
	if err := p.ReadFrameF32(0, make([]float32, 3)); err == nil {
		t.Fatalf("ReadFrameF32 short output succeeded; want error")
	}
}
