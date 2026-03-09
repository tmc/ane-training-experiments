//go:build darwin

package iosurface

import "testing"

func TestSurfaceF32RoundTrip(t *testing.T) {
	s, err := Create(4 * 4)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer s.Close()

	in := []float32{1.25, -3.5, 0, 42}
	if err := s.WriteF32(in); err != nil {
		t.Fatalf("WriteF32: %v", err)
	}

	out := make([]float32, len(in))
	if err := s.ReadF32(out); err != nil {
		t.Fatalf("ReadF32: %v", err)
	}

	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], in[i])
		}
	}
}

func TestSurfaceF32LengthValidation(t *testing.T) {
	s, err := Create(4 * 4)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer s.Close()

	if err := s.WriteF32([]float32{1, 2, 3}); err == nil {
		t.Fatalf("WriteF32 short input succeeded; want error")
	}
	if err := s.ReadF32(make([]float32, 3)); err == nil {
		t.Fatalf("ReadF32 short output succeeded; want error")
	}
}
