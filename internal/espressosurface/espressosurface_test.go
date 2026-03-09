//go:build darwin

package espressosurface

import "testing"

func TestOpenValidation(t *testing.T) {
	tests := []struct {
		name   string
		bytes  int
		frames uint64
	}{
		{name: "zero bytes", bytes: 0, frames: 1},
		{name: "negative bytes", bytes: -1, frames: 1},
		{name: "zero frames", bytes: 16, frames: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := Open(tt.bytes, tt.frames)
			if err == nil {
				if s != nil {
					s.Cleanup()
				}
				t.Fatalf("Open(%d, %d) succeeded; want error", tt.bytes, tt.frames)
			}
		})
	}
}
