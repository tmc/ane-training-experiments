package kernelio

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/maderix/ANE/internal/evalbuffer"
)

type fakeKernel struct {
	writeBytes []byte
	writeF32   []float32
	readBytes  []byte
	readF32    []float32
	writeErr   error
	readErr    error
	writeByteN int
	writeF32N  int
	readByteN  int
	readF32N   int
}

func (k *fakeKernel) WriteInput(_ int, data []byte) error {
	k.writeByteN++
	if k.writeErr != nil {
		return k.writeErr
	}
	k.writeBytes = append(k.writeBytes[:0], data...)
	return nil
}

func (k *fakeKernel) ReadOutput(_ int, data []byte) error {
	k.readByteN++
	if k.readErr != nil {
		return k.readErr
	}
	copy(data, k.readBytes)
	return nil
}

func (k *fakeKernel) WriteInputF32(_ int, data []float32) error {
	k.writeF32N++
	if k.writeErr != nil {
		return k.writeErr
	}
	k.writeF32 = append(k.writeF32[:0], data...)
	return nil
}

func (k *fakeKernel) ReadOutputF32(_ int, data []float32) error {
	k.readF32N++
	if k.readErr != nil {
		return k.readErr
	}
	copy(data, k.readF32)
	return nil
}

func TestWriteBytes(t *testing.T) {
	buf := evalbuffer.New(8, 8)
	if err := buf.StageBytes([]byte{1, 2, 3}); err != nil {
		t.Fatalf("StageBytes: %v", err)
	}
	k := &fakeKernel{}
	if err := WriteBytes(k, buf, "op"); err != nil {
		t.Fatalf("WriteBytes: %v", err)
	}
	if k.writeByteN != 1 || k.writeF32N != 0 {
		t.Fatalf("write calls bytes=%d f32=%d, want 1/0", k.writeByteN, k.writeF32N)
	}
	if got := k.writeBytes; len(got) != 8 || got[0] != 1 || got[1] != 2 || got[2] != 3 || got[3] != 0 {
		t.Fatalf("write bytes=%v", got)
	}
}

func TestReadBytes(t *testing.T) {
	buf := evalbuffer.New(8, 8)
	k := &fakeKernel{readBytes: []byte{9, 8, 7, 6}}
	if err := ReadBytes(k, buf, "op"); err != nil {
		t.Fatalf("ReadBytes: %v", err)
	}
	out := make([]byte, 4)
	if err := buf.CopyBytes(out); err != nil {
		t.Fatalf("CopyBytes: %v", err)
	}
	if want := []byte{9, 8, 7, 6}; string(out) != string(want) {
		t.Fatalf("bytes=%v want %v", out, want)
	}
}

func TestWriteF32(t *testing.T) {
	tests := []struct {
		name        string
		inputBytes  int
		outputBytes int
		wantBytes   bool
	}{
		{name: "typed", inputBytes: 8, outputBytes: 8, wantBytes: false},
		{name: "packed", inputBytes: 6, outputBytes: 6, wantBytes: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := evalbuffer.New(tt.inputBytes, tt.outputBytes)
			if err := buf.StageF32([]float32{1.5}); err != nil {
				t.Fatalf("StageF32: %v", err)
			}
			k := &fakeKernel{}
			if err := WriteF32(k, buf, "op"); err != nil {
				t.Fatalf("WriteF32: %v", err)
			}
			if tt.wantBytes {
				if k.writeByteN != 1 || k.writeF32N != 0 {
					t.Fatalf("write calls bytes=%d f32=%d, want 1/0", k.writeByteN, k.writeF32N)
				}
				got := uint32(k.writeBytes[0]) | uint32(k.writeBytes[1])<<8 | uint32(k.writeBytes[2])<<16 | uint32(k.writeBytes[3])<<24
				if got != math.Float32bits(1.5) {
					t.Fatalf("packed bits=%08x want %08x", got, math.Float32bits(1.5))
				}
				return
			}
			if k.writeF32N != 1 || k.writeByteN != 0 {
				t.Fatalf("write calls bytes=%d f32=%d, want 0/1", k.writeByteN, k.writeF32N)
			}
			if got := k.writeF32; len(got) != 2 || got[0] != 1.5 || got[1] != 0 {
				t.Fatalf("write f32=%v", got)
			}
		})
	}
}

func TestReadF32(t *testing.T) {
	tests := []struct {
		name        string
		inputBytes  int
		outputBytes int
		kernel      *fakeKernel
		want        []float32
	}{
		{
			name:        "typed",
			inputBytes:  8,
			outputBytes: 8,
			kernel:      &fakeKernel{readF32: []float32{2.5, -1}},
			want:        []float32{2.5},
		},
		{
			name:        "packed",
			inputBytes:  6,
			outputBytes: 6,
			kernel: &fakeKernel{readBytes: []byte{
				byte(math.Float32bits(3.25)),
				byte(math.Float32bits(3.25) >> 8),
				byte(math.Float32bits(3.25) >> 16),
				byte(math.Float32bits(3.25) >> 24),
			}},
			want: []float32{3.25},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := evalbuffer.New(tt.inputBytes, tt.outputBytes)
			got := make([]float32, len(tt.want))
			if err := ReadF32(tt.kernel, buf, got, "op"); err != nil {
				t.Fatalf("ReadF32: %v", err)
			}
			if len(got) != len(tt.want) || got[0] != tt.want[0] {
				t.Fatalf("output=%v want %v", got, tt.want)
			}
		})
	}
}

func TestErrorWrapping(t *testing.T) {
	buf := evalbuffer.New(8, 8)
	if err := buf.StageF32([]float32{1}); err != nil {
		t.Fatalf("StageF32: %v", err)
	}
	want := errors.New("boom")
	k := &fakeKernel{writeErr: want}
	err := WriteF32(k, buf, "op")
	if err == nil {
		t.Fatalf("WriteF32 error=nil, want error")
	}
	if !errors.Is(err, want) {
		t.Fatalf("errors.Is(%v, %v)=false", err, want)
	}
	if !strings.Contains(err.Error(), "op: write input") {
		t.Fatalf("error=%q missing context", err)
	}
}
