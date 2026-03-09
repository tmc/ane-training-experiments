package evalbuffer

import "testing"

func TestStageBytesZeroFillsTail(t *testing.T) {
	b := New(8, 8)
	if err := b.StageBytes([]byte{1, 2, 3}); err != nil {
		t.Fatalf("StageBytes: %v", err)
	}
	want := []byte{1, 2, 3, 0, 0, 0, 0, 0}
	if got := b.InputBytesScratch(); len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	} else {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("buf[%d]=%d want %d", i, got[i], want[i])
			}
		}
	}
}

func TestStageF32FallbackAndDecode(t *testing.T) {
	b := New(6, 6)
	if b.TypedF32() {
		t.Fatalf("TypedF32=true want false")
	}
	in := []float32{1.5}
	if err := b.StageF32(in); err != nil {
		t.Fatalf("StageF32: %v", err)
	}
	copy(b.OutputBytesScratch(), b.InputBytesScratch())
	out := make([]float32, len(in))
	if err := b.DecodeF32(out); err != nil {
		t.Fatalf("DecodeF32: %v", err)
	}
	if out[0] != in[0] {
		t.Fatalf("out[0]=%v want %v", out[0], in[0])
	}
}

func TestStageF32Typed(t *testing.T) {
	b := New(8, 8)
	if !b.TypedF32() {
		t.Fatalf("TypedF32=false want true")
	}
	in := []float32{1, 2}
	if err := b.StageF32(in[:1]); err != nil {
		t.Fatalf("StageF32: %v", err)
	}
	got := b.InputF32Scratch()
	if got[0] != 1 || got[1] != 0 {
		t.Fatalf("input=%v want [1 0]", got)
	}
	copy(b.OutputF32Scratch(), in)
	out := make([]float32, 2)
	if err := b.DecodeF32(out); err != nil {
		t.Fatalf("DecodeF32: %v", err)
	}
	if out[0] != 1 || out[1] != 2 {
		t.Fatalf("output=%v want [1 2]", out)
	}
}
