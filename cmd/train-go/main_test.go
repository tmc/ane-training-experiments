package main

import (
	"math/rand"
	"testing"
)

func TestCPULogitsInto(t *testing.T) {
	p := cpuLogitsProvider{}
	w := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	dst := make([]float32, 6)
	st, err := p.LogitsInto(dst, w, []int{2, 0}, 3)
	if err != nil {
		t.Fatalf("LogitsInto: %v", err)
	}
	_ = st
	want := []float32{7, 8, 9, 1, 2, 3}
	for i, v := range want {
		if dst[i] != v {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], v)
		}
	}
	if _, err := p.LogitsInto(dst[:5], w, []int{2, 0}, 3); err == nil {
		t.Fatalf("LogitsInto short dst succeeded; want error")
	}
}

func TestSampleBatchIntoReusesBuffers(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tokens := []uint16{1, 2, 3, 4, 5, 6}
	xs0 := make([]int, 0, 8)
	ys0 := make([]int, 0, 8)
	xs, ys := sampleBatchInto(xs0, ys0, tokens, 4, false, 10, rng)
	if len(xs) != 4 || len(ys) != 4 {
		t.Fatalf("got lens (%d,%d) want (4,4)", len(xs), len(ys))
	}
	if cap(xs) != cap(xs0) || cap(ys) != cap(ys0) {
		t.Fatalf("sampleBatchInto replaced reusable buffers")
	}
}

func TestGrowFloat32ReusesCapacity(t *testing.T) {
	buf := make([]float32, 2, 8)
	got := growFloat32(buf, 6)
	if len(got) != 6 {
		t.Fatalf("len=%d want 6", len(got))
	}
	if cap(got) != 8 {
		t.Fatalf("cap=%d want 8", cap(got))
	}
	got = growFloat32(got, 16)
	if len(got) != 16 {
		t.Fatalf("len=%d want 16", len(got))
	}
	if cap(got) < 16 {
		t.Fatalf("cap=%d want >=16", cap(got))
	}
}

func TestAppendUniqueInt(t *testing.T) {
	got := appendUniqueInt([]int{1, 3}, 2)
	got = appendUniqueInt(got, 3)
	want := []int{1, 3, 2}
	if len(got) != len(want) {
		t.Fatalf("len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%d want=%d", i, got[i], want[i])
		}
	}
}

func TestUpdateWeightsRowMajorRowsToIO(t *testing.T) {
	src := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	dst := make([]float32, len(src))
	transposeWeightsRowMajorOIToIO(dst, src, 3, 3)

	src[1*3+0] = 40
	src[1*3+1] = 50
	src[1*3+2] = 60
	updateWeightsRowMajorRowsToIO(dst, src, []int{1}, 3)

	want := []float32{
		1, 40, 7,
		2, 50, 8,
		3, 60, 9,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], want[i])
		}
	}
}
