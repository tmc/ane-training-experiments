package dynamicmatmul

import (
	"reflect"
	"testing"
)

func TestPackInput(t *testing.T) {
	const (
		batch  = 2
		inDim  = 2
		outDim = 3
	)
	x := []float32{
		1, 2,
		3, 4,
	}
	w := []float32{
		10, 30, 50,
		20, 40, 60,
	}
	got := make([]float32, inDim*(batch+outDim))
	packInput(got, x, w, batch, inDim, outDim)
	want := []float32{
		1, 3, 10, 30, 50,
		2, 4, 20, 40, 60,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("packed input=%v want %v", got, want)
	}
}

func TestUnpackOutput(t *testing.T) {
	got := make([]float32, 6)
	unpackOutput(got, []float32{
		1, 2,
		3, 4,
		5, 6,
	}, 2, 3)
	want := []float32{
		1, 3, 5,
		2, 4, 6,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unpacked output=%v want %v", got, want)
	}
}

func TestPackInputTile(t *testing.T) {
	const (
		batch      = 2
		inDim      = 2
		fullOutDim = 5
		outOffset  = 1
		tileOutDim = 3
	)
	x := []float32{
		1, 2,
		3, 4,
	}
	w := []float32{
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
	}
	got := make([]float32, inDim*(batch+tileOutDim))
	packInputTile(got, x, w, batch, inDim, fullOutDim, outOffset, tileOutDim)
	want := []float32{
		1, 3, 11, 12, 13,
		2, 4, 21, 22, 23,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("packed tile input=%v want %v", got, want)
	}
}

func TestUnpackOutputTile(t *testing.T) {
	got := make([]float32, 10)
	unpackOutputTile(got, []float32{
		1, 2,
		3, 4,
		5, 6,
	}, 2, 5, 1, 3)
	want := []float32{
		0, 1, 3, 5, 0,
		0, 2, 4, 6, 0,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unpacked tile output=%v want %v", got, want)
	}
}

func TestDefaultTileCandidates(t *testing.T) {
	got := defaultTileCandidates(768)
	want := []int{512, 384, 256, 128, 64}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tile candidates=%v want %v", got, want)
	}
}

func TestStageOneHotActivations(t *testing.T) {
	prev := []int{2, 0, -1}
	cur := []int{1, 2}
	got := make([]float32, 4*(3+2))
	stageOneHotActivations(got, prev, cur, 3, 2)
	if got[2*(3+2)+0] != 0 {
		t.Fatalf("prev slot not cleared")
	}
	if got[0*(3+2)+1] != 0 {
		t.Fatalf("prev slot not cleared")
	}
	if got[1*(3+2)+0] != 1 {
		t.Fatalf("current slot not set")
	}
	if got[2*(3+2)+1] != 1 {
		t.Fatalf("current slot not set")
	}
}

func TestStageChannelFirstActivations(t *testing.T) {
	got := make([]float32, 3*(2+2))
	xCF := []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	stageChannelFirstActivations(got, xCF, 2, 3, 2)
	want := []float32{
		1, 2, 0, 0,
		3, 4, 0, 0,
		5, 6, 0, 0,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("channel-first activations=%v want %v", got, want)
	}
}

func TestCopyOutputTileCF(t *testing.T) {
	got := make([]float32, 10)
	copyOutputTileCF(got, []float32{
		1, 2,
		3, 4,
		5, 6,
	}, 2, 5, 1, 3)
	want := []float32{
		0, 0,
		1, 2,
		3, 4,
		5, 6,
		0, 0,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("channel-first output=%v want %v", got, want)
	}
}

func TestUpdatePrevOneHot(t *testing.T) {
	prev := initPrevOneHot(4)
	updatePrevOneHot(prev, []int{3, 1})
	want := []int{3, 1, -1, -1}
	if !reflect.DeepEqual(prev, want) {
		t.Fatalf("prev=%v want %v", prev, want)
	}
}

func TestTouchedOneHotRows(t *testing.T) {
	got := touchedOneHotRows([]int{2, -1, 4, 2}, []int{1, 4})
	want := []int{2, 4, 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rows=%v want %v", got, want)
	}
}

func TestNewRejectsBadShape(t *testing.T) {
	if _, err := New(0, 1, 1, Options{}); err == nil {
		t.Fatal("New accepted invalid batch")
	}
	if _, err := New(1, 0, 1, Options{}); err == nil {
		t.Fatal("New accepted invalid inDim")
	}
	if _, err := New(1, 1, 0, Options{}); err == nil {
		t.Fatal("New accepted invalid outDim")
	}
}

func TestEvalIntoRejectsBadLengths(t *testing.T) {
	ex := &Executor{batch: 2, inDim: 3, outDim: 4}
	if _, err := ex.EvalInto(make([]float32, 8), make([]float32, 5), make([]float32, 12)); err == nil {
		t.Fatal("EvalInto accepted invalid input length")
	}
	if _, err := ex.EvalInto(make([]float32, 8), make([]float32, 6), make([]float32, 11)); err == nil {
		t.Fatal("EvalInto accepted invalid weight length")
	}
	if _, err := ex.EvalInto(make([]float32, 7), make([]float32, 6), make([]float32, 12)); err == nil {
		t.Fatal("EvalInto accepted invalid output length")
	}
}

func TestEvalIntoRejectsClosedExecutor(t *testing.T) {
	ex := &Executor{
		batch:  2,
		inDim:  2,
		outDim: 2,
	}
	_, err := ex.EvalInto(make([]float32, 4), make([]float32, 4), make([]float32, 4))
	if err == nil {
		t.Fatal("EvalInto accepted closed executor")
	}
}

func TestEvalCFRejectsBadLengths(t *testing.T) {
	ex := &Executor{batch: 2, inDim: 3, outDim: 4}
	if _, err := ex.EvalCF(make([]float32, 5)); err == nil {
		t.Fatal("EvalCF accepted invalid input length")
	}
	if err := ex.ReadOutputCF(make([]float32, 7)); err == nil {
		t.Fatal("ReadOutputCF accepted invalid output length")
	}
}
