package storiesane

import "testing"

func TestAccumLinearGradCF(t *testing.T) {
	outCh, inCh, seq := 2, 3, 4
	dy := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	x := []float32{
		2, 1, 0, -1,
		3, 2, 1, 0,
		4, 3, 2, 1,
	}
	got := []float32{
		10, 11, 12,
		13, 14, 15,
	}
	want := append([]float32(nil), got...)
	for o := 0; o < outCh; o++ {
		for i := 0; i < inCh; i++ {
			sum := float32(0)
			for tpos := 0; tpos < seq; tpos++ {
				sum += dy[o*seq+tpos] * x[i*seq+tpos]
			}
			want[o*inCh+i] += sum
		}
	}

	accumLinearGradCF(got, dy, x, outCh, inCh, seq)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dW[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestLinearBackwardDXCF(t *testing.T) {
	outCh, inCh, seq := 2, 3, 4
	w := []float32{
		2, 3, 4,
		5, 6, 7,
	}
	dy := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	got := make([]float32, inCh*seq)
	want := make([]float32, inCh*seq)
	for i := 0; i < inCh; i++ {
		for tpos := 0; tpos < seq; tpos++ {
			sum := float32(0)
			for o := 0; o < outCh; o++ {
				sum += w[o*inCh+i] * dy[o*seq+tpos]
			}
			want[i*seq+tpos] = sum
		}
	}

	linearBackwardDXCF(got, w, dy, outCh, inCh, seq)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dx[%d]=%v want %v", i, got[i], want[i])
		}
	}
}
