package stories

import "testing"

func TestCrossEntropyLossShape(t *testing.T) {
	v, s := 4, 3
	logits := []float32{
		2, 0, -1,
		0, 1, 0,
		-1, 0, 2,
		0, 0, 0,
	}
	d := make([]float32, len(logits))
	tgt := []uint16{0, 1, 2}
	loss := CrossEntropyLoss(d, logits, tgt, v, s)
	if loss <= 0 {
		t.Fatalf("loss=%v want > 0", loss)
	}
	for tpos := 0; tpos < s; tpos++ {
		sum := float32(0)
		for i := 0; i < v; i++ {
			sum += d[i*s+tpos]
		}
		if sum < -1e-5 || sum > 1e-5 {
			t.Fatalf("grad col %d sum=%g want ~0", tpos, sum)
		}
	}
}

func TestEmbedLookupBackward(t *testing.T) {
	dim, seq, vocab := 3, 2, 5
	embed := make([]float32, vocab*dim)
	for i := range embed {
		embed[i] = float32(i + 1)
	}
	toks := []uint16{1, 3}
	x := make([]float32, dim*seq)
	EmbedLookup(x, embed, toks, dim, seq)
	if x[0] != embed[1*dim+0] || x[1] != embed[3*dim+0] {
		t.Fatalf("lookup mismatch")
	}
	dx := []float32{1, 2, 3, 4, 5, 6}
	dEmbed := make([]float32, len(embed))
	EmbedBackward(dEmbed, dx, toks, dim, seq)
	if dEmbed[1*dim+0] != 1 || dEmbed[1*dim+1] != 3 || dEmbed[1*dim+2] != 5 {
		t.Fatalf("backward token 1 mismatch: %v", dEmbed[1*dim:1*dim+dim])
	}
	if dEmbed[3*dim+0] != 2 || dEmbed[3*dim+1] != 4 || dEmbed[3*dim+2] != 6 {
		t.Fatalf("backward token 3 mismatch: %v", dEmbed[3*dim:3*dim+dim])
	}
}

func TestEmbedLookupOutOfRangeTokenZeroesColumn(t *testing.T) {
	dim, seq := 3, 2
	embed := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	// token 5 is out of range for vocab=2.
	toks := []uint16{1, 5}
	out := make([]float32, dim*seq)
	EmbedLookup(out, embed, toks, dim, seq)
	for d := 0; d < dim; d++ {
		if got := out[d*seq+0]; got != embed[dim+d] {
			t.Fatalf("col0 dim%d=%v want %v", d, got, embed[dim+d])
		}
		if got := out[d*seq+1]; got != 0 {
			t.Fatalf("col1 dim%d=%v want 0", d, got)
		}
	}
}

func TestCrossEntropyLossOutOfRangeTargetSkipsColumn(t *testing.T) {
	v, s := 3, 2
	logits := []float32{
		2, 0,
		0, 1,
		-1, 0,
	}
	d := make([]float32, len(logits))
	// second target is out of range.
	targets := []uint16{1, 99}
	loss := CrossEntropyLoss(d, logits, targets, v, s)
	if loss <= 0 {
		t.Fatalf("loss=%v want > 0 from valid column", loss)
	}
	// Invalid target column gradient must be all zeros.
	for i := 0; i < v; i++ {
		if got := d[i*s+1]; got != 0 {
			t.Fatalf("invalid col grad[%d]=%v want 0", i, got)
		}
	}
	// Valid target column should still have zero-sum gradient.
	sum := float32(0)
	for i := 0; i < v; i++ {
		sum += d[i*s+0]
	}
	if sum < -1e-5 || sum > 1e-5 {
		t.Fatalf("valid col sum=%g want ~0", sum)
	}
}
