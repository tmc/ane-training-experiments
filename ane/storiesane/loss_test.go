package storiesane

import (
	"math"
	"slices"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

func BenchmarkCrossEntropyLossFromProbs(b *testing.B) {
	const (
		vocab = stories.Vocab
		seq   = stories.SeqDefault
	)
	probs := makeBenchmarkProbs(vocab, seq)
	targets := makeBenchmarkTargets(vocab, seq)
	dLogits := make([]float32, len(probs))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = crossEntropyLossFromProbs(dLogits, probs, targets, vocab, seq)
	}
}

func BenchmarkCrossEntropyLossFromProbsSerial(b *testing.B) {
	const (
		vocab = stories.Vocab
		seq   = stories.SeqDefault
	)
	probs := makeBenchmarkProbs(vocab, seq)
	targets := makeBenchmarkTargets(vocab, seq)
	dLogits := make([]float32, len(probs))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = crossEntropyLossFromProbsSerial(dLogits, probs, targets, vocab, seq)
	}
}

func BenchmarkCrossEntropyLossCPU(b *testing.B) {
	const (
		vocab = stories.Vocab
		seq   = stories.SeqDefault
	)
	logits := makeBenchmarkLogits(vocab, seq)
	targets := makeBenchmarkTargets(vocab, seq)
	dLogits := make([]float32, len(logits))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = stories.CrossEntropyLoss(dLogits, logits, targets, vocab, seq)
	}
}

func TestCrossEntropyLossFromProbsInPlaceMatchesCopy(t *testing.T) {
	const (
		vocab = 257
		seq   = 23
	)
	probs := makeBenchmarkProbs(vocab, seq)
	targets := makeBenchmarkTargets(vocab, seq)
	wantGrad := make([]float32, len(probs))
	wantLoss := crossEntropyLossFromProbsSerial(wantGrad, probs, targets, vocab, seq)

	gotGrad := slices.Clone(probs)
	gotLoss := crossEntropyLossFromProbs(gotGrad, gotGrad, targets, vocab, seq)

	if math.Abs(float64(gotLoss-wantLoss)) > 1e-6 {
		t.Fatalf("loss mismatch: got %.8f want %.8f", gotLoss, wantLoss)
	}
	for i := range gotGrad {
		if math.Abs(float64(gotGrad[i]-wantGrad[i])) > 1e-5 {
			t.Fatalf("grad[%d]=%.8f want %.8f", i, gotGrad[i], wantGrad[i])
		}
	}
}

func TestCrossEntropyLossFromProbsUnscaledMatchesScaledAfterMean(t *testing.T) {
	const (
		vocab = 257
		seq   = 23
	)
	probs := makeBenchmarkProbs(vocab, seq)
	targets := makeBenchmarkTargets(vocab, seq)
	wantGrad := make([]float32, len(probs))
	wantLoss := crossEntropyLossFromProbsSerial(wantGrad, probs, targets, vocab, seq)

	gotGrad := slices.Clone(probs)
	gotLoss, valid := crossEntropyLossFromProbsUnscaled(gotGrad, gotGrad, targets, vocab, seq)
	if valid != seq {
		t.Fatalf("valid=%d want %d", valid, seq)
	}
	scale := float32(1.0 / float64(valid))
	for i := range gotGrad {
		gotGrad[i] *= scale
	}

	if math.Abs(float64(gotLoss-wantLoss)) > 1e-6 {
		t.Fatalf("loss mismatch: got %.8f want %.8f", gotLoss, wantLoss)
	}
	for i := range gotGrad {
		if math.Abs(float64(gotGrad[i]-wantGrad[i])) > 1e-5 {
			t.Fatalf("grad[%d]=%.8f want %.8f", i, gotGrad[i], wantGrad[i])
		}
	}
}

func makeBenchmarkLogits(vocab, seq int) []float32 {
	logits := make([]float32, vocab*seq)
	for v := 0; v < vocab; v++ {
		for t := 0; t < seq; t++ {
			logits[v*seq+t] = float32(math.Sin(float64(v*seq+t)) * 0.1)
		}
	}
	return logits
}

func makeBenchmarkProbs(vocab, seq int) []float32 {
	logits := makeBenchmarkLogits(vocab, seq)
	probs := make([]float32, len(logits))
	for t := 0; t < seq; t++ {
		maxv := logits[t]
		for v := 1; v < vocab; v++ {
			if x := logits[v*seq+t]; x > maxv {
				maxv = x
			}
		}
		sum := 0.0
		for v := 0; v < vocab; v++ {
			e := math.Exp(float64(logits[v*seq+t] - maxv))
			probs[v*seq+t] = float32(e)
			sum += e
		}
		inv := float32(1.0 / sum)
		for v := 0; v < vocab; v++ {
			probs[v*seq+t] *= inv
		}
	}
	return probs
}

func makeBenchmarkTargets(vocab, seq int) []uint16 {
	targets := make([]uint16, seq)
	for t := range targets {
		targets[t] = uint16((t * 997) % vocab)
	}
	return targets
}

func crossEntropyLossFromProbsSerial(dLogits, probs []float32, targets []uint16, vocab, seq int) float32 {
	if vocab <= 0 || seq <= 0 {
		return 0
	}
	if len(probs) < vocab*seq || len(dLogits) < vocab*seq || len(targets) < seq {
		for i := range dLogits {
			dLogits[i] = 0
		}
		return 0
	}
	copy(dLogits[:vocab*seq], probs[:vocab*seq])
	loss := 0.0
	valid := 0
	for t := 0; t < seq; t++ {
		tgt := int(targets[t])
		if tgt < 0 || tgt >= vocab {
			for v := 0; v < vocab; v++ {
				dLogits[v*seq+t] = 0
			}
			continue
		}
		p := probs[tgt*seq+t]
		if p < 1e-10 {
			p = 1e-10
		}
		loss -= math.Log(float64(p))
		dLogits[tgt*seq+t] -= 1
		valid++
	}
	if valid == 0 {
		return 0
	}
	scale := float32(1.0 / float64(valid))
	for t := 0; t < seq; t++ {
		tgt := int(targets[t])
		if tgt < 0 || tgt >= vocab {
			continue
		}
		for v := 0; v < vocab; v++ {
			dLogits[v*seq+t] *= scale
		}
	}
	return float32(loss / float64(valid))
}
