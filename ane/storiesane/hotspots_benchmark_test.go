package storiesane

import (
	"fmt"
	"testing"

	"github.com/maderix/ANE/ane/stories"
)

func BenchmarkDirectGoCPUHotspots(b *testing.B) {
	for _, seq := range []int{stories.SeqDefault, 384} {
		seq := seq
		b.Run(fmt.Sprintf("final_head_bundle_seq_%d", seq), func(b *testing.B) {
			fx := newFinalHeadBenchmarkFixture(seq)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(fx.dy)
				clear(fx.dx)
				clear(fx.gRMS)
				clear(fx.gEmbed)
				loss := crossEntropyLossFromProbs(fx.dLogits, fx.probs, fx.targets, stories.Vocab, seq)
				stories.MatMulEmbedT(fx.dy, fx.embed, fx.dLogits, stories.Vocab, stories.Dim, seq)
				stories.MatMulGradEmbed(fx.gEmbed, fx.dLogits, fx.xNorm, stories.Vocab, stories.Dim, seq)
				stories.RMSNormBackward(fx.dx, fx.gRMS, fx.dy, fx.finalHidden, fx.rmsFinal, stories.Dim, seq)
				finalHeadLossSink = loss
			}
		})
		b.Run(fmt.Sprintf("final_head_bundle_compact_9205_seq_%d", seq), func(b *testing.B) {
			fx := newCompactFinalHeadBenchmarkFixture(seq, 9205)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(fx.dLogits)
				clear(fx.dy)
				clear(fx.gRMS)
				clear(fx.gEmbedCompact)
				clear(fx.gEmbedFull)
				clear(fx.dx)
				loss := crossEntropyLossFromProbs(fx.dLogits, fx.probs, fx.targets, fx.compactVocab, seq)
				stories.MatMulEmbedT(fx.dy, fx.embedCompact, fx.dLogits, fx.compactVocab, stories.Dim, seq)
				stories.MatMulGradEmbed(fx.gEmbedCompact, fx.dLogits, fx.xNorm, fx.compactVocab, stories.Dim, seq)
				scatterCompactGrad(fx.gEmbedFull, fx.gEmbedCompact, fx.compactToFull, stories.Dim)
				stories.RMSNormBackward(fx.dx, fx.gRMS, fx.dy, fx.finalHidden, fx.rmsFinal, stories.Dim, seq)
				finalHeadLossSink = loss
			}
		})
		b.Run(fmt.Sprintf("dw_job_ffn_seq_%d", seq), func(b *testing.B) {
			fx := newDWBenchmarkFixture(seq)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(fx.ffnW2)
				clear(fx.ffnW1)
				clear(fx.ffnW3)
				accumLinearGradCF(fx.ffnW2, fx.dOut, fx.gate, stories.Dim, stories.Hidden, seq)
				accumLinearGradCF(fx.ffnW1, fx.dh1, fx.x2Norm, stories.Hidden, stories.Dim, seq)
				accumLinearGradCF(fx.ffnW3, fx.dh3, fx.x2Norm, stories.Hidden, stories.Dim, seq)
			}
		})
		b.Run(fmt.Sprintf("dw_job_attn_out_seq_%d", seq), func(b *testing.B) {
			fx := newDWBenchmarkFixture(seq)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(fx.attnWo)
				accumLinearGradCF(fx.attnWo, fx.dx2, fx.attOut, stories.Dim, stories.Dim, seq)
			}
		})
		b.Run(fmt.Sprintf("dw_job_attn_qkv_seq_%d", seq), func(b *testing.B) {
			fx := newDWBenchmarkFixture(seq)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(fx.attnWq)
				clear(fx.attnWk)
				clear(fx.attnWv)
				accumLinearGradCF(fx.attnWq, fx.dq, fx.xNorm, stories.Dim, stories.Dim, seq)
				accumLinearGradCF(fx.attnWk, fx.dk, fx.xNorm, stories.Dim, stories.Dim, seq)
				accumLinearGradCF(fx.attnWv, fx.dv, fx.xNorm, stories.Dim, stories.Dim, seq)
			}
		})
		b.Run(fmt.Sprintf("rms_dw_final_seq_%d", seq), func(b *testing.B) {
			fx := newFinalHeadBenchmarkFixture(seq)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(fx.gRMS)
				rmsNormGradWeights(fx.gRMS, fx.dy, fx.finalHidden, fx.rmsFinal, stories.Dim, seq)
			}
		})
	}

	b.Run("adam_embed_parallel", func(b *testing.B) {
		w, g, st := newAdamBenchmarkFixture(stories.Vocab * stories.Dim)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			adamUpdateCF(w, g, st, i+1, 3e-4, 0.9, 0.999, 1e-8, 0)
		}
	})
	b.Run("adam_embed_serial", func(b *testing.B) {
		w, g, st := newAdamBenchmarkFixture(stories.Vocab * stories.Dim)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			stories.AdamUpdate(w, g, st, i+1, 3e-4, 0.9, 0.999, 1e-8)
		}
	})
	b.Run("adam_w2_parallel", func(b *testing.B) {
		w, g, st := newAdamBenchmarkFixture(stories.W2Size)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			adamUpdateCF(w, g, st, i+1, 3e-4, 0.9, 0.999, 1e-8, 0)
		}
	})
	b.Run("adam_w2_serial", func(b *testing.B) {
		w, g, st := newAdamBenchmarkFixture(stories.W2Size)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			stories.AdamUpdate(w, g, st, i+1, 3e-4, 0.9, 0.999, 1e-8)
		}
	})
}

var finalHeadLossSink float32

type finalHeadBenchmarkFixture struct {
	probs       []float32
	dLogits     []float32
	targets     []uint16
	embed       []float32
	xNorm       []float32
	finalHidden []float32
	rmsFinal    []float32
	dy          []float32
	dx          []float32
	gRMS        []float32
	gEmbed      []float32
}

type compactFinalHeadBenchmarkFixture struct {
	compactVocab  int
	probs         []float32
	dLogits       []float32
	targets       []uint16
	embedCompact  []float32
	compactToFull []int
	xNorm         []float32
	finalHidden   []float32
	rmsFinal      []float32
	dy            []float32
	dx            []float32
	gRMS          []float32
	gEmbedCompact []float32
	gEmbedFull    []float32
}

func newFinalHeadBenchmarkFixture(seq int) finalHeadBenchmarkFixture {
	fx := finalHeadBenchmarkFixture{
		probs:       makeBenchmarkProbs(stories.Vocab, seq),
		dLogits:     make([]float32, stories.Vocab*seq),
		targets:     makeBenchmarkTargets(stories.Vocab, seq),
		embed:       make([]float32, stories.Vocab*stories.Dim),
		xNorm:       make([]float32, stories.Dim*seq),
		finalHidden: make([]float32, stories.Dim*seq),
		rmsFinal:    make([]float32, stories.Dim),
		dy:          make([]float32, stories.Dim*seq),
		dx:          make([]float32, stories.Dim*seq),
		gRMS:        make([]float32, stories.Dim),
		gEmbed:      make([]float32, stories.Vocab*stories.Dim),
	}
	fillBenchmarkFloats(fx.embed, 0.0002)
	fillBenchmarkFloats(fx.xNorm, 0.01)
	fillBenchmarkFloats(fx.finalHidden, 0.015)
	fillBenchmarkFloats(fx.rmsFinal, 0.5)
	fillBenchmarkFloats(fx.dy, 0.02)
	return fx
}

func newCompactFinalHeadBenchmarkFixture(seq, compactVocab int) compactFinalHeadBenchmarkFixture {
	base := newFinalHeadBenchmarkFixture(seq)
	if compactVocab < 1 || compactVocab > stories.Vocab {
		compactVocab = stories.Vocab
	}
	compactToFull := make([]int, compactVocab)
	for i := range compactToFull {
		compactToFull[i] = i
	}
	targets := make([]uint16, seq)
	for i := range targets {
		targets[i] = uint16(int(base.targets[i]) % compactVocab)
	}
	embedCompact := make([]float32, compactVocab*stories.Dim)
	for c, full := range compactToFull {
		copy(
			embedCompact[c*stories.Dim:(c+1)*stories.Dim],
			base.embed[full*stories.Dim:(full+1)*stories.Dim],
		)
	}
	return compactFinalHeadBenchmarkFixture{
		compactVocab:  compactVocab,
		probs:         makeBenchmarkProbs(compactVocab, seq),
		dLogits:       make([]float32, compactVocab*seq),
		targets:       targets,
		embedCompact:  embedCompact,
		compactToFull: compactToFull,
		xNorm:         base.xNorm,
		finalHidden:   base.finalHidden,
		rmsFinal:      base.rmsFinal,
		dy:            make([]float32, stories.Dim*seq),
		dx:            make([]float32, stories.Dim*seq),
		gRMS:          make([]float32, stories.Dim),
		gEmbedCompact: make([]float32, compactVocab*stories.Dim),
		gEmbedFull:    make([]float32, stories.Vocab*stories.Dim),
	}
}

func scatterCompactGrad(full, compact []float32, compactToFull []int, dim int) {
	for c, fullTok := range compactToFull {
		copy(
			full[fullTok*dim:(fullTok+1)*dim],
			compact[c*dim:(c+1)*dim],
		)
	}
}

type dwBenchmarkFixture struct {
	dOut   []float32
	gate   []float32
	dh1    []float32
	dh3    []float32
	x2Norm []float32
	dx2    []float32
	attOut []float32
	dq     []float32
	dk     []float32
	dv     []float32
	xNorm  []float32
	ffnW2  []float32
	ffnW1  []float32
	ffnW3  []float32
	attnWo []float32
	attnWq []float32
	attnWk []float32
	attnWv []float32
}

func newDWBenchmarkFixture(seq int) dwBenchmarkFixture {
	fx := dwBenchmarkFixture{
		dOut:   make([]float32, stories.Dim*seq),
		gate:   make([]float32, stories.Hidden*seq),
		dh1:    make([]float32, stories.Hidden*seq),
		dh3:    make([]float32, stories.Hidden*seq),
		x2Norm: make([]float32, stories.Dim*seq),
		dx2:    make([]float32, stories.Dim*seq),
		attOut: make([]float32, stories.Dim*seq),
		dq:     make([]float32, stories.Dim*seq),
		dk:     make([]float32, stories.Dim*seq),
		dv:     make([]float32, stories.Dim*seq),
		xNorm:  make([]float32, stories.Dim*seq),
		ffnW2:  make([]float32, stories.W2Size),
		ffnW1:  make([]float32, stories.W1Size),
		ffnW3:  make([]float32, stories.W3Size),
		attnWo: make([]float32, stories.WOSize),
		attnWq: make([]float32, stories.WQSize),
		attnWk: make([]float32, stories.WQSize),
		attnWv: make([]float32, stories.WQSize),
	}
	fillBenchmarkFloats(fx.dOut, 0.02)
	fillBenchmarkFloats(fx.gate, 0.015)
	fillBenchmarkFloats(fx.dh1, 0.017)
	fillBenchmarkFloats(fx.dh3, 0.019)
	fillBenchmarkFloats(fx.x2Norm, 0.01)
	fillBenchmarkFloats(fx.dx2, 0.013)
	fillBenchmarkFloats(fx.attOut, 0.009)
	fillBenchmarkFloats(fx.dq, 0.011)
	fillBenchmarkFloats(fx.dk, 0.012)
	fillBenchmarkFloats(fx.dv, 0.014)
	fillBenchmarkFloats(fx.xNorm, 0.008)
	return fx
}

func newAdamBenchmarkFixture(n int) ([]float32, []float32, *stories.AdamState) {
	w := make([]float32, n)
	g := make([]float32, n)
	fillBenchmarkFloats(w, 0.01)
	fillBenchmarkFloats(g, 0.001)
	st := &stories.AdamState{
		M: make([]float32, n),
		V: make([]float32, n),
	}
	return w, g, st
}

func fillBenchmarkFloats(dst []float32, scale float32) {
	for i := range dst {
		dst[i] = scale * float32((i%23)-11)
	}
}
