package storiesane

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/maderix/ANE/ane/stories"
)

type layerCache struct {
	x      []float32
	xNorm  []float32
	q      []float32
	k      []float32
	v      []float32
	attOut []float32
	x2     []float32
	x2Norm []float32
	h1     []float32
	h3     []float32
	gate   []float32
	dOut   []float32
	dh1    []float32
	dh3    []float32
	dx2    []float32
	dq     []float32
	dk     []float32
	dv     []float32

	attTapsReady bool
	ffnTapsReady bool
}

type modelGrad struct {
	Layers   []stories.LayerWeights
	RMSFinal []float32
	Embed    []float32
}

func newLayerCache(seq int) layerCache {
	return layerCache{
		x:      make([]float32, stories.Dim*seq),
		xNorm:  make([]float32, stories.Dim*seq),
		q:      make([]float32, stories.Dim*seq),
		k:      make([]float32, stories.Dim*seq),
		v:      make([]float32, stories.Dim*seq),
		attOut: make([]float32, stories.Dim*seq),
		x2:     make([]float32, stories.Dim*seq),
		x2Norm: make([]float32, stories.Dim*seq),
		h1:     make([]float32, stories.Hidden*seq),
		h3:     make([]float32, stories.Hidden*seq),
		gate:   make([]float32, stories.Hidden*seq),
		dOut:   make([]float32, stories.Dim*seq),
		dh1:    make([]float32, stories.Hidden*seq),
		dh3:    make([]float32, stories.Hidden*seq),
		dx2:    make([]float32, stories.Dim*seq),
		dq:     make([]float32, stories.Dim*seq),
		dk:     make([]float32, stories.Dim*seq),
		dv:     make([]float32, stories.Dim*seq),
	}
}

func newLayerGrad() stories.LayerWeights {
	return stories.LayerWeights{
		Wq:     make([]float32, stories.WQSize),
		Wk:     make([]float32, stories.WQSize),
		Wv:     make([]float32, stories.WQSize),
		Wo:     make([]float32, stories.WOSize),
		W1:     make([]float32, stories.W1Size),
		W2:     make([]float32, stories.W2Size),
		W3:     make([]float32, stories.W3Size),
		RMSAtt: make([]float32, stories.Dim),
		RMSFFN: make([]float32, stories.Dim),
	}
}

func newModelGrad(vocab int) *modelGrad {
	g := &modelGrad{
		Layers:   make([]stories.LayerWeights, stories.NLayers),
		RMSFinal: make([]float32, stories.Dim),
		Embed:    make([]float32, vocab*stories.Dim),
	}
	for i := range g.Layers {
		g.Layers[i] = newLayerGrad()
	}
	return g
}

func clearLayerGrad(g *stories.LayerWeights) {
	clear(g.Wq)
	clear(g.Wk)
	clear(g.Wv)
	clear(g.Wo)
	clear(g.W1)
	clear(g.W2)
	clear(g.W3)
	clear(g.RMSAtt)
	clear(g.RMSFFN)
}

func clearModelGrad(g *modelGrad) {
	if g == nil {
		return
	}
	for i := range g.Layers {
		clearLayerGrad(&g.Layers[i])
	}
	clear(g.RMSFinal)
	clear(g.Embed)
}

func scaleLayerGrad(g *stories.LayerWeights, scale float32) {
	scaleSlice(g.Wq, scale)
	scaleSlice(g.Wk, scale)
	scaleSlice(g.Wv, scale)
	scaleSlice(g.Wo, scale)
	scaleSlice(g.W1, scale)
	scaleSlice(g.W2, scale)
	scaleSlice(g.W3, scale)
	scaleSlice(g.RMSAtt, scale)
	scaleSlice(g.RMSFFN, scale)
}

func scaleModelGrad(g *modelGrad, scale float32) {
	for i := range g.Layers {
		scaleLayerGrad(&g.Layers[i], scale)
	}
	scaleSlice(g.RMSFinal, scale)
	scaleSlice(g.Embed, scale)
}

func scaleSlice(v []float32, scale float32) {
	for i := range v {
		v[i] *= scale
	}
}

func addLayerGrad(dst, src *stories.LayerWeights) {
	addSlice(dst.Wq, src.Wq)
	addSlice(dst.Wk, src.Wk)
	addSlice(dst.Wv, src.Wv)
	addSlice(dst.Wo, src.Wo)
	addSlice(dst.W1, src.W1)
	addSlice(dst.W2, src.W2)
	addSlice(dst.W3, src.W3)
	addSlice(dst.RMSAtt, src.RMSAtt)
	addSlice(dst.RMSFFN, src.RMSFFN)
}

func addSlice(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func accumLinearGradCF(dW, dy, x []float32, outCh, inCh, seq int) {
	if accumLinearGradCFAccelerate(dW, dy, x, outCh, inCh, seq) {
		return
	}
	for o := 0; o < outCh; o++ {
		row := dW[o*inCh : (o+1)*inCh]
		dyRow := dy[o*seq : (o+1)*seq]
		for i := 0; i < inCh; i++ {
			xRow := x[i*seq : (i+1)*seq]
			sum := float32(0)
			for t := 0; t < seq; t++ {
				sum += dyRow[t] * xRow[t]
			}
			row[i] += sum
		}
	}
}

func linearBackwardDXCF(dx, w, dy []float32, outCh, inCh, seq int) {
	if linearBackwardDXCFAccelerate(dx, w, dy, outCh, inCh, seq) {
		return
	}
	for i := 0; i < inCh; i++ {
		row := dx[i*seq : (i+1)*seq]
		for t := range row {
			row[t] = 0
		}
		for o := 0; o < outCh; o++ {
			weight := w[o*inCh+i]
			dyRow := dy[o*seq : (o+1)*seq]
			for t := 0; t < seq; t++ {
				row[t] += weight * dyRow[t]
			}
		}
	}
}

func causalAttentionBackwardCF(dq, dk, dv, dOut, q, k, v []float32, heads, headDim, seq int) {
	clear(dq)
	clear(dk)
	clear(dv)
	scores := make([]float32, seq)
	probs := make([]float32, seq)
	dScores := make([]float32, seq)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := 0; h < heads; h++ {
		base := h * headDim
		for t := 0; t < seq; t++ {
			maxv := float32(math.Inf(-1))
			for s := 0; s <= t; s++ {
				sum := float32(0)
				for i := 0; i < headDim; i++ {
					sum += q[(base+i)*seq+t] * k[(base+i)*seq+s]
				}
				scores[s] = sum * scale
				if scores[s] > maxv {
					maxv = scores[s]
				}
			}
			total := float64(0)
			for s := 0; s <= t; s++ {
				e := math.Exp(float64(scores[s] - maxv))
				probs[s] = float32(e)
				total += e
			}
			invTotal := float32(1.0 / total)
			for s := 0; s <= t; s++ {
				probs[s] *= invTotal
			}

			dsSum := float32(0)
			for s := 0; s <= t; s++ {
				dot := float32(0)
				for i := 0; i < headDim; i++ {
					dot += dOut[(base+i)*seq+t] * v[(base+i)*seq+s]
				}
				dScores[s] = dot
				dsSum += probs[s] * dot
			}

			for s := 0; s <= t; s++ {
				ds := probs[s] * (dScores[s] - dsSum) * scale
				for i := 0; i < headDim; i++ {
					qIdx := (base + i) * seq
					dq[qIdx+t] += ds * k[qIdx+s]
					dk[qIdx+s] += ds * q[qIdx+t]
					dv[qIdx+s] += probs[s] * dOut[qIdx+t]
				}
			}
		}
	}
}

func sampleFromLogits(rng *rand.Rand, logits []float32, temperature float32) int {
	if temperature <= 1e-6 {
		return argmax(logits)
	}
	maxv := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxv {
			maxv = logits[i]
		}
	}
	invTemp := float64(1 / temperature)
	total := 0.0
	for i := range logits {
		total += math.Exp((float64(logits[i]) - float64(maxv)) * invTemp)
	}
	target := rng.Float64() * total
	acc := 0.0
	for i := range logits {
		acc += math.Exp((float64(logits[i]) - float64(maxv)) * invTemp)
		if acc >= target {
			return i
		}
	}
	return len(logits) - 1
}

func argmax(v []float32) int {
	best := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = i
		}
	}
	return best
}

func (e *Engine) disableLayerForward(err error) {
	if err == nil {
		return
	}
	for i := range e.layers {
		if e.layers[i] != nil {
			e.layers[i].close()
		}
	}
	e.layers = nil
	e.layersInit = true
	e.layerInitErr = err
}

func (e *Engine) forwardTraining(input []uint16) ([]float32, error) {
	if e.useANE && e.ensureLayers() == nil {
		if out, err := e.forwardTrainingANE(input); err == nil {
			return out, nil
		} else {
			e.disableLayerForward(err)
		}
	}
	return e.forwardTrainingCPU(input), nil
}

func (e *Engine) forwardTrainingCPU(input []uint16) []float32 {
	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	for i := range e.mw.Layers {
		layer := e.mw.Layers[i]
		cache := &e.caches[i]
		copy(cache.x, cur)
		rmsNormCF(cache.xNorm, cache.x, layer.RMSAtt, stories.Dim, e.seq)
		linearCF(cache.q, layer.Wq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		linearCF(cache.k, layer.Wk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		linearCF(cache.v, layer.Wv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		causalAttentionCF(cache.attOut, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
		linearCF(next, layer.Wo, cache.attOut, stories.Dim, stories.Dim, e.seq)
		for j := range cache.x2 {
			cache.x2[j] = cache.x[j] + next[j]
		}
		rmsNormCF(cache.x2Norm, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
		linearCF(cache.h1, layer.W1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		linearCF(cache.h3, layer.W3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		for j := range cache.gate {
			cache.gate[j] = silu32(cache.h1[j]) * cache.h3[j]
		}
		cache.attTapsReady = true
		cache.ffnTapsReady = true
		linearCF(next, layer.W2, cache.gate, stories.Dim, stories.Hidden, e.seq)
		for j := range next {
			next[j] += cache.x2[j]
		}
		cur, next = next, cur
	}
	return cur
}

func (e *Engine) forwardTrainingANE(input []uint16) ([]float32, error) {
	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	for i := range e.layers {
		cache := &e.caches[i]
		copy(cache.x, cur)
		if err := e.layers[i].runWithTaps(next, cur, cache); err != nil {
			return nil, fmt.Errorf("storiesane step: layer %d: %w", i, err)
		}
		cur, next = next, cur
	}
	return cur, nil
}

func (e *Engine) runFinalHead(finalHidden []float32, target []uint16) (float32, error) {
	e.ensureOffload()
	clear(e.gRMS)
	clear(e.gEmbed)

	if e.off == nil || !e.off.hasRMSForward() {
		stories.RMSNorm(e.xNorm, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	} else if err := e.off.runRMSForward(e.xNorm, finalHidden); err != nil {
		e.off.disableRMSForward()
		stories.RMSNorm(e.xNorm, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	}

	loss := float32(0)
	combinedSoftmax := false
	if e.off != nil && e.off.hasClassifierForward() && e.off.hasSoftmax() {
		if err := e.off.runClassifierSoftmax(e.logits, e.xNorm); err != nil {
			e.off.disableClassifierForward()
			e.off.disableSoftmax()
		} else {
			loss = crossEntropyLossFromProbs(e.dLogits, e.logits, target, stories.Vocab, e.seq)
			combinedSoftmax = true
		}
	}
	if !combinedSoftmax {
		if e.off == nil || !e.off.hasClassifierForward() {
			stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
		} else if err := e.off.runClassifierForward(e.logits, e.xNorm); err != nil {
			e.off.disableClassifierForward()
			stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
		}
		if e.off == nil || !e.off.hasSoftmax() {
			loss = stories.CrossEntropyLoss(e.dLogits, e.logits, target, stories.Vocab, e.seq)
		} else if err := e.off.runSoftmax(e.logits); err != nil {
			e.off.disableSoftmax()
			loss = stories.CrossEntropyLoss(e.dLogits, e.logits, target, stories.Vocab, e.seq)
		} else {
			loss = crossEntropyLossFromProbs(e.dLogits, e.logits, target, stories.Vocab, e.seq)
		}
	}

	embedAsync := e.off != nil && e.off.hasClassifierBackward()
	e.embedGradDone = nil
	if embedAsync {
		e.embedGradDone = make(chan struct{})
		done := e.embedGradDone
		go func() {
			stories.MatMulGradEmbed(e.gEmbed, e.dLogits, e.xNorm, stories.Vocab, stories.Dim, e.seq)
			close(done)
		}()
	}
	if e.off == nil || !e.off.hasClassifierBackward() {
		stories.MatMulEmbedT(e.dy, e.mw.Embed, e.dLogits, stories.Vocab, stories.Dim, e.seq)
	} else if err := e.off.runClassifierBackward(e.dy, e.dLogits); err != nil {
		e.off.disableClassifierBackward()
		stories.MatMulEmbedT(e.dy, e.mw.Embed, e.dLogits, stories.Vocab, stories.Dim, e.seq)
	}
	if !embedAsync {
		stories.MatMulGradEmbed(e.gEmbed, e.dLogits, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	}
	if e.off == nil || !e.off.hasRMSBackward() {
		stories.RMSNormBackward(e.dx, e.gRMS, e.dy, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	} else if err := e.off.runRMSBackward(e.dx, e.dy, finalHidden); err != nil {
		e.off.disableRMSBackward()
		stories.RMSNormBackward(e.dx, e.gRMS, e.dy, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	} else {
		rmsNormGradWeights(e.gRMS, e.dy, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	}
	return loss, nil
}

func (e *Engine) ensureAttentionCache(layer *stories.LayerWeights, cache *layerCache) {
	if cache.attTapsReady {
		return
	}
	rmsNormCF(cache.xNorm, cache.x, layer.RMSAtt, stories.Dim, e.seq)
	linearCF(cache.q, layer.Wq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	linearCF(cache.k, layer.Wk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	linearCF(cache.v, layer.Wv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	causalAttentionCF(cache.attOut, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
	cache.attTapsReady = true
}

func (e *Engine) ensureFFNCache(layer *stories.LayerWeights, cache *layerCache) {
	if cache.ffnTapsReady {
		return
	}
	rmsNormCF(cache.x2Norm, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
	linearCF(cache.h1, layer.W1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	linearCF(cache.h3, layer.W3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	for i := range cache.gate {
		cache.gate[i] = silu32(cache.h1[i]) * cache.h3[i]
	}
	cache.ffnTapsReady = true
}

func (e *Engine) backwardFFNCPU(layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dCur, dPrev []float32) {
	e.ensureFFNCache(layer, cache)
	linearBackwardDXCF(e.gradGate, layer.W2, dCur, stories.Dim, stories.Hidden, e.seq)
	for i := range e.gradGate {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-cache.h1[i]))))
		siluGrad := sig * (1 + cache.h1[i]*(1-sig))
		e.gradH1[i] = e.gradGate[i] * cache.h3[i] * siluGrad
		e.gradH3[i] = e.gradGate[i] * (cache.h1[i] * sig)
	}
	linearBackwardDXCF(e.gradXNorm, layer.W1, e.gradH1, stories.Hidden, stories.Dim, e.seq)
	linearBackwardDXCF(dPrev, layer.W3, e.gradH3, stories.Hidden, stories.Dim, e.seq)
	for i := range e.gradXNorm {
		e.gradXNorm[i] += dPrev[i]
	}
	stories.RMSNormBackward(dPrev, grad.RMSFFN, e.gradXNorm, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
	for i := range e.gradX2 {
		e.gradX2[i] += dPrev[i]
	}
}

func (e *Engine) backwardFFNHybrid(lb *layerBackward, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dCur, dPrev []float32) error {
	e.ensureFFNCache(layer, cache)
	if err := lb.runFFN(e.gradXNorm, e.gradH1, e.gradH3, dCur, cache.h1, cache.h3); err != nil {
		return err
	}
	stories.RMSNormBackward(dPrev, grad.RMSFFN, e.gradXNorm, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
	for i := range e.gradX2 {
		e.gradX2[i] += dPrev[i]
	}
	return nil
}

func (e *Engine) backwardAttentionCPU(layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dPrev []float32) {
	e.ensureAttentionCache(layer, cache)
	linearBackwardDXCF(e.gradAtt, layer.Wo, e.gradX2, stories.Dim, stories.Dim, e.seq)
	causalAttentionBackwardCF(e.gradQ, e.gradK, e.gradV, e.gradAtt, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
	linearBackwardDXCF(e.gradXNorm, layer.Wq, e.gradQ, stories.Dim, stories.Dim, e.seq)
	linearBackwardDXCF(dPrev, layer.Wk, e.gradK, stories.Dim, stories.Dim, e.seq)
	for i := range e.gradXNorm {
		e.gradXNorm[i] += dPrev[i]
	}
	linearBackwardDXCF(dPrev, layer.Wv, e.gradV, stories.Dim, stories.Dim, e.seq)
	for i := range e.gradXNorm {
		e.gradXNorm[i] += dPrev[i]
	}
	stories.RMSNormBackward(dPrev, grad.RMSAtt, e.gradXNorm, cache.x, layer.RMSAtt, stories.Dim, e.seq)
	for i := range dPrev {
		dPrev[i] += e.gradX2[i]
	}
}

func (e *Engine) backwardAttentionHybrid(lb *layerBackward, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dPrev []float32) error {
	e.ensureAttentionCache(layer, cache)
	if err := lb.runAttention(e.gradXNorm, e.gradQ, e.gradK, e.gradV, cache.q, cache.k, cache.v, e.gradX2); err != nil {
		return err
	}
	stories.RMSNormBackward(dPrev, grad.RMSAtt, e.gradXNorm, cache.x, layer.RMSAtt, stories.Dim, e.seq)
	for i := range dPrev {
		dPrev[i] += e.gradX2[i]
	}
	return nil
}

func (e *Engine) backwardAndUpdate(input []uint16) time.Duration {
	stepT := int(e.state.AdamT) + 1
	useHybrid := false
	if e.hybridBackwardRequested {
		if err := e.ensureBackward(); err == nil {
			useHybrid = true
		}
	}
	if e.accum != nil {
		return e.backwardAndAccumulate(input, useHybrid)
	}
	return e.backwardAndApply(input, stepT, useHybrid)
}

func (e *Engine) backwardAndAccumulate(input []uint16, useHybrid bool) time.Duration {
	dCur := e.dx
	dPrev := e.gradPrev
	gradTasks := newGradTasks()
	for l := stories.NLayers - 1; l >= 0; l-- {
		layer := &e.mw.Layers[l]
		cache := &e.caches[l]
		grad := &e.accum.Layers[l]

		copy(cache.dOut, dCur)
		copy(e.gradX2, dCur)
		if useHybrid {
			if err := e.backwardFFNHybrid(e.backward[l], layer, cache, grad, dCur, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid ffn backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardFFNCPU(layer, cache, grad, dCur, dPrev)
		}
		copy(cache.dh1, e.gradH1)
		copy(cache.dh3, e.gradH3)
		copy(cache.dx2, e.gradX2)
		gradTasks.Go(func() {
			accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
			accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
			accumLinearGradCF(grad.W3, cache.dh3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		})

		if useHybrid {
			if err := e.backwardAttentionHybrid(e.backward[l], layer, cache, grad, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid attention backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardAttentionCPU(layer, cache, grad, dPrev)
		}
		copy(cache.dq, e.gradQ)
		copy(cache.dk, e.gradK)
		copy(cache.dv, e.gradV)
		gradTasks.Go(func() {
			accumLinearGradCF(grad.Wo, cache.dx2, cache.attOut, stories.Dim, stories.Dim, e.seq)
			accumLinearGradCF(grad.Wq, cache.dq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
			accumLinearGradCF(grad.Wk, cache.dk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
			accumLinearGradCF(grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		})
		dCur, dPrev = dPrev, dCur
	}
	gradTasks.Wait()

	if e.embedGradDone != nil {
		<-e.embedGradDone
		e.embedGradDone = nil
	}
	stories.EmbedBackward(e.gEmbed, dCur, input, stories.Dim, e.seq)
	addSlice(e.accum.RMSFinal, e.gRMS)
	addSlice(e.accum.Embed, e.gEmbed)
	e.state.PendingSteps++
	if int(e.state.PendingSteps) >= e.accumSteps {
		return e.flushPending()
	}
	return 0
}

func (e *Engine) backwardAndApply(input []uint16, stepT int, useHybrid bool) time.Duration {
	dCur := e.dx
	dPrev := e.gradPrev
	for l := stories.NLayers - 1; l >= 0; l-- {
		layer := &e.mw.Layers[l]
		cache := &e.caches[l]
		grad := &e.layerGrad
		clearLayerGrad(grad)
		gradTasks := newGradTasks()

		copy(e.gradX2, dCur)
		gradTasks.Go(func() {
			accumLinearGradCF(grad.W2, dCur, cache.gate, stories.Dim, stories.Hidden, e.seq)
		})
		if useHybrid {
			if err := e.backwardFFNHybrid(e.backward[l], layer, cache, grad, dCur, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid ffn backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardFFNCPU(layer, cache, grad, dCur, dPrev)
		}
		gradTasks.Go(func() {
			accumLinearGradCF(grad.W1, e.gradH1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		})
		gradTasks.Go(func() {
			accumLinearGradCF(grad.W3, e.gradH3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		})

		gradTasks.Go(func() {
			accumLinearGradCF(grad.Wo, e.gradX2, cache.attOut, stories.Dim, stories.Dim, e.seq)
		})
		if useHybrid {
			if err := e.backwardAttentionHybrid(e.backward[l], layer, cache, grad, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid attention backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardAttentionCPU(layer, cache, grad, dPrev)
		}
		gradTasks.Go(func() {
			accumLinearGradCF(grad.Wq, e.gradQ, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		})
		gradTasks.Go(func() {
			accumLinearGradCF(grad.Wk, e.gradK, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		})
		gradTasks.Go(func() {
			accumLinearGradCF(grad.Wv, e.gradV, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		})
		gradTasks.Wait()
		applyLayerAdam(layer, grad, &e.opt.Layers[l], stepT, e.lr)
		dCur, dPrev = dPrev, dCur
	}

	if e.embedGradDone != nil {
		<-e.embedGradDone
		e.embedGradDone = nil
	}
	stories.EmbedBackward(e.gEmbed, dCur, input, stories.Dim, e.seq)
	stories.AdamUpdate(e.mw.RMSFinal, e.gRMS, &e.opt.RMSFinal, stepT, e.lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(e.mw.Embed, e.gEmbed, &e.opt.Embed, stepT, e.lr, 0.9, 0.999, 1e-8)
	compileDur := e.refreshANERuntimeForWeights()
	e.state.AdamT = uint32(stepT)
	e.state.CumBatches++
	return compileDur
}

func applyLayerAdam(dst *stories.LayerWeights, grad *stories.LayerWeights, st *stories.LayerOptimState, t int, lr float32) {
	stories.AdamUpdate(dst.Wq, grad.Wq, &st.Wq, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.Wk, grad.Wk, &st.Wk, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.Wv, grad.Wv, &st.Wv, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.Wo, grad.Wo, &st.Wo, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.W1, grad.W1, &st.W1, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.W2, grad.W2, &st.W2, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.W3, grad.W3, &st.W3, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.RMSAtt, grad.RMSAtt, &st.RMSAtt, t, lr, 0.9, 0.999, 1e-8)
	stories.AdamUpdate(dst.RMSFFN, grad.RMSFFN, &st.RMSFFN, t, lr, 0.9, 0.999, 1e-8)
}
