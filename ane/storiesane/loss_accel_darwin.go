//go:build darwin && cgo

package storiesane

// crossEntropyLossAccel computes softmax + cross-entropy loss with vectorized
// exp() via Accelerate's vvexpf. Uses dispatch_apply for parallelism.
// Layout is channel-first: logits[vocab_idx * seq + token_idx].
func crossEntropyLossAccel(dLogits, logits []float32, targets []uint16, v, s int) float32 {
	if v <= 0 || s <= 0 {
		return 0
	}
	if len(logits) < v*s || len(dLogits) < v*s || len(targets) < s {
		for i := range dLogits {
			dLogits[i] = 0
		}
		return 0
	}

	totalLoss, totalValid := crossEntropyLossParallelAccel(dLogits, logits, targets, v, s)
	if totalValid == 0 {
		return 0
	}

	invValid := float32(1.0 / float64(totalValid))
	scaleSliceAccel(dLogits[:v*s], invValid)

	return float32(totalLoss / float64(totalValid))
}

