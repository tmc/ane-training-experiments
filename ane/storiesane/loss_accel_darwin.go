//go:build darwin && cgo

package storiesane

import (
	"runtime"
	"sync"
)

// crossEntropyLossAccel computes softmax + cross-entropy loss with vectorized
// exp() via Accelerate's vvexpf. Replaces stories.CrossEntropyLoss on darwin.
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

	type shard struct {
		loss  float64
		valid int
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > s {
		workers = s
	}
	shards := make([]shard, workers)
	chunk := (s + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		if start >= s {
			break
		}
		end := start + chunk
		if end > s {
			end = s
		}
		wg.Add(1)
		go func(w, start, end int) {
			defer wg.Done()
			shards[w].loss, shards[w].valid = softmaxStridedCEBatchAccel(dLogits, logits, targets, v, s, start, end)
		}(w, start, end)
	}
	wg.Wait()

	totalLoss := 0.0
	totalValid := 0
	for _, sh := range shards {
		totalLoss += sh.loss
		totalValid += sh.valid
	}
	if totalValid == 0 {
		return 0
	}

	invValid := float32(1.0 / float64(totalValid))
	scaleSliceAccel(dLogits[:v*s], invValid)

	return float32(totalLoss / float64(totalValid))
}

