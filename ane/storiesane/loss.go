package storiesane

import (
	"math"
	"runtime"
	"sync"
)

func crossEntropyLossFromProbs(dLogits, probs []float32, targets []uint16, vocab, seq int) float32 {
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
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || seq < workers*4 {
		loss, valid = crossEntropyLossFromProbsRange(dLogits, probs, targets, vocab, seq, 0, seq)
		if valid == 0 {
			return 0
		}
		scale := float32(1.0 / float64(valid))
		parallelForCF(seq, func(start, end int) {
			scaleCrossEntropyGradRange(dLogits, targets, vocab, seq, scale, start, end)
		})
		return float32(loss / float64(valid))
	}
	if workers > seq {
		workers = seq
	}
	chunk := (seq + workers - 1) / workers
	type shard struct {
		loss  float64
		valid int
	}
	shards := make([]shard, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		if start >= seq {
			break
		}
		end := start + chunk
		if end > seq {
			end = seq
		}
		wg.Add(1)
		go func(start, end, worker int) {
			defer wg.Done()
			shards[worker].loss, shards[worker].valid = crossEntropyLossFromProbsRange(dLogits, probs, targets, vocab, seq, start, end)
		}(start, end, worker)
	}
	wg.Wait()
	for _, shard := range shards {
		loss += shard.loss
		valid += shard.valid
	}
	if valid == 0 {
		return 0
	}
	scale := float32(1.0 / float64(valid))
	parallelForCF(seq, func(start, end int) {
		scaleCrossEntropyGradRange(dLogits, targets, vocab, seq, scale, start, end)
	})
	return float32(loss / float64(valid))
}

func crossEntropyLossFromProbsRange(dLogits, probs []float32, targets []uint16, vocab, seq, start, end int) (float64, int) {
	loss := 0.0
	valid := 0
	for t := start; t < end; t++ {
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
	return loss, valid
}

func scaleCrossEntropyGradRange(dLogits []float32, targets []uint16, vocab, seq int, scale float32, start, end int) {
	for t := start; t < end; t++ {
		tgt := int(targets[t])
		if tgt < 0 || tgt >= vocab {
			continue
		}
		for v := 0; v < vocab; v++ {
			dLogits[v*seq+t] *= scale
		}
	}
}
