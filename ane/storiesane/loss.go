package storiesane

import "math"

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
