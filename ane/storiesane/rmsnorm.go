package storiesane

import (
	"math"
)

func rmsNormGradWeights(dw, dy, x, w []float32, d, s int) {
	rmsNormGradWeightsWithRRMS(dw, dy, x, nil, d, s)
}

func rmsNormGradWeightsWithRRMS(dw, dy, x, rrms []float32, d, s int) {
	rmsNormGradWeightsRange(dw, dy, x, rrms, d, s, 0, s)
}

func rmsNormGradWeightsRange(dw, dy, x, rrms []float32, d, s, start, end int) {
	for t := start; t < end; t++ {
		scale := 0.0
		if len(rrms) > t {
			scale = float64(rrms[t])
		} else {
			sum := 0.0
			for i := 0; i < d; i++ {
				v := float64(x[i*s+t])
				sum += v * v
			}
			scale = 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		}
		for i := 0; i < d; i++ {
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * scale)
		}
	}
}
