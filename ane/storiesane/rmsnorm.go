package storiesane

import "math"

func rmsNormGradWeights(dw, dy, x, w []float32, d, s int) {
	for t := 0; t < s; t++ {
		sum := 0.0
		for i := 0; i < d; i++ {
			v := float64(x[i*s+t])
			sum += v * v
		}
		rrms := 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		for i := 0; i < d; i++ {
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * rrms)
		}
	}
}
