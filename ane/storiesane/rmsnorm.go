package storiesane

import "math"

func rmsNormGradWeights(dw, dy, x, w []float32, d, s int) {
	rmsNormGradWeightsRange(dw, dy, x, nil, d, s, 0, s)
}

// rmsNormBackwardPooled computes both dx and dw for RMS norm backward.
// On darwin, uses a C function with dispatch_apply for parallelism,
// avoiding Go goroutine overhead. Falls back to pooled Go goroutines
// on other platforms.
func rmsNormBackwardPooled(dx, dw, dy, x, w []float32, d, s int) {
	rmsNormBackwardAccel(dx, dw, dy, x, w, d, s)
}

func rmsNormBackwardRange(dx, dw, dy, x, w []float32, d, s, start, end int) {
	for t := start; t < end; t++ {
		sum := 0.0
		for i := 0; i < d; i++ {
			v := float64(x[i*s+t])
			sum += v * v
		}
		rrms := 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		rrms2InvD := (rrms * rrms) / float64(d)
		dot := 0.0
		for i := 0; i < d; i++ {
			dot += float64(dy[i*s+t] * x[i*s+t] * w[i])
		}
		for i := 0; i < d; i++ {
			v := float64(dy[i*s+t]) - float64(x[i*s+t])*dot*rrms2InvD
			dx[i*s+t] = float32(v * rrms * float64(w[i]))
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * rrms)
		}
	}
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
