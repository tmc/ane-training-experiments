//go:build !darwin || !cgo

package storiesane

import (
	"math"

	"github.com/maderix/ANE/ane/stories"
)

func rmsNormCFWithRRMSImpl(out, rrms, x, w []float32, dim, seq int) {
	parallelForCF(seq, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			scale := float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
			if rrms != nil {
				rrms[t] = scale
			}
			for i := 0; i < dim; i++ {
				out[i*seq+t] = x[i*seq+t] * scale * w[i]
			}
		}
	})
}

func scaleSliceAccel(v []float32, scale float32) {
	for i := range v {
		v[i] *= scale
	}
}

func addSliceAccel(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func scaleIntoAccel(dst, src []float32, scale float32) {
	for i := range dst {
		dst[i] = src[i] * scale
	}
}

func blendResidualInPlaceAccel(sum, base []float32, scale float32) {
	for i := range sum {
		sum[i] = base[i] + (sum[i]-base[i])*scale
	}
}

func blendResidualAccel(dst, base, branch []float32, scale float32) {
	for i := range dst {
		dst[i] = base[i] + (branch[i]-base[i])*scale
	}
}

func addScaledResidualAccel(dst, base, branch []float32, scale float32) {
	for i := range dst {
		dst[i] = base[i] + scale*branch[i]
	}
}

func siluGateForwardAccel(gate, h1, h3 []float32) {
	for i := range gate {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-h1[i]))))
		gate[i] = (h1[i] * sig) * h3[i]
	}
}

func siluBackwardAccel(dh1, dh3, dGate, h1, h3 []float32) {
	for i := range dh1 {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-h1[i]))))
		siluGrad := sig * (1 + h1[i]*(1-sig))
		dh1[i] = dGate[i] * h3[i] * siluGrad
		dh3[i] = dGate[i] * (h1[i] * sig)
	}
}

func softmaxStridedCEAccel(dLogits, logits []float32, target, vocab, stride, t int) float32 {
	mx := logits[t]
	for i := 1; i < vocab; i++ {
		v := logits[i*stride+t]
		if v > mx {
			mx = v
		}
	}
	sum := 0.0
	for i := 0; i < vocab; i++ {
		e := math.Exp(float64(logits[i*stride+t] - mx))
		dLogits[i*stride+t] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := 0; i < vocab; i++ {
		dLogits[i*stride+t] *= inv
	}
	if target < 0 || target >= vocab {
		for i := 0; i < vocab; i++ {
			dLogits[i*stride+t] = 0
		}
		return 0
	}
	p := dLogits[target*stride+t]
	if p < 1e-10 {
		p = 1e-10
	}
	dLogits[target*stride+t] -= 1
	return float32(-math.Log(float64(p)))
}

func softmaxStridedCEBatchAccel(dLogits, logits []float32, targets []uint16, vocab, stride, tStart, tEnd int) (float64, int) {
	totalLoss := 0.0
	totalValid := 0
	for t := tStart; t < tEnd; t++ {
		loss := softmaxStridedCEAccel(dLogits, logits, int(targets[t]), vocab, stride, t)
		tgt := int(targets[t])
		if tgt >= 0 && tgt < vocab {
			totalLoss += float64(loss)
			totalValid++
		}
	}
	return totalLoss, totalValid
}

func softmaxRowAccel(out, in []float32) {
	maxv := in[0]
	for i := 1; i < len(in); i++ {
		if in[i] > maxv {
			maxv = in[i]
		}
	}
	sum := 0.0
	for i := range in {
		e := math.Exp(float64(in[i] - maxv))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
}

func rmsNormBackwardAccel(dx, dw, dy, x, w []float32, dim, seq int) {
	rmsNormBackwardPooled(dx, dw, dy, x, w, dim, seq)
}

func transposeClassifierForwardTileAccel(dst, embed []float32, start, size int) {
	dim := stories.Dim
	for d := 0; d < dim; d++ {
		row := dst[d*size : (d+1)*size]
		for i := 0; i < size; i++ {
			row[i] = embed[(start+i)*dim+d]
		}
	}
}

