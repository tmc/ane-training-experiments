package storiesane

import (
	"math"

	"github.com/maderix/ANE/ane/stories"
)

var layerResidualScale = float32(1.0 / math.Sqrt(2.0*float64(stories.NLayers)))

func blendResidualInPlace(sum, base []float32) {
	for i := range sum {
		sum[i] = base[i] + (sum[i]-base[i])*layerResidualScale
	}
}

func addScaledResidual(dst, base, branch []float32) {
	for i := range dst {
		dst[i] = base[i] + layerResidualScale*branch[i]
	}
}

func scaleInto(dst, src []float32, scale float32) {
	for i := range dst {
		dst[i] = src[i] * scale
	}
}
