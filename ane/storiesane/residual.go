package storiesane

import (
	"math"

	"github.com/maderix/ANE/ane/stories"
)

var layerResidualScale = float32(1.0 / math.Sqrt(2.0*float64(stories.NLayers)))

func blendResidualInPlace(sum, base []float32) {
	blendResidualInPlaceAccel(sum, base, layerResidualScale)
}

func blendResidual(dst, base, branch []float32) {
	blendResidualAccel(dst, base, branch, layerResidualScale)
}

func addScaledResidual(dst, base, branch []float32) {
	addScaledResidualAccel(dst, base, branch, layerResidualScale)
}

func scaleInto(dst, src []float32, scale float32) {
	scaleIntoAccel(dst, src, scale)
}
