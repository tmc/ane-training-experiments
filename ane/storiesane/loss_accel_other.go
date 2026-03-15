//go:build !darwin || !cgo

package storiesane

import (
	"github.com/maderix/ANE/ane/stories"
)

func crossEntropyLossAccel(dLogits, logits []float32, targets []uint16, v, s int) float32 {
	return stories.CrossEntropyLoss(dLogits, logits, targets, v, s)
}
