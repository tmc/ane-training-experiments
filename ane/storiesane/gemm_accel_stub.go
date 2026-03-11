//go:build !darwin || !cgo

package storiesane

func accumLinearGradCFAccelerate(dW, dy, x []float32, outCh, inCh, seq int) bool {
	return false
}

func linearBackwardDXCFAccelerate(dx, w, dy []float32, outCh, inCh, seq int) bool {
	return false
}
