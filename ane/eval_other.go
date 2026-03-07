//go:build !darwin

package ane

import "fmt"

// Evaluator is unavailable on non-darwin platforms.
type Evaluator struct{}

// OpenEvaluator always fails on non-darwin platforms.
func OpenEvaluator(EvalOptions) (*Evaluator, error) {
	return nil, fmt.Errorf("ane evaluator requires darwin")
}

// Close is a no-op on non-darwin platforms.
func (e *Evaluator) Close() error { return nil }

// EspressoEnabled always reports false on non-darwin platforms.
func (e *Evaluator) EspressoEnabled() bool { return false }

// EvalBytes always fails on non-darwin platforms.
func (e *Evaluator) EvalBytes([]byte, []byte) error {
	return fmt.Errorf("ane evaluator requires darwin")
}

// EvalF32 always fails on non-darwin platforms.
func (e *Evaluator) EvalF32([]float32, []float32) error {
	return fmt.Errorf("ane evaluator requires darwin")
}
