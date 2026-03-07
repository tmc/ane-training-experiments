//go:build !darwin

package pipeline

import (
	"fmt"
	"time"
)

// Options configures an asymmetric pipeline runner.
type Options struct {
	ModelPath     string
	ModelKey      string
	InputBytes    uint32
	OutputBytes   uint32
	WaitTimeoutMS uint32
}

// Runner is unavailable on non-darwin platforms.
type Runner struct{}

// Open always fails on non-darwin platforms.
func Open(Options) (*Runner, error) {
	return nil, fmt.Errorf("asymmetric pipeline is only supported on darwin")
}

// Close is a no-op on non-darwin platforms.
func (r *Runner) Close() error { return nil }

// WaitPort always returns zero on non-darwin platforms.
func (r *Runner) WaitPort() uint32 { return 0 }

// SignalPort always returns zero on non-darwin platforms.
func (r *Runner) SignalPort() uint32 { return 0 }

// SignalWaitFromCPU always fails on non-darwin platforms.
func (r *Runner) SignalWaitFromCPU(uint64) error {
	return fmt.Errorf("asymmetric pipeline is only supported on darwin")
}

// WaitForSignal always fails on non-darwin platforms.
func (r *Runner) WaitForSignal(uint64, time.Duration) (bool, error) {
	return false, fmt.Errorf("asymmetric pipeline is only supported on darwin")
}

// Eval always fails on non-darwin platforms.
func (r *Runner) Eval(uint64, uint64, []float32, []float32) error {
	return fmt.Errorf("asymmetric pipeline is only supported on darwin")
}

// StepCPUOnly always fails on non-darwin platforms.
func (r *Runner) StepCPUOnly(uint64, uint64, []float32, []float32) error {
	return fmt.Errorf("asymmetric pipeline is only supported on darwin")
}
