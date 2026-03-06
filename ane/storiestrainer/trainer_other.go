//go:build !darwin

package storiestrainer

import "fmt"

// Trainer is unavailable on non-darwin platforms.
type Trainer struct{}

// Open always fails on non-darwin platforms.
func Open(Options) (*Trainer, error) {
	return nil, fmt.Errorf("stories trainer is only supported on darwin")
}

// Step always fails on non-darwin platforms.
func (t *Trainer) Step() (StepStats, error) {
	return StepStats{}, fmt.Errorf("stories trainer is only supported on darwin")
}

// SaveCheckpoint always fails on non-darwin platforms.
func (t *Trainer) SaveCheckpoint(string) error {
	return fmt.Errorf("stories trainer is only supported on darwin")
}

// LoadCheckpoint always fails on non-darwin platforms.
func (t *Trainer) LoadCheckpoint(string) error {
	return fmt.Errorf("stories trainer is only supported on darwin")
}

// Diagnostics returns zero values on non-darwin platforms.
func (t *Trainer) Diagnostics() Diagnostics { return Diagnostics{} }

// Backend reports the trainer backend on non-darwin platforms.
func (t *Trainer) Backend() string { return "unsupported" }

// Close is a no-op on non-darwin platforms.
func (t *Trainer) Close() error { return nil }
