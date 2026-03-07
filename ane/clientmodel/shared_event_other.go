//go:build !darwin

package clientmodel

import (
	"fmt"
	"time"
)

// SharedEvent is unavailable on non-darwin platforms.
type SharedEvent struct{}

// NewSharedEvent always fails on non-darwin platforms.
func NewSharedEvent() (*SharedEvent, error) {
	return nil, fmt.Errorf("shared events require darwin")
}

// SharedEventFromPort always fails on non-darwin platforms.
func SharedEventFromPort(uint32) (*SharedEvent, error) {
	return nil, fmt.Errorf("shared events require darwin")
}

// Port always returns zero on non-darwin platforms.
func (e *SharedEvent) Port() uint32 { return 0 }

// SignaledValue always fails on non-darwin platforms.
func (e *SharedEvent) SignaledValue() (uint64, error) {
	return 0, fmt.Errorf("shared events require darwin")
}

// Signal always fails on non-darwin platforms.
func (e *SharedEvent) Signal(uint64) error {
	return fmt.Errorf("shared events require darwin")
}

// Wait always fails on non-darwin platforms.
func (e *SharedEvent) Wait(uint64, time.Duration) (bool, error) {
	return false, fmt.Errorf("shared events require darwin")
}

// Close is a no-op on non-darwin platforms.
func (e *SharedEvent) Close() error { return nil }

// SignalEventCPU always fails on non-darwin platforms.
func SignalEventCPU(uint32, uint64) error {
	return fmt.Errorf("shared events require darwin")
}

// WaitEventCPU always fails on non-darwin platforms.
func WaitEventCPU(uint32, uint64, time.Duration) (bool, error) {
	return false, fmt.Errorf("shared events require darwin")
}
