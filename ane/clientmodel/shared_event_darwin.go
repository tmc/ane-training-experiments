//go:build darwin

package clientmodel

import (
	"fmt"
	"time"

	appiosurface "github.com/tmc/apple/iosurface"
)

// SharedEvent wraps IOSurfaceSharedEvent for CPU signal/wait control.
type SharedEvent struct {
	ev appiosurface.IOSurfaceSharedEvent
}

// NewSharedEvent creates a new shared event.
func NewSharedEvent() (*SharedEvent, error) {
	ev := appiosurface.NewIOSurfaceSharedEventWithOptions(0)
	if ev.GetID() == 0 {
		return nil, fmt.Errorf("new shared event: create failed")
	}
	return &SharedEvent{ev: ev}, nil
}

// SharedEventFromPort binds an existing shared-event Mach port.
func SharedEventFromPort(port uint32) (*SharedEvent, error) {
	if port == 0 {
		return nil, fmt.Errorf("shared event from port: port is zero")
	}
	ev := appiosurface.NewIOSurfaceSharedEventWithMachPort(port)
	if ev.GetID() == 0 {
		return nil, fmt.Errorf("shared event from port: bind port %d failed", port)
	}
	return &SharedEvent{ev: ev}, nil
}

// Port returns the underlying Mach port name.
func (e *SharedEvent) Port() uint32 {
	if e == nil || e.ev.GetID() == 0 {
		return 0
	}
	return e.ev.EventPort()
}

// SignaledValue returns the current event value.
func (e *SharedEvent) SignaledValue() (uint64, error) {
	if e == nil || e.ev.GetID() == 0 {
		return 0, fmt.Errorf("shared event signaled value: event is closed")
	}
	return e.ev.SignaledValue(), nil
}

// Signal sets the event value from CPU.
func (e *SharedEvent) Signal(value uint64) error {
	if e == nil || e.ev.GetID() == 0 {
		return fmt.Errorf("shared event signal: event is closed")
	}
	e.ev.SetSignaledValue(value)
	return nil
}

// Wait blocks until the event reaches value or timeout.
func (e *SharedEvent) Wait(value uint64, timeout time.Duration) (bool, error) {
	if e == nil || e.ev.GetID() == 0 {
		return false, fmt.Errorf("shared event wait: event is closed")
	}
	if timeout < 0 {
		return false, fmt.Errorf("shared event wait: timeout must be >= 0")
	}
	timeoutMS := uint64(timeout / time.Millisecond)
	if timeout > 0 && timeoutMS == 0 {
		timeoutMS = 1
	}
	return e.ev.WaitUntilSignaledValueTimeoutMS(value, timeoutMS), nil
}

// Close releases the underlying event object.
func (e *SharedEvent) Close() error {
	if e == nil {
		return nil
	}
	if e.ev.GetID() != 0 {
		e.ev.Release()
		e.ev = appiosurface.IOSurfaceSharedEvent{}
	}
	return nil
}

// SignalEventCPU sets the event value for an existing Mach port from CPU.
func SignalEventCPU(port uint32, value uint64) error {
	ev, err := SharedEventFromPort(port)
	if err != nil {
		return err
	}
	defer ev.Close()
	return ev.Signal(value)
}

// WaitEventCPU waits on an existing Mach-port event from CPU.
func WaitEventCPU(port uint32, value uint64, timeout time.Duration) (bool, error) {
	ev, err := SharedEventFromPort(port)
	if err != nil {
		return false, err
	}
	defer ev.Close()
	return ev.Wait(value, timeout)
}
