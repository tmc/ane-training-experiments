//go:build darwin

package clientmodel

import (
	"testing"
	"time"
)

func TestSharedEventCPUSignalAndWait(t *testing.T) {
	ev, err := NewSharedEvent()
	if err != nil {
		t.Fatalf("NewSharedEvent: %v", err)
	}
	defer func() { _ = ev.Close() }()

	const value = uint64(3)
	if err := ev.Signal(value); err != nil {
		t.Fatalf("Signal: %v", err)
	}
	ok, err := ev.Wait(value, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if !ok {
		t.Fatalf("Wait timeout for value=%d", value)
	}
	got, err := ev.SignaledValue()
	if err != nil {
		t.Fatalf("SignaledValue: %v", err)
	}
	if got < value {
		t.Fatalf("SignaledValue=%d want >= %d", got, value)
	}
}

func TestSharedEventCPUHelpersByPort(t *testing.T) {
	ev, err := NewSharedEvent()
	if err != nil {
		t.Fatalf("NewSharedEvent: %v", err)
	}
	defer func() { _ = ev.Close() }()

	port := ev.Port()
	if port == 0 {
		t.Fatalf("Port=0")
	}

	const value = uint64(7)
	if err := SignalEventCPU(port, value); err != nil {
		t.Fatalf("SignalEventCPU: %v", err)
	}
	ok, err := WaitEventCPU(port, value, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("WaitEventCPU: %v", err)
	}
	if !ok {
		t.Fatalf("WaitEventCPU timeout for value=%d", value)
	}
}

func TestSharedEventFromPortValidation(t *testing.T) {
	if _, err := SharedEventFromPort(0); err == nil {
		t.Fatalf("SharedEventFromPort(0) error=nil, want error")
	}
	if err := SignalEventCPU(0, 1); err == nil {
		t.Fatalf("SignalEventCPU(0) error=nil, want error")
	}
	if _, err := WaitEventCPU(0, 1, 10*time.Millisecond); err == nil {
		t.Fatalf("WaitEventCPU(0) error=nil, want error")
	}
}
