//go:build darwin

package pipeline

import "testing"

func TestOpenValidation(t *testing.T) {
	if _, err := Open(Options{}); err == nil {
		t.Fatalf("Open with empty options succeeded; want error")
	}
	if _, err := Open(Options{ModelPath: "/tmp/model.mlmodelc"}); err == nil {
		t.Fatalf("Open with zero bytes succeeded; want error")
	}
}

func TestClosedRunnerErrors(t *testing.T) {
	var r Runner
	if err := r.SignalWaitFromCPU(1); err == nil {
		t.Fatalf("SignalWaitFromCPU on closed runner succeeded; want error")
	}
	if _, err := r.WaitForSignal(1, 0); err == nil {
		t.Fatalf("WaitForSignal on closed runner succeeded; want error")
	}
	if err := r.Eval(1, 1, nil, nil); err == nil {
		t.Fatalf("Eval on closed runner succeeded; want error")
	}
}
