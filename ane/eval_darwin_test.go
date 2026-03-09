//go:build darwin

package ane

import "testing"

func TestOpenEvaluatorValidation(t *testing.T) {
	if _, err := OpenEvaluator(EvalOptions{}); err == nil {
		t.Fatalf("OpenEvaluator with empty options succeeded; want error")
	}
}

func TestEvaluatorNilSafety(t *testing.T) {
	var e *Evaluator
	if err := e.Close(); err != nil {
		t.Fatalf("Close(nil): %v", err)
	}
	if err := e.EvalBytes(nil, nil); err == nil {
		t.Fatalf("EvalBytes(nil) error=nil, want error")
	}
	if err := e.EvalF32(nil, nil); err == nil {
		t.Fatalf("EvalF32(nil) error=nil, want error")
	}
}
