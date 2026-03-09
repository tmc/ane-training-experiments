//go:build darwin

package pipeline

import "testing"

func TestOpenEvalValidation(t *testing.T) {
	if _, err := OpenEval(EvalOptions{}); err == nil {
		t.Fatalf("OpenEval with empty options succeeded; want error")
	}
	if _, err := OpenEval(EvalOptions{
		ModelPath:   "/tmp/model.mlmodelc",
		ModelType:   "kANEFModelMIL",
		InputBytes:  0,
		OutputBytes: 1,
	}); err == nil {
		t.Fatalf("OpenEval with zero input bytes on client fallback succeeded; want error")
	}
	if _, err := OpenEval(EvalOptions{
		ModelPath:   "/tmp/model.mlmodelc",
		ModelType:   "kANEFModelMIL",
		InputBytes:  1,
		OutputBytes: 0,
	}); err == nil {
		t.Fatalf("OpenEval with zero output bytes on client fallback succeeded; want error")
	}
}
