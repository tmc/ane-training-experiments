//go:build darwin

package pipeline

import "testing"

func TestOpenEvalValidation(t *testing.T) {
	if _, err := OpenEval(EvalOptions{}); err == nil {
		t.Fatalf("OpenEval with empty options succeeded; want error")
	}
}
