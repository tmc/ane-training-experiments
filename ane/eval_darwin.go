//go:build darwin

package ane

import (
	"fmt"

	"github.com/maderix/ANE/ane/pipeline"
)

type evalRunner interface {
	Close() error
	EvalBytes([]byte, []byte) error
	EvalF32([]float32, []float32) error
}

// Evaluator provides a high-level model eval entrypoint.
type Evaluator struct {
	r evalRunner
}

// OpenEvaluator opens a model evaluator backed by x/ane.
func OpenEvaluator(opts EvalOptions) (*Evaluator, error) {
	r, err := pipeline.OpenEval(pipeline.EvalOptions{
		ModelPath:        opts.ModelPath,
		ModelPackagePath: opts.ModelPackagePath,
		ModelKey:         opts.ModelKey,
		QoS:              opts.QoS,
		InputBytes:       opts.InputBytes,
		OutputBytes:      opts.OutputBytes,
	})
	if err != nil {
		return nil, err
	}
	return &Evaluator{r: r}, nil
}

// Close releases evaluator resources.
func (e *Evaluator) Close() error {
	if e == nil {
		return nil
	}
	if e.r == nil {
		return nil
	}
	err := e.r.Close()
	e.r = nil
	return err
}

// EvalBytes runs one evaluation with byte input/output.
func (e *Evaluator) EvalBytes(input, output []byte) error {
	if e == nil || e.r == nil {
		return fmt.Errorf("ane evaluator is closed")
	}
	return e.r.EvalBytes(input, output)
}

// EvalF32 runs one evaluation with float32 input/output.
func (e *Evaluator) EvalF32(input, output []float32) error {
	if e == nil || e.r == nil {
		return fmt.Errorf("ane evaluator is closed")
	}
	return e.r.EvalF32(input, output)
}
